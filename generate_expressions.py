"""
Generate boolean expressions using the trained GPT model.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import random
from pathlib import Path

import torch

from data.dataset import VOCAB, ID_TO_TOKEN, tokenize
from data.expression_dataset import BOS_ID, EOS_ID, PAD_ID
from data.conditional_dataset import RESULT_TRUE_ID, RESULT_FALSE_ID
from model.decoder_gpt import ExpressionGPT


def load_model(checkpoint_path: str, device: torch.device, max_length: int = 64) -> ExpressionGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ExpressionGPT(
        vocab_size=len(VOCAB),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        max_length=max_length,
        dropout=0.0,
        pad_id=PAD_ID,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def generate(
    model: ExpressionGPT,
    device: torch.device,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int = None,
    prefix: str = None,
    result: bool = None,
) -> str:
    """
    Generate a single expression autoregressively.
    If result is True/False, use conditional format [BOS] [RESULT_X] and generate.
    If prefix is provided, start from BOS + prefix tokens and complete the sequence.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    if result is not None:
        result_id = RESULT_TRUE_ID if result else RESULT_FALSE_ID
        generated = [BOS_ID, result_id]
    elif prefix:
        prefix_ids = tokenize(prefix.strip())
        generated = [BOS_ID] + prefix_ids
    else:
        generated = [BOS_ID]
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]

            # Mask pad, BOS (don't generate these mid-sequence)
            next_token_logits[PAD_ID] = float("-inf")
            next_token_logits[BOS_ID] = float("-inf")

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, -1]] = float("-inf")
                probs = torch.softmax(next_token_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = next_token_logits.argmax().item()

            generated.append(next_id)
            if next_id == EOS_ID:
                break

    # strip BOS, EOS, and result token (if conditional); include prefix tokens in output
    if result is not None:
        # Skip BOS and RESULT_X, keep expression tokens
        tokens = [ID_TO_TOKEN[i] for i in generated[2:] if i != EOS_ID]
    else:
        tokens = [ID_TO_TOKEN[i] for i in generated[1:] if i != EOS_ID]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator/latest.pt")
    parser.add_argument("--n", type=int, default=10, help="Number of expressions to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None, help="Token sequence to complete (e.g. 'True AND ')")
    parser.add_argument("--result", type=str, default=None, choices=["True", "False"],
        help="Generate expressions that evaluate to this result (requires conditional checkpoint)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train the generator first: python train_generator.py --epochs 50")
        return

    result_bool = None
    if args.result:
        result_bool = args.result == "True"
        # Use conditional checkpoint by default when --result is specified
        if args.checkpoint == "checkpoints/generator/latest.pt":
            args.checkpoint = "checkpoints/generator_conditional/latest.pt"

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    n = 1 if (args.prefix or args.result) and args.n == 10 else args.n
    if args.result:
        print(f"Generating expressions that evaluate to {args.result} (temperature={args.temperature})\n")
    elif args.prefix:
        print(f"Completing prefix: '{args.prefix}' (temperature={args.temperature})\n")
    else:
        print(f"Generating {n} expressions (temperature={args.temperature})\n")

    for i in range(n):
        expr = generate(
            model, device,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            prefix=args.prefix,
            result=result_bool,
        )
        if args.prefix or args.result:
            print(f"  {expr}")
        else:
            print(f"  {i+1}. {expr}")

    if not args.prefix:
        print("\nNote: Some outputs may be invalid. Train longer for better grammar.")


if __name__ == "__main__":
    main()
