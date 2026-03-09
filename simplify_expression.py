"""
Simplify boolean expressions using the trained seq2seq model.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from data.simplifier_vocab import (
    VOCAB,
    ID_TO_TOKEN,
    tokenize,
    detokenize,
    BOS_ID,
    EOS_ID,
    PAD_ID,
)
from model.seq2seq import SimplifierTransformer


def load_model(checkpoint_path: str, device: torch.device, max_length: int = 64) -> SimplifierTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SimplifierTransformer(
        vocab_size=len(VOCAB),
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        max_length=max_length,
        dropout=0.0,
        pad_id=PAD_ID,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def simplify(
    model: SimplifierTransformer,
    expression: str,
    device: torch.device,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    """
    Simplify a boolean expression autoregressively.
    """
    src_ids = tokenize(expression.strip())
    if len(src_ids) == 0:
        return ""

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = [BOS_ID]
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src_tensor, tgt_tensor)
            next_token_logits = logits[0, -1, :]

            next_token_logits[PAD_ID] = float("-inf")
            next_token_logits[BOS_ID] = float("-inf")

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = next_token_logits.argmax().item()

            generated.append(next_id)
            if next_id == EOS_ID:
                break

    # Strip BOS and EOS from generated sequence
    output_ids = [i for i in generated[1:] if i != EOS_ID and i != PAD_ID]
    return detokenize(output_ids, strip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Simplify boolean expressions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simplifier/best.pt")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, >0 = sampling")
    parser.add_argument("--interactive", action="store_true", help="Enter expressions interactively")
    parser.add_argument("expression", type=str, nargs="*", help="Expression(s) to simplify")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Generate data: python -m data.generate_simplification_data --n-samples 100000")
        print("Train: python train_simplifier.py --epochs 50")
        return

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded. Format: space-separated tokens, e.g.  ( True AND ( A ) )")
    print("Variables: A, B, C, D, E. Commands: quit, exit, help\n")

    if args.interactive or not args.expression:
        while True:
            try:
                expr = input("> ").strip() if not args.expression else args.expression[0]
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not expr:
                if args.expression:
                    break
                continue
            if expr.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break
            if expr.lower() == "help":
                print("Format: ( True AND ( A ) )  ->  A")
                print("Tokens: True, False, A, B, C, D, E, AND, OR, NOT, (, )")
                continue

            result = simplify(model, expr, device, temperature=args.temperature)
            print(f"  -> {result}\n")

            if args.expression:
                break
    else:
        for expr in args.expression:
            result = simplify(model, expr, device, temperature=args.temperature)
            print(f"{expr}  ->  {result}")


if __name__ == "__main__":
    main()
