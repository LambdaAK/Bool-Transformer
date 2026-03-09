"""
Interactive inference: load a model and evaluate expressions you enter.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from data.dataset import tokenize, VOCAB
from model.transformer import BooleanTransformer

MAX_LENGTH = 64
PAD_ID = 0


def load_model(checkpoint_path: str, device: torch.device) -> BooleanTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BooleanTransformer(
        vocab_size=len(VOCAB),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        max_length=MAX_LENGTH,
        dropout=0.0,
        pad_id=PAD_ID,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict(model: BooleanTransformer, expression: str, device: torch.device) -> str:
    """Run model inference on an expression. Returns 'True' or 'False'."""
    ids = tokenize(expression)
    if len(ids) > MAX_LENGTH:
        ids = ids[:MAX_LENGTH]
    padding = [PAD_ID] * (MAX_LENGTH - len(ids))
    input_ids = torch.tensor([ids + padding], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)
    pred = logits.argmax(dim=1).item()
    return "True" if pred == 1 else "False"


def main():
    parser = argparse.ArgumentParser(description="Interactive boolean expression evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first with: python train.py")
        return

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded. Enter expressions to evaluate.")
    print("Format: space-separated tokens, e.g.  True AND ( False OR True )")
    print("Commands: 'quit' or 'exit' to stop, 'help' for format reminder.\n")

    while True:
        try:
            expr = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not expr:
            continue
        if expr.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if expr.lower() == "help":
            print("Format: True AND ( False OR True )")
            print("Tokens: True, False, AND, OR, NOT, (, )")
            continue

        result = predict(model, expr, device)
        print(f"  -> {result}\n")


if __name__ == "__main__":
    main()
