"""
Interactive inference for mixed (bool + int) expressions.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from data.dataset import tokenize, VOCAB, unscale_int_pred
from model.transformer import MixedTransformer

BOOL_TOKENS = {"True", "False", "AND", "OR", "NOT"}


def is_bool_expression(expr: str) -> bool:
    """Heuristic: if expression contains bool tokens, it's boolean."""
    tokens = set(expr.split())
    return bool(tokens & BOOL_TOKENS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mixed/best.pt")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train first: python train_mixed.py --epochs 20")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MixedTransformer(vocab_size=len(VOCAB), max_length=args.max_length)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Mixed expression evaluator. Enter bool or int expressions.")
    print("Bool: True AND ( False OR True )")
    print("Int:  ( 4 2 + 1 7 )  (digit-level: 42+17)")
    print("Commands: quit, exit, help\n")

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
            print("Bool: True, False, AND, OR, NOT, ( )")
            print("Int: 0-9, +, -, *, //, ( )  (e.g. 4 2 for 42)")
            continue

        ids = tokenize(expr)
        if len(ids) > args.max_length:
            ids = ids[:args.max_length]
        padding = [0] * (args.max_length - len(ids))
        input_ids = torch.tensor([ids + padding], dtype=torch.long, device=device)

        with torch.no_grad():
            bool_logits, int_pred = model(input_ids)

        if is_bool_expression(expr):
            pred = bool_logits.argmax(dim=1).item()
            print(f"  -> {pred == 1}\n")
        else:
            pred = unscale_int_pred(int_pred[0, 0].item())
            print(f"  -> {pred}\n")


if __name__ == "__main__":
    main()
