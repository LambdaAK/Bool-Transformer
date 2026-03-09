"""
Evaluate the trained model on the test set.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from data.dataset import BooleanExpressionDataset, VOCAB, collate_fn
from model.transformer import BooleanTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = BooleanTransformer(
        vocab_size=len(VOCAB),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        max_length=64,
        dropout=0.0,
        pad_id=0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    test_ds = BooleanExpressionDataset(Path(args.data_dir) / "test.json")
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
