"""
Evaluate the mixed (bool + int) model on the test set.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from data.dataset import MixedExpressionDataset, VOCAB, collate_mixed, unscale_int_pred
from model.transformer import MixedTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mixed/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MixedTransformer(vocab_size=len(VOCAB), max_length=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_ds = MixedExpressionDataset(Path(args.data_dir) / "test.json")
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_mixed(b),
    )

    bool_correct = 0
    bool_total = 0
    int_correct = 0
    int_total = 0

    with torch.no_grad():
        for input_ids, types, labels in loader:
            input_ids = input_ids.to(device)
            types = types.to(device)
            labels = labels.to(device)

            bool_logits, int_pred = model(input_ids)

            bool_mask = types == 0
            int_mask = types == 1

            if bool_mask.any():
                lb = labels[bool_mask].long()
                bool_correct += (bool_logits[bool_mask].argmax(1) == lb).sum().item()
                bool_total += bool_mask.sum().item()
            if int_mask.any():
                li = labels[int_mask].view(-1)
                preds = int_pred[int_mask].view(-1)
                pred_ints = [unscale_int_pred(p.item()) for p in preds]
                label_ints = [unscale_int_pred(l.item()) for l in li]
                int_correct += sum(p == l for p, l in zip(pred_ints, label_ints))
                int_total += int_mask.sum().item()

    print(f"Test: bool acc: {bool_correct}/{bool_total} = {bool_correct/bool_total:.4f}" if bool_total else "Test: no bool samples")
    print(f"Test: int acc: {int_correct}/{int_total} = {int_correct/int_total:.4f}" if int_total else "Test: no int samples")
    if bool_total and int_total:
        print(f"Overall: {(bool_correct + int_correct) / (bool_total + int_total):.4f}")


if __name__ == "__main__":
    main()
