"""
Train the mixed (bool + int) expression evaluator.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import MixedExpressionDataset, VOCAB, collate_mixed, unscale_int_pred
from model.transformer import MixedTransformer


def train_epoch(model, loader, optimizer, ce_loss, mse_loss, device):
    model.train()
    total_loss = 0.0
    bool_correct = 0
    bool_total = 0
    int_correct = 0
    int_total = 0

    for input_ids, types, labels in loader:
        input_ids = input_ids.to(device)
        types = types.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        bool_logits, int_pred = model(input_ids)

        bool_mask = types == 0
        int_mask = types == 1

        loss = torch.tensor(0.0, device=device)
        if bool_mask.any():
            lb = labels[bool_mask].long()
            loss = loss + ce_loss(bool_logits[bool_mask], lb)
            bool_correct += (bool_logits[bool_mask].argmax(1) == lb).sum().item()
            bool_total += bool_mask.sum().item()
        if int_mask.any():
            li = labels[int_mask].unsqueeze(1)
            loss = loss + mse_loss(int_pred[int_mask], li)
            preds = int_pred[int_mask].view(-1)
            labs = li.view(-1)
            pred_ints = torch.tensor([unscale_int_pred(p.item()) for p in preds], device=device)
            label_ints = torch.tensor([unscale_int_pred(l.item()) for l in labs], device=device)
            int_correct += (pred_ints == label_ints).sum().item()
            int_total += int_mask.sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    n = len(loader)
    bool_acc = bool_correct / bool_total if bool_total else 0
    int_acc = int_correct / int_total if int_total else 0
    return total_loss / n, bool_acc, int_acc


@torch.no_grad()
def evaluate(model, loader, ce_loss, mse_loss, device):
    model.eval()
    total_loss = 0.0
    bool_correct = 0
    bool_total = 0
    int_correct = 0
    int_total = 0

    for input_ids, types, labels in loader:
        input_ids = input_ids.to(device)
        types = types.to(device)
        labels = labels.to(device)

        bool_logits, int_pred = model(input_ids)

        bool_mask = types == 0
        int_mask = types == 1

        if bool_mask.any():
            lb = labels[bool_mask].long()
            total_loss += ce_loss(bool_logits[bool_mask], lb).item()
            bool_correct += (bool_logits[bool_mask].argmax(1) == lb).sum().item()
            bool_total += bool_mask.sum().item()
        if int_mask.any():
            li = labels[int_mask].unsqueeze(1)
            total_loss += mse_loss(int_pred[int_mask], li).item()
            pred_ints = [unscale_int_pred(p.item()) for p in int_pred[int_mask].view(-1)]
            label_ints = [unscale_int_pred(l.item()) for l in li.view(-1)]
            int_correct += sum(p == l for p, l in zip(pred_ints, label_ints))
            int_total += int_mask.sum().item()

    n = len(loader)
    bool_acc = bool_correct / bool_total if bool_total else 0
    int_acc = int_correct / int_total if int_total else 0
    return total_loss / n, bool_acc, int_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mixed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_ds = MixedExpressionDataset(data_dir / "train.json")
    val_ds = MixedExpressionDataset(data_dir / "val.json")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_mixed(b, max_length=args.max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_mixed(b, max_length=args.max_length),
    )

    model = MixedTransformer(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_length=args.max_length,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    epoch_iter = itertools.count(1) if args.epochs is None else range(1, args.epochs + 1)
    for epoch in epoch_iter:
        train_loss, train_bool, train_int = train_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, device
        )
        val_loss, val_bool, val_int = evaluate(model, val_loader, ce_loss, mse_loss, device)
        val_acc = (val_bool + val_int) / 2 if (val_bool or val_int) else 0

        print(f"Epoch {epoch:3d} | loss: {train_loss:.4f} | "
              f"train bool: {train_bool:.4f} int: {train_int:.4f} | "
              f"val bool: {val_bool:.4f} int: {val_int:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  -> Saved best (val acc: {val_acc:.4f})")

        if args.epochs is not None and epoch >= args.epochs:
            break

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
