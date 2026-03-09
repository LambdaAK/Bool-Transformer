"""
Train the boolean expression simplifier (seq2seq).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.simplifier_vocab import VOCAB, PAD_ID
from data.simplifier_dataset import SimplifierDataset, collate_simplifier
from model.seq2seq import SimplifierTransformer


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src_ids, tgt_input_ids, labels in loader:
        src_ids = src_ids.to(device)
        tgt_input_ids = tgt_input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(src_ids, tgt_input_ids)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/simplifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (avoids MPS issues on Mac)")
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample training data (e.g. 20000 for faster runs)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (4 recommended for CPU speed)")
    parser.add_argument("--fast", action="store_true", help="Fast preset: 20k samples, 25 epochs, batch 128")
    args = parser.parse_args()

    if args.fast:
        if args.max_samples is None:
            args.max_samples = 20000
        args.epochs = 25
        args.batch_size = 128
        args.num_workers = 4
        print("Using --fast preset")

    torch.manual_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        # Prefer CPU over MPS: nn.Transformer has known MPS bugs (nested tensor masks)
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_ds = SimplifierDataset(
        data_dir / "simplifier_train.json",
        max_length=args.max_length,
    )
    val_ds = SimplifierDataset(
        data_dir / "simplifier_val.json",
        max_length=args.max_length,
    )
    if args.max_samples is not None:
        from torch.utils.data import Subset
        n = min(args.max_samples, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        print(f"Using {n} training samples (subsampled)")
    collate_fn = partial(collate_simplifier, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = SimplifierTransformer(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 2,
        max_length=args.max_length,
        dropout=args.dropout,
        pad_id=PAD_ID,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_ids, tgt_input_ids, labels in val_loader:
                src_ids = src_ids.to(device)
                tgt_input_ids = tgt_input_ids.to(device)
                labels = labels.to(device)
                logits = model(src_ids, tgt_input_ids)
                val_loss += criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_dir / "best.pt",
            )

        if epoch % 5 == 0 or epoch == 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / "latest.pt",
            )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
