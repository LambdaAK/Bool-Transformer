"""
Train the expression generator (GPT-style language model).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import VOCAB
from data.expression_dataset import (
    ExpressionSequenceDataset,
    collate_sequences,
    PAD_ID,
)
from model.decoder_gpt import ExpressionGPT


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
        # labels: (batch, seq_len), -100 is ignored
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/generator")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (default: unlimited)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
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
    train_ds = ExpressionSequenceDataset(
        [data_dir / "train.json", data_dir / "val.json"],
        max_length=args.max_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sequences(b, max_length=args.max_length),
    )

    model = ExpressionGPT(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        max_length=args.max_length,
        dropout=args.dropout,
        pad_id=PAD_ID,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_iter = itertools.count(1) if args.epochs is None else range(1, args.epochs + 1)
    for epoch in epoch_iter:
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch:3d} | loss: {loss:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / "latest.pt",
            )

        if args.epochs is not None and epoch >= args.epochs:
            break

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
