"""
Dataset for training the expression generator (language modeling).
"""
import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from data.dataset import tokenize, VOCAB, TOKEN_TO_ID

BOS_ID = VOCAB.index("[BOS]")
EOS_ID = VOCAB.index("[EOS]")
PAD_ID = 0


class ExpressionSequenceDataset(Dataset):
    """
    Dataset of token sequences for next-token prediction.
    Each sample is BOS + tokens + EOS.
    """

    def __init__(self, paths: Union[str, Path, list], max_length: int = 64):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(p) for p in paths]

        self.samples = []
        self.max_length = max_length

        for path in paths:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                expr = r["expression"]
                ids = tokenize(expr)
                # BOS + tokens + EOS
                seq = [BOS_ID] + ids + [EOS_ID]
                if len(seq) <= max_length:
                    self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]


def collate_sequences(
    batch: list[list[int]],
    max_length: int = 64,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences and create input/target for next-token prediction.
    input = [BOS, t1, ..., t_{n-1}]  (predict next token at each position)
    target = [t1, t2, ..., EOS]      (shifted by 1)
    Both padded to max_length. Padding in target is ignored via ignore_index.
    """
    input_ids = []
    labels = []

    for seq in batch:
        if len(seq) > max_length:
            seq = seq[:max_length]
        # input: all but last token; target: all but first (shifted)
        inp = seq[:-1]
        tgt = seq[1:]
        pad_len = max_length - len(inp)
        input_ids.append(inp + [pad_id] * pad_len)
        labels.append(tgt + [pad_id] * pad_len)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
