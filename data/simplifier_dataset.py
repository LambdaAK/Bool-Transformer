"""
Dataset for the boolean expression simplifier (seq2seq).
Loads (complex, simple) pairs from JSON.
"""
import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from data.simplifier_vocab import tokenize, BOS_ID, EOS_ID, PAD_ID


class SimplifierDataset(Dataset):
    """
    Dataset of (complex, simple) expression pairs.
    Each sample: (src_ids, tgt_ids) where tgt is BOS + simple + EOS.
    """

    def __init__(
        self,
        paths: Union[str, Path, list],
        max_length: int = 64,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(p) for p in paths]

        self.samples = []
        self.max_length = max_length

        for path in paths:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                complex_expr = r["complex"]
                simple_expr = r["simple"]
                src_ids = tokenize(complex_expr)
                tgt_ids = [BOS_ID] + tokenize(simple_expr) + [EOS_ID]
                if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
                    self.samples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def collate_simplifier(
    batch: list[tuple[list[int], list[int]]],
    max_length: int = 64,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate (src, tgt) pairs.
    Returns:
        src_ids: (batch, src_len) padded
        tgt_input_ids: (batch, tgt_len) decoder input = BOS + simple[:-1], padded
        labels: (batch, tgt_len) decoder target = simple + EOS, padded, -100 for pad
    """
    src_ids_list = []
    tgt_input_list = []
    labels_list = []

    for src_ids, tgt_ids in batch:
        # Truncate if needed
        src_ids = src_ids[:max_length]
        tgt_ids = tgt_ids[:max_length]

        # Decoder input: all but last token (BOS, t1, ..., t_{n-1})
        # Labels: all but first token (t1, ..., t_n, EOS)
        tgt_input = tgt_ids[:-1]
        labels = tgt_ids[1:]

        # Pad
        src_pad = [pad_id] * (max_length - len(src_ids))
        tgt_pad_len = max_length - len(tgt_input)
        labels_pad_len = max_length - len(labels)

        src_ids_list.append(src_ids + src_pad)
        tgt_input_list.append(tgt_input + [pad_id] * tgt_pad_len)
        # Use -100 for padding in labels so CrossEntropyLoss ignores them
        labels_list.append(labels + [-100] * labels_pad_len)

    return (
        torch.tensor(src_ids_list, dtype=torch.long),
        torch.tensor(tgt_input_list, dtype=torch.long),
        torch.tensor(labels_list, dtype=torch.long),
    )
