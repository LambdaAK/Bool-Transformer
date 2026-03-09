"""
PyTorch Dataset for boolean expression evaluation.
"""
import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset


# Vocabulary for boolean expressions
VOCAB = [
    "[PAD]",
    "[UNK]",
    "True",
    "False",
    "AND",
    "OR",
    "NOT",
    "(",
    ")",
]
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}


def tokenize(expression: str) -> list[int]:
    """
    Tokenize an expression into token IDs.
    Splits on whitespace and maps each token to vocabulary.
    """
    tokens = expression.split()
    ids = []
    for t in tokens:
        ids.append(TOKEN_TO_ID.get(t, TOKEN_TO_ID["[UNK]"]))
    return ids


def collate_fn(batch: list[tuple[list[int], int]], max_length: int = 64, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to max_length and stack into tensors.
    """
    input_ids = []
    labels = []

    for ids, label in batch:
        if len(ids) > max_length:
            ids = ids[:max_length]
        padding = [pad_id] * (max_length - len(ids))
        input_ids.append(ids + padding)
        labels.append(label)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class BooleanExpressionDataset(Dataset):
    """
    Dataset of (expression, result) pairs for boolean evaluation.
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        with open(path) as f:
            records = json.load(f)

        self.samples = []
        for r in records:
            expr = r["expression"]
            result = r["result"]
            ids = tokenize(expr)
            self.samples.append((ids, 1 if result else 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]
