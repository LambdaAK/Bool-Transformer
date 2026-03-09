"""
PyTorch Dataset for boolean expression evaluation.
"""
import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset


# Vocabulary for boolean and integer expressions
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
    "[BOS]",
    "[EOS]",
    "[RESULT_TRUE]",
    "[RESULT_FALSE]",
    "[RESULT_INT]",
    # Integer: digits and operators
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "//",
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


def int_to_tokens(n: int) -> list[str]:
    """Convert integer to digit-level token strings. Range -99 to 99."""
    if n < 0:
        return ["-"] + list(str(-n))
    return list(str(n))


DIGITS = set("0123456789")

# Integer result range for normalization (mixed model)
INT_MIN, INT_MAX = -1000, 1000


def scale_int_label(value: float) -> float:
    """Scale int to [0, 1] for MSE loss. Clamp to INT_MIN..INT_MAX."""
    clamped = max(INT_MIN, min(INT_MAX, value))
    return (clamped - INT_MIN) / (INT_MAX - INT_MIN)


def unscale_int_pred(scaled: float) -> int:
    """Unscale from [0, 1] back to integer."""
    return round(scaled * (INT_MAX - INT_MIN) + INT_MIN)


def tokens_to_int_string(tokens: list[str]) -> str:
    """Group digit tokens into numbers for evaluation. Returns eval-able string."""
    result = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in "()" or t in "+-*" or t == "//":
            result.append(t)
            i += 1
        elif t == "-" and i + 1 < len(tokens) and tokens[i + 1] in DIGITS:
            num = "-"
            i += 1
            while i < len(tokens) and tokens[i] in DIGITS:
                num += tokens[i]
                i += 1
            result.append(num)
        elif t in DIGITS:
            num = ""
            while i < len(tokens) and tokens[i] in DIGITS:
                num += tokens[i]
                i += 1
            result.append(num)
        else:
            i += 1
    return " ".join(result)


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
    For backward compatibility with old data format (no type field).
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


class MixedExpressionDataset(Dataset):
    """
    Dataset of (expression, type, result) for mixed bool/int evaluation.
    type: 0 = bool, 1 = int
    result: for bool 0/1, for int the float value
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        with open(path) as f:
            records = json.load(f)

        self.samples = []
        for r in records:
            expr = r["expression"]
            t = r["type"]
            result = r["result"]
            ids = tokenize(expr)
            if t == "bool":
                self.samples.append((ids, 0, 1.0 if result else 0.0))
            else:
                self.samples.append((ids, 1, scale_int_label(float(result))))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int, float]:
        return self.samples[idx]


def collate_mixed(batch: list, max_length: int = 64, pad_id: int = 0):
    """Collate for mixed dataset: (input_ids, types, labels)."""
    input_ids = []
    types = []
    labels = []
    for ids, t, label in batch:
        if len(ids) > max_length:
            ids = ids[:max_length]
        padding = [pad_id] * (max_length - len(ids))
        input_ids.append(ids + padding)
        types.append(t)
        labels.append(label)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(types, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )
