"""
Dataset for conditional expression generation (result=True/False).
Format: [BOS] [RESULT_TRUE] or [RESULT_FALSE] + expression tokens + [EOS]
"""
import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from data.dataset import tokenize, VOCAB

BOS_ID = VOCAB.index("[BOS]")
EOS_ID = VOCAB.index("[EOS]")
RESULT_TRUE_ID = VOCAB.index("[RESULT_TRUE]")
RESULT_FALSE_ID = VOCAB.index("[RESULT_FALSE]")
PAD_ID = 0


class ConditionalExpressionDataset(Dataset):
    """
    Dataset of conditional sequences: [BOS] [RESULT_X] expr [EOS].
    Uses (expression, result) pairs from the evaluator data.
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
                result = r["result"]
                ids = tokenize(expr)
                result_token = RESULT_TRUE_ID if result else RESULT_FALSE_ID
                # BOS + [RESULT_X] + tokens + EOS
                seq = [BOS_ID, result_token] + ids + [EOS_ID]
                if len(seq) <= max_length:
                    self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]
