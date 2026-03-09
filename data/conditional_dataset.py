"""
Dataset for conditional expression generation.
Bool: [BOS] [RESULT_TRUE/FALSE] expr [EOS]
Int:  [BOS] [RESULT_INT] <digits> expr [EOS]
"""
import json
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from data.dataset import tokenize, VOCAB, int_to_tokens

BOS_ID = VOCAB.index("[BOS]")
EOS_ID = VOCAB.index("[EOS]")
RESULT_TRUE_ID = VOCAB.index("[RESULT_TRUE]")
RESULT_FALSE_ID = VOCAB.index("[RESULT_FALSE]")
RESULT_INT_ID = VOCAB.index("[RESULT_INT]")
PAD_ID = 0
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}


class ConditionalExpressionDataset(Dataset):
    """
    Dataset of conditional sequences for mixed bool/int.
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
                t = r.get("type", "bool")
                result = r["result"]

                if t == "bool":
                    result_token = RESULT_TRUE_ID if result else RESULT_FALSE_ID
                    seq = [BOS_ID, result_token] + tokenize(expr) + [EOS_ID]
                else:
                    result_digit_ids = [TOKEN_TO_ID.get(c, PAD_ID) for c in int_to_tokens(result)]
                    seq = [BOS_ID, RESULT_INT_ID] + result_digit_ids + tokenize(expr) + [EOS_ID]

                if len(seq) <= max_length:
                    self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.samples[idx]
