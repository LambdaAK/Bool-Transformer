"""
Vocabulary for the boolean expression simplifier.
Includes variables (A, B, C, D, E) in addition to constants and operators.
"""
from typing import Optional

# Variables + original tokens
VARIABLES = ["A", "B", "C", "D", "E"]

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
] + VARIABLES

TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}

PAD_ID = 0
BOS_ID = VOCAB.index("[BOS]")
EOS_ID = VOCAB.index("[EOS]")


def tokenize(expression: str) -> list[int]:
    """Tokenize an expression into token IDs. Splits on whitespace."""
    tokens = expression.split()
    ids = []
    for t in tokens:
        ids.append(TOKEN_TO_ID.get(t, TOKEN_TO_ID["[UNK]"]))
    return ids


def detokenize(ids: list[int], strip_special: bool = True) -> str:
    """Convert token IDs back to a string."""
    tokens = []
    for i in ids:
        if strip_special and i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        tokens.append(ID_TO_TOKEN.get(i, "[UNK]"))
    return " ".join(tokens)
