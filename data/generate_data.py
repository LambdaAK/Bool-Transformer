"""
Generate random boolean and integer expressions and their evaluation results.
"""
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
from pathlib import Path
from typing import Optional, Union

from data.dataset import int_to_tokens, tokens_to_int_string


def generate_bool_expression(depth: int = 0, max_depth: int = 3) -> str:
    """
    Recursively generate random boolean expressions.
    Uses True, False, AND, OR, NOT, and parentheses.
    """
    if depth >= max_depth:
        return random.choice(["True", "False"])

    op = random.choice(["AND", "OR", "NOT"])
    if op == "NOT":
        inner = generate_bool_expression(depth + 1, max_depth)
        return f"NOT ( {inner} )"
    else:
        left = generate_bool_expression(depth + 1, max_depth)
        right = generate_bool_expression(depth + 1, max_depth)
        return f"( {left} {op} {right} )"


def _int_literal() -> str:
    """Generate integer literal as digit-level tokens. Range -99 to 99."""
    n = random.randint(-99, 99)
    return " ".join(int_to_tokens(n))


def generate_int_expression(depth: int = 0, max_depth: int = 3) -> Optional[str]:
    """
    Recursively generate integer expressions. +, -, *, //
    Digit-level: 42 -> "4 2", -17 -> "- 1 7"
    Skips division by zero.
    """
    if depth >= max_depth:
        return _int_literal()

    op = random.choice(["+", "-", "*", "//"])
    left = generate_int_expression(depth + 1, max_depth)
    right = generate_int_expression(depth + 1, max_depth)
    if left is None or right is None:
        return None
    # Check division by zero
    if op == "//":
        right_tokens = right.split()
        right_str = tokens_to_int_string(right_tokens)
        try:
            if eval(right_str) == 0:
                return None
        except Exception:
            return None
    return f"( {left} {op} {right} )"


def evaluate_bool_expression(expr: str) -> bool:
    """
    Safely evaluate a boolean expression.
    Only allows True, False, and, or, not - no arbitrary code execution.
    """
    # Normalize for Python eval (AND -> and, OR -> or, NOT -> not)
    safe = expr.replace(" AND ", " and ").replace(" OR ", " or ").replace("NOT ", "not ")
    return bool(eval(safe))


def evaluate_int_expression(expr: str) -> Optional[int]:
    """Safely evaluate integer expression. Returns None on error."""
    try:
        tokens = expr.split()
        eval_str = tokens_to_int_string(tokens)
        return int(eval(eval_str))
    except Exception:
        return None


def generate_dataset(
    n_samples: int,
    max_depth: int = 3,
    seed: Optional[int] = None,
    bool_ratio: float = 0.5,
) -> list[tuple[str, str, Union[bool, int]]]:
    """
    Generate mixed boolean and integer expressions.
    Returns list of (expression, type, result).
    """
    if seed is not None:
        random.seed(seed)

    data = []
    n_bool = int(n_samples * bool_ratio)
    n_int = n_samples - n_bool

    # Boolean expressions
    n_depths = max_depth + 1
    target_bool = n_bool // n_depths
    for depth in range(n_depths):
        for _ in range(target_bool):
            expr = generate_bool_expression(max_depth=depth)
            result = evaluate_bool_expression(expr)
            data.append((expr, "bool", result))

    # Integer expressions
    target_int = n_int // n_depths
    for depth in range(n_depths):
        count = 0
        attempts = 0
        while count < target_int and attempts < target_int * 20:
            attempts += 1
            expr = generate_int_expression(max_depth=depth)
            if expr is None:
                continue
            result = evaluate_int_expression(expr)
            if result is not None:
                data.append((expr, "int", result))
                count += 1

    random.shuffle(data)
    return data


def save_splits(
    data: list[tuple[str, str, Union[bool, int]]],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> None:
    """
    Split data into train/val/test and save as JSON files.
    """
    if seed is not None:
        random.seed(seed)

    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train : n_train + n_val]
    test_data = shuffled[n_train + n_val :]

    def deduplicate(pairs: list) -> list:
        """Keep first occurrence of each expression, no duplicates within split."""
        seen = set()
        unique = []
        for item in pairs:
            expr = item[0]
            if expr not in seen:
                seen.add(expr)
                unique.append(item)
        return unique

    train_data = deduplicate(train_data)
    val_data = deduplicate(val_data)
    test_data = deduplicate(test_data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_records(pairs: list) -> list[dict]:
        return [{"expression": expr, "type": t, "result": r} for expr, t, r in pairs]

    (output_dir / "train.json").write_text(
        json.dumps(to_records(train_data), indent=2)
    )
    (output_dir / "val.json").write_text(
        json.dumps(to_records(val_data), indent=2)
    )
    (output_dir / "test.json").write_text(
        json.dumps(to_records(test_data), indent=2)
    )

    print(f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples to {output_dir} (deduplicated within each split)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--bool-ratio", type=float, default=0.5, help="Fraction of boolean expressions")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_dataset(args.n_samples, args.max_depth, args.seed, args.bool_ratio)
    save_splits(data, args.output_dir, seed=args.seed)
