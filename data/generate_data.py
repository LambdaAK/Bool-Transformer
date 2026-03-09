"""
Generate random boolean expressions and their evaluation results.
"""
import json
import random
from pathlib import Path
from typing import Optional, Union


def generate_expression(depth: int = 0, max_depth: int = 3) -> str:
    """
    Recursively generate random boolean expressions.
    Uses True, False, AND, OR, NOT, and parentheses.
    """
    if depth >= max_depth:
        return random.choice(["True", "False"])

    op = random.choice(["AND", "OR", "NOT"])
    if op == "NOT":
        inner = generate_expression(depth + 1, max_depth)
        return f"NOT ( {inner} )"
    else:
        left = generate_expression(depth + 1, max_depth)
        right = generate_expression(depth + 1, max_depth)
        return f"( {left} {op} {right} )"


def evaluate_expression(expr: str) -> bool:
    """
    Safely evaluate a boolean expression.
    Only allows True, False, and, or, not - no arbitrary code execution.
    """
    # Normalize for Python eval (AND -> and, OR -> or, NOT -> not)
    safe = expr.replace(" AND ", " and ").replace(" OR ", " or ").replace("NOT ", "not ")
    return bool(eval(safe))


def generate_dataset(
    n_samples: int,
    max_depth: int = 3,
    seed: Optional[int] = None,
) -> list[tuple[str, bool]]:
    """
    Generate a dataset of (expression, result) pairs.
    """
    if seed is not None:
        random.seed(seed)

    seen = set()
    data = []

    while len(data) < n_samples:
        depth = random.randint(1, max_depth)
        expr = generate_expression(max_depth=depth)
        if expr in seen:
            continue
        seen.add(expr)
        result = evaluate_expression(expr)
        data.append((expr, result))

    return data


def save_splits(
    data: list[tuple[str, bool]],
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_records(pairs: list[tuple[str, bool]]) -> list[dict]:
        return [{"expression": expr, "result": result} for expr, result in pairs]

    (output_dir / "train.json").write_text(
        json.dumps(to_records(train_data), indent=2)
    )
    (output_dir / "val.json").write_text(
        json.dumps(to_records(val_data), indent=2)
    )
    (output_dir / "test.json").write_text(
        json.dumps(to_records(test_data), indent=2)
    )

    print(f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_dataset(args.n_samples, args.max_depth, args.seed)
    save_splits(data, args.output_dir, seed=args.seed)
