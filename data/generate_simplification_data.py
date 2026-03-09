"""
Generate (complex, simplified) boolean expression pairs for seq2seq training.
Uses expansion rules (inverse of logical identities) to create complex forms
that simplify down to simpler expressions.
"""
import json
import random
from pathlib import Path
from typing import Optional, Union

from data.simplifier_vocab import VARIABLES

# Atoms: variables + constants
ATOMS = set(VARIABLES) | {"True", "False"}


def _unwrap_if_single(expr: str) -> str:
    """( X ) -> X when X is a single atom."""
    e = expr.strip()
    if e.startswith("(") and e.endswith(")") and e.count("(") == 1:
        return e[1:-1].strip()
    return expr


def _needs_parens(expr: str) -> bool:
    """True if expr is compound (contains AND/OR at top level) and needs wrapping."""
    e = expr.strip()
    if e in ATOMS:
        return False
    if e.startswith("NOT "):
        return False
    return " AND " in e or " OR " in e


def _wrap_if_compound(expr: str) -> str:
    """Wrap in parens if compound, to avoid ambiguity in ( left OP right )."""
    if _needs_parens(expr) and not (expr.strip().startswith("(") and expr.strip().endswith(")")):
        return f"( {expr.strip()} )"
    return expr.strip()


def simplify_to_canonical(expr: str) -> str:
    """
    Recursively simplify an expression to canonical form.
    Applies: idempotence (X AND X = X), double negation, identity, annihilation.
    Returns the simplest equivalent expression.
    """
    expr = expr.strip()

    # Base case: single atom
    if expr in ATOMS:
        return expr

    # NOT ( X )
    if expr.startswith("NOT ( ") and expr.endswith(" )"):
        inner = expr[5:-2].strip()
        inner_simplified = simplify_to_canonical(inner)
        if inner_simplified == "True":
            return "False"
        if inner_simplified == "False":
            return "True"
        if inner_simplified.startswith("NOT ( ") and inner_simplified.endswith(" )"):
            # Double negation: NOT ( NOT ( X ) ) -> X
            return simplify_to_canonical(inner_simplified[5:-2].strip())
        return f"NOT ( {inner_simplified} )"

    # ( left OP right )
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1].strip()
        depth = 0
        tokens = inner.split()
        for k, t in enumerate(tokens):
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
            elif t in ("AND", "OR") and depth == 0:
                left = " ".join(tokens[:k])
                right = " ".join(tokens[k + 1 :])
                left_s = simplify_to_canonical(left)
                right_s = simplify_to_canonical(right)

                if t == "AND":
                    if left_s == "False" or right_s == "False":
                        return "False"
                    if left_s == "True":
                        return _unwrap_if_single(right_s)
                    if right_s == "True":
                        return _unwrap_if_single(left_s)
                    if left_s == right_s:
                        return left_s
                else:
                    if left_s == "True" or right_s == "True":
                        return "True"
                    if left_s == "False":
                        return _unwrap_if_single(right_s)
                    if right_s == "False":
                        return _unwrap_if_single(left_s)
                    if left_s == right_s:
                        return left_s
                return f"( {_wrap_if_compound(left_s)} {t} {_wrap_if_compound(right_s)} )"

    return expr


def generate_simple_expression(
    depth: int = 0,
    max_depth: int = 3,
    use_variables: bool = True,
) -> str:
    """
    Recursively generate a boolean expression with variables.
    Atoms: A, B, C, D, E, True, False.
    """
    atoms = list(VARIABLES) + ["True", "False"] if use_variables else ["True", "False"]
    if depth >= max_depth:
        return random.choice(atoms)

    op = random.choice(["AND", "OR", "NOT"])
    if op == "NOT":
        inner = generate_simple_expression(depth + 1, max_depth, use_variables)
        return f"NOT ( {inner} )"
    else:
        left = generate_simple_expression(depth + 1, max_depth, use_variables)
        right = generate_simple_expression(depth + 1, max_depth, use_variables)
        return f"( {left} {op} {right} )"


def expand_once(expr: str) -> str:
    """
    Apply one random expansion rule (inverse of simplification).
    Each expansion adds redundancy that the simplifier will learn to remove.
    """
    # Identity: True AND X = X, X AND True = X
    # Identity: False OR X = X, X OR False = X
    # Double negation: NOT NOT X = X
    # NOT constants: NOT True = False, NOT False = True
    rules = [
        lambda x: f"( True AND ( {x} ) )",
        lambda x: f"( ( {x} ) AND True )",
        lambda x: f"( False OR ( {x} ) )",
        lambda x: f"( ( {x} ) OR False )",
        lambda x: f"NOT ( NOT ( {x} ) )",
    ]
    if expr == "True":
        rules.append(lambda _: "NOT ( False )")
    elif expr == "False":
        rules.append(lambda _: "NOT ( True )")

    return random.choice(rules)(expr)


def expand(expr: str, num_expansions: int = 1) -> str:
    """Apply num_expansions expansion rules. Each applies to the whole current expression."""
    for _ in range(num_expansions):
        expr = expand_once(expr)
    return expr


def generate_simplification_pair(
    max_depth: int = 3,
    num_expansions: tuple[int, int] = (1, 3),
    use_variables: bool = True,
) -> tuple[str, str]:
    """
    Generate a (complex, simple) pair.
    - simple: randomly generated expression, canonicalized (fully simplified)
    - complex: simple expression with 1-3 expansion rules applied
    """
    raw_simple = generate_simple_expression(max_depth=max_depth, use_variables=use_variables)
    simple = simplify_to_canonical(raw_simple)
    n = random.randint(num_expansions[0], num_expansions[1])
    complex_expr = expand(simple, num_expansions=n)
    return complex_expr, simple


def generate_dataset(
    n_samples: int,
    max_depth: int = 3,
    num_expansions: tuple[int, int] = (1, 3),
    use_variables: bool = True,
    seed: Optional[int] = None,
) -> list[tuple[str, str]]:
    """Generate dataset of (complex, simple) pairs."""
    if seed is not None:
        random.seed(seed)

    data = []
    for _ in range(n_samples):
        pair = generate_simplification_pair(
            max_depth=max_depth,
            num_expansions=num_expansions,
            use_variables=use_variables,
        )
        data.append(pair)
    return data


def save_splits(
    data: list[tuple[str, str]],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> None:
    """Split data into train/val/test and save as JSON."""
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

    def to_records(pairs: list[tuple[str, str]]) -> list[dict]:
        return [{"complex": c, "simple": s} for c, s in pairs]

    (output_dir / "simplifier_train.json").write_text(
        json.dumps(to_records(train_data), indent=2)
    )
    (output_dir / "simplifier_val.json").write_text(
        json.dumps(to_records(val_data), indent=2)
    )
    (output_dir / "simplifier_test.json").write_text(
        json.dumps(to_records(test_data), indent=2)
    )

    print(
        f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test "
        f"samples to {output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-expansions", type=int, default=1)
    parser.add_argument("--max-expansions", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_dataset(
        args.n_samples,
        max_depth=args.max_depth,
        num_expansions=(args.min_expansions, args.max_expansions),
        seed=args.seed,
    )
    save_splits(data, args.output_dir, seed=args.seed)
