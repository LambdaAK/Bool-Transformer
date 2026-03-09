# Bool-Transformer

Transformer models for boolean expressions: **evaluation**, **generation**, and **simplification**.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

## 1. Expression Evaluator

Encoder-only transformer that classifies expressions as True or False.

### Generate Data

```bash
python -m data.generate_data --n-samples 50000 --output-dir data/splits
```

### Train

```bash
python train.py --epochs 20 --batch-size 64
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

### Interactive Inference

```bash
python infer.py --checkpoint checkpoints/best.pt
```

Enter expressions with space-separated tokens, e.g. `True AND ( False OR True )`. Type `quit` to exit.

---

## 2. Expression Generator (GPT-style)

Decoder-only transformer that *generates* valid boolean expressions.

```bash
python train_generator.py --epochs 50
python generate_expressions.py --n 10 --temperature 0.8
```

### Conditional Generation (result=True/False)

```bash
python train_conditional_generator.py --epochs 50
python generate_expressions.py --result True --n 5
python generate_expressions.py --result False --n 5
```

---

## 3. Expression Simplifier (Seq2Seq)

Encoder-decoder transformer that simplifies boolean expressions by learning logical identities (e.g. `True AND X` → `X`, `NOT NOT X` → `X`). Uses variables **A, B, C, D, E** in addition to True/False.

### Generate Data

```bash
python -m data.generate_simplification_data --n-samples 100000 --output-dir data/splits
```

### Train

```bash
python train_simplifier.py --epochs 50 --batch-size 64
```

For faster training (20k samples, 25 epochs):

```bash
python train_simplifier.py --fast
```

### Evaluate

```bash
python evaluate_simplifier.py --checkpoint checkpoints/simplifier/best.pt
```

### Simplify Expressions

```bash
# Single expression
python simplify_expression.py "( True AND ( A ) )"

# Multiple expressions
python simplify_expression.py "( True AND ( A ) )" "( False OR ( B ) )" "NOT ( NOT ( C ) )"

# Interactive mode
python simplify_expression.py --interactive
```

Format: space-separated tokens. Variables: A, B, C, D, E.

---