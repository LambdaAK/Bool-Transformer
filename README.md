# Bool-Transformer

A transformer model that learns to evaluate boolean expressions (True, False, AND, OR, NOT).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Generate Data

```bash
python data/generate_data.py --n-samples 50000 --output-dir data/splits
```

## Train

```bash
python train.py --epochs 20 --batch-size 64
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

## Interactive Inference

Load the model and evaluate expressions you type:

```bash
python infer.py --checkpoint checkpoints/best.pt
```

Enter expressions with space-separated tokens, e.g. `True AND ( False OR True )`. Type `quit` to exit.

## Project Structure

- `data/generate_data.py` - Synthetic boolean expression generator
- `data/dataset.py` - PyTorch Dataset and tokenizer
- `model/transformer.py` - Encoder-only transformer
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `infer.py` - Interactive expression evaluator
