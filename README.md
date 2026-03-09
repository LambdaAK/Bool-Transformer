# Bool-Transformer

A transformer model that learns to evaluate and generate boolean and integer expressions.

**Boolean:** True, False, AND, OR, NOT  
**Integer:** digit-level (0-9), +, -, *, //, range -99 to 99

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

Generates mixed boolean and integer expressions. Use `--bool-ratio 0.5` to control the mix.

## Train

**Boolean-only (legacy):**
```bash
python train.py --epochs 20 --batch-size 64
```

**Mixed (bool + int):**
```bash
python train_mixed.py --epochs 20 --batch-size 64
```

## Evaluate

**Boolean-only:**
```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

**Mixed:**
```bash
python evaluate_mixed.py --checkpoint checkpoints/mixed/best.pt
```

## Expression Generator (GPT-style)

Train a decoder-only transformer to *generate* valid boolean expressions:

```bash
python train_generator.py --epochs 50
python generate_expressions.py --n 10 --temperature 0.8
```

### Conditional Generation (result=True/False)

Train a generator that produces expressions evaluating to a specified result:

```bash
python train_conditional_generator.py --epochs 50
python generate_expressions.py --result True --n 5
python generate_expressions.py --result False --n 5
```

## Interactive Inference

Load the model and evaluate expressions you type:

```bash
python infer.py --checkpoint checkpoints/best.pt
```

**Boolean:** `python infer.py`  
**Mixed (bool + int):** `python infer_mixed.py`

Enter expressions. Integer literals use digit-level: `4 2` for 42, `- 1 7` for -17.

## Project Structure

- `data/generate_data.py` - Synthetic boolean expression generator
- `data/dataset.py` - PyTorch Dataset and tokenizer
- `data/expression_dataset.py` - Sequence dataset for the generator
- `data/conditional_dataset.py` - Conditional dataset (result token + expression)
- `model/transformer.py` - Encoder-only transformer
- `model/decoder_gpt.py` - GPT-style decoder for expression generation
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `infer.py` - Interactive expression evaluator
- `train_generator.py` - Train GPT-style expression generator
- `train_conditional_generator.py` - Train conditional generator (result=True/False)
- `generate_expressions.py` - Generate expressions with the trained model
