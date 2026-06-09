# Domain Classifier

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build dataset

```bash
PYTHONPATH=src python src/build_dataset.py
```

Optional custom CSV:

```bash
PYTHONPATH=src python src/build_dataset.py --custom_csv data/my_prompts.csv
```

Custom CSV format:

```csv
text,label
"Debug this Python error",coding
"Rewrite this paragraph",editing
```

## Train

```bash
PYTHONPATH=src python src/train.py --epochs 3 --batch_size 16
```

## Run API locally

```bash
PYTHONPATH=. MODEL_DIR=models/domain-classifier uvicorn src.api:app --reload
```

Test:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt":"write a python function to parse a csv"}'
```
