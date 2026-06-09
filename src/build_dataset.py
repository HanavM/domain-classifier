from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from label_config import DOLLY_CATEGORY_MAP, LABEL2ID


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip()


def load_dolly() -> pd.DataFrame:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for ex in ds:
        source_label = ex.get("category")
        label = DOLLY_CATEGORY_MAP.get(source_label, "other")
        instruction = clean_text(ex.get("instruction", ""))
        context = clean_text(ex.get("context", ""))
        text = instruction if not context else f"{instruction}\nContext: {context}"
        if text and label in LABEL2ID:
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def load_custom_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Custom CSV must have columns: text,label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"].ne("") & df["label"].isin(LABEL2ID)]
    return df


def add_seed_examples() -> pd.DataFrame:
    """Small hand-labeled set to cover domains Dolly doesn't cover well."""
    examples = [
        ("Write a Python FastAPI endpoint that classifies a prompt.", "coding"),
        ("Debug this websocket authentication error in my app.", "coding"),
        ("Create a SQL query to find daily active users.", "data_analysis"),
        ("Analyze this CSV and tell me the revenue trend.", "data_analysis"),
        ("Solve this quadratic equation step by step.", "math"),
        ("Find the derivative of x squared times sine x.", "math"),
        ("Plan a 3 day Oahu itinerary with food and beaches.", "planning"),
        ("Make a weekly study schedule for finals.", "planning"),
        ("Rewrite this paragraph to sound more professional.", "editing"),
        ("Fix grammar and improve clarity in this email.", "editing"),
        ("Find papers about diffusion models for 3D generation.", "research"),
        ("Compare current indoor navigation startups.", "research"),
    ]
    return pd.DataFrame(examples, columns=["text", "label"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_csv", type=str, default=None, help="Optional CSV with text,label columns")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load_dolly(), add_seed_examples(), load_custom_csv(args.custom_csv)], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        random_state=42,
        stratify=train_df["label"],
    )

    train_df.to_json(out_dir / "train.jsonl", orient="records", lines=True)
    val_df.to_json(out_dir / "val.jsonl", orient="records", lines=True)
    test_df.to_json(out_dir / "test.jsonl", orient="records", lines=True)

    print(f"Saved {len(train_df)} train, {len(val_df)} val, {len(test_df)} test examples to {out_dir}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
