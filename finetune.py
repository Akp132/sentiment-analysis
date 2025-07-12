import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_OUTPUT_DIR = Path("backend/model")

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer model for binary sentiment analysis.")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL dataset file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (used with configurable scheduler).")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "polynomial"], help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warm-up steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def load_jsonl_dataset(path: str) -> Dataset:
    """Load a local JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("text")
            label_str = obj.get("label")
            if text is None or label_str is None:
                continue
            label_str = label_str.lower()
            if label_str not in LABEL2ID:
                raise ValueError(f"Invalid label '{label_str}'. Expected 'positive' or 'negative'.")
            records.append({"text": text, "label": LABEL2ID[label_str]})

    if not records:
        raise ValueError("Dataset is empty or invalid.")
    return Dataset.from_list(records)

def main():
    args = parse_args()

    set_seed(args.seed)

    # Load dataset
    train_dataset = load_jsonl_dataset(args.data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Tokenize dataset
    def preprocess(example):
        return tokenizer(example["text"], truncation=True)

    train_dataset = train_dataset.map(preprocess, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",  # Temporary output
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=8,
        weight_decay=0.01,
        evaluation_strategy="no",
        logging_steps=50,
        save_strategy="no",
        gradient_accumulation_steps=1,
        seed=args.seed,
        dataloader_num_workers=2,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Gradient clipping already handled by Trainer with default value if needed.

    # Save fine-tuned model
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print(f"Model saved to {MODEL_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main() 