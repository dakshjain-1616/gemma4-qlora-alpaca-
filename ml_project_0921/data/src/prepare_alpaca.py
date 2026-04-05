#!/usr/bin/env python
"""
Data preparation pipeline for Alpaca instruction tuning dataset.
Model: google/gemma-4-E2B-it
"""

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app/ml_project_0921")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-4-E2B-it"


def format_gemma_instruction(example):
    """Format instruction for Gemma 4 model."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    user_content = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
    return f"<user>\n{user_content}<end>\n<assistant>\n{output_text}<end>"


def main(subset_size: int = 5000):
    logger.info("=" * 60)
    logger.info(f"Alpaca data preparation — tokenizer: {MODEL_ID}")
    logger.info("=" * 60)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load dataset
    logger.info("Loading tatsu-lab/alpaca...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
    logger.info(f"Loaded {len(dataset)} samples")

    # Format
    formatted = [format_gemma_instruction(ex) for ex in dataset]
    logger.info(f"Formatted {len(formatted)} samples")

    # Tokenize
    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    tokenized = []
    for i, text in enumerate(formatted):
        enc = tokenizer(text, max_length=512, truncation=True, padding=False, return_tensors=None)
        tokenized.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"].copy(),
            "text": text,
            "instruction": dataset[i].get("instruction", ""),
            "output": dataset[i].get("output", ""),
        })
    logger.info(f"Tokenized {len(tokenized)} samples")

    # Split 90/10
    random.shuffle(tokenized)
    split = int(len(tokenized) * 0.9)
    train_data = tokenized[:split]
    test_data = tokenized[split:]
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Save
    with open(DATA_DIR / "train.json", "w") as f:
        json.dump(train_data, f)
    with open(DATA_DIR / "test.json", "w") as f:
        json.dump(test_data, f)
    metadata = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "tokenizer": MODEL_ID,
        "dataset": "tatsu-lab/alpaca",
        "subset_size": subset_size,
        "max_length": 512,
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved to {DATA_DIR}")
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()