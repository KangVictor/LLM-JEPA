"""Pre-process Wikipedia into tokenized paragraph samples (train + val splits).

Usage:
    python preprocess.py --config configs/default.yaml --output data/processed

Produces:
    data/processed/train.pt   — list of {input_ids, num_sentences}
    data/processed/val.pt     — list of {input_ids, num_sentences}
    data/processed/meta.pt    — dataset statistics
"""

import argparse
import os
import random
import re

import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.data import split_sentences


def main():
    parser = argparse.ArgumentParser(description="Pre-process Wikipedia for SentenceJEPA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    local_path = data_cfg.get("local_path")
    if local_path:
        ds = load_from_disk(local_path)
    else:
        ds = load_dataset(
            data_cfg["dataset"],
            data_cfg["config"],
            streaming=True,
            split="train",
            trust_remote_code=True,
        )

    min_sent = data_cfg["min_sentences"]
    max_sent = data_cfg["max_sentences"]
    max_tokens = data_cfg["max_tokens_per_sentence"]
    val_size = cfg["training"].get("val_paragraphs", 1000)

    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"], use_fast=True)

    print("Processing articles...")
    all_samples = []
    num_articles = 0
    sentence_counts = []

    for article in ds:
        num_articles += 1
        text = article["text"]
        paragraphs = re.split(r'\n\n+', text)
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            sentences = split_sentences(para)
            n = len(sentences)
            if n < min_sent:
                continue
            sentences = sentences[:max_sent]

            encoded = tokenizer(
                sentences,
                padding=False,
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=False,
            )
            all_samples.append({
                "input_ids": encoded["input_ids"],
                "num_sentences": len(encoded["input_ids"]),
            })
            sentence_counts.append(len(sentences))

        if num_articles % 100_000 == 0:
            print(f"  ...{num_articles:,} articles, {len(all_samples):,} paragraphs")

    print(f"\nTotal: {num_articles:,} articles, {len(all_samples):,} paragraphs")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_samples)

    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    # Stats
    sentence_counts_t = torch.tensor(sentence_counts, dtype=torch.float32)
    meta = {
        "num_articles": num_articles,
        "num_paragraphs": len(all_samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "sentence_count_mean": sentence_counts_t.mean().item(),
        "sentence_count_min": sentence_counts_t.min().item(),
        "sentence_count_max": sentence_counts_t.max().item(),
    }

    # Save
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.pt")
    val_path = os.path.join(args.output, "val.pt")
    meta_path = os.path.join(args.output, "meta.pt")

    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    torch.save(meta, meta_path)

    print(f"\n{'='*50}")
    print(f"Saved preprocessed dataset to {args.output}/")
    print(f"{'='*50}")
    print(f"  train.pt: {len(train_samples):,} paragraphs ({os.path.getsize(train_path) / 1e9:.2f} GB)")
    print(f"  val.pt:   {len(val_samples):,} paragraphs ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print(f"  meta.pt:  dataset statistics")
    for k, v in meta.items():
        print(f"    {k}: {v}")
    print(f"{'='*50}")
    print(f"\nTo train with this data:")
    print(f"  python train.py --config configs/default.yaml --override data.preprocessed_path={args.output}")


if __name__ == "__main__":
    main()
