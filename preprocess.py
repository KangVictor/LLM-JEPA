"""Pre-process Wikipedia into tokenized paragraph samples (train + val splits).

Usage:
    python preprocess.py --config configs/default.yaml --output data/processed
    python preprocess.py --config configs/default.yaml --output data/processed --workers 8

Produces:
    data/processed/train.pt   — list of {input_ids, num_sentences}
    data/processed/val.pt     — list of {input_ids, num_sentences}
    data/processed/meta.pt    — dataset statistics
"""

import argparse
import os
import random
import re
import time
from multiprocessing import Pool

import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.data import split_sentences

# ── Step 1: Extract paragraphs (parallelizable) ──────────────────────────────

# Module-level config for worker processes
_min_sent = None
_max_sent = None


def _init_worker(min_s, max_s):
    global _min_sent, _max_sent
    _min_sent = min_s
    _max_sent = max_s


def _extract_paragraphs(text):
    """Extract qualifying sentence lists from an article text. Runs in worker."""
    results = []
    paragraphs = re.split(r'\n\n+', text)
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:
            continue
        sentences = split_sentences(para)
        if len(sentences) < _min_sent:
            continue
        results.append(sentences[:_max_sent])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-process Wikipedia for SentenceJEPA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for sentence splitting")
    parser.add_argument("--tokenizer_batch", type=int, default=50_000, help="Sentences per tokenizer batch")
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

    t0 = time.time()

    # ── Phase 1: Extract sentence lists (parallel) ────────────────────────────
    print(f"Phase 1: Extracting paragraphs ({args.workers} workers)...")

    # Collect article texts in chunks, process in parallel
    all_paragraph_sentences = []  # list of sentence-lists
    num_articles = 0
    chunk_size = 10_000
    text_chunk = []

    pool = Pool(args.workers, initializer=_init_worker, initargs=(min_sent, max_sent))

    for article in ds:
        text_chunk.append(article["text"])
        num_articles += 1

        if len(text_chunk) >= chunk_size:
            results = pool.map(_extract_paragraphs, text_chunk, chunksize=500)
            for para_list in results:
                all_paragraph_sentences.extend(para_list)
            text_chunk = []

            elapsed = time.time() - t0
            rate = num_articles / elapsed
            print(f"  {num_articles:,} articles | {len(all_paragraph_sentences):,} paragraphs | {rate:.0f} art/s")

    # Process remaining
    if text_chunk:
        results = pool.map(_extract_paragraphs, text_chunk, chunksize=500)
        for para_list in results:
            all_paragraph_sentences.extend(para_list)

    pool.close()
    pool.join()

    t1 = time.time()
    print(f"  Done: {num_articles:,} articles, {len(all_paragraph_sentences):,} paragraphs in {t1 - t0:.0f}s")

    # ── Phase 2: Batch tokenization ───────────────────────────────────────────
    print(f"Phase 2: Tokenizing ({args.tokenizer_batch:,} sentences/batch)...")

    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"], use_fast=True)

    # Flatten all sentences with paragraph boundary tracking
    flat_sentences = []
    para_boundaries = []  # (start_idx, end_idx) into flat_sentences
    for sentences in all_paragraph_sentences:
        start = len(flat_sentences)
        flat_sentences.extend(sentences)
        para_boundaries.append((start, len(flat_sentences)))

    print(f"  Total sentences to tokenize: {len(flat_sentences):,}")

    # Batch tokenize all sentences at once (in chunks to limit memory)
    all_token_ids = []
    batch_sz = args.tokenizer_batch
    for i in range(0, len(flat_sentences), batch_sz):
        batch = flat_sentences[i : i + batch_sz]
        encoded = tokenizer(
            batch,
            padding=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=False,
        )
        all_token_ids.extend(encoded["input_ids"])

        if (i // batch_sz) % 10 == 0:
            pct = (i + len(batch)) / len(flat_sentences) * 100
            print(f"  {pct:.0f}% tokenized")

    t2 = time.time()
    print(f"  Done: {len(all_token_ids):,} sentences tokenized in {t2 - t1:.0f}s")

    # ── Phase 3: Reassemble into paragraph samples ────────────────────────────
    print("Phase 3: Assembling samples...")

    all_samples = []
    sentence_counts = []
    for start, end in para_boundaries:
        ids = all_token_ids[start:end]
        all_samples.append({
            "input_ids": ids,
            "num_sentences": len(ids),
        })
        sentence_counts.append(len(ids))

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

    print("Saving...")
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    torch.save(meta, meta_path)

    t3 = time.time()
    print(f"\n{'='*50}")
    print(f"Saved preprocessed dataset to {args.output}/")
    print(f"{'='*50}")
    print(f"  train.pt: {len(train_samples):,} paragraphs ({os.path.getsize(train_path) / 1e9:.2f} GB)")
    print(f"  val.pt:   {len(val_samples):,} paragraphs ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print(f"  meta.pt:  dataset statistics")
    for k, v in meta.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.1f}")
        else:
            print(f"    {k}: {v:,}")
    print(f"  Total time: {t3 - t0:.0f}s")
    print(f"{'='*50}")
    print(f"\nTo train with this data:")
    print(f"  python train.py --config configs/default.yaml --override data.preprocessed_path={args.output}")


if __name__ == "__main__":
    main()
