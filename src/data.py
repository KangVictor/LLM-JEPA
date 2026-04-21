import os
import re

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def split_sentences(text):
    """Split text into sentences using regex. Handles common abbreviations."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    # Filter out very short fragments (< 3 chars)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


def summarize_dataset(cfg):
    """Scan the dataset, print summary stats, and collect a validation set.

    Returns:
        num_paragraphs: total qualifying paragraph count (excluding val)
        val_samples: list of tokenized val samples (dicts with input_ids, num_sentences)
    """
    print("Scanning dataset for summary statistics...")
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

    num_articles = 0
    num_paragraphs = 0
    sentence_counts = []
    val_samples = []

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
            n = len(sentences)
            num_paragraphs += 1
            sentence_counts.append(n)

            # Collect validation samples from the tail of the scan
            if len(val_samples) < val_size:
                encoded = tokenizer(
                    sentences,
                    padding=False,
                    truncation=True,
                    max_length=max_tokens,
                    return_attention_mask=False,
                )
                val_samples.append({
                    "input_ids": encoded["input_ids"],
                    "num_sentences": len(encoded["input_ids"]),
                })

        if num_articles % 100_000 == 0:
            print(f"  ...scanned {num_articles:,} articles, {num_paragraphs:,} paragraphs so far")

    # Exclude val from training count
    num_train = num_paragraphs - len(val_samples)

    sentence_counts_t = torch.tensor(sentence_counts, dtype=torch.float32)
    batch_size = cfg["training"]["batch_size"]
    steps_per_epoch = num_train // batch_size

    print(f"\n{'='*50}")
    print(f"Dataset Summary")
    print(f"{'='*50}")
    print(f"Articles:           {num_articles:,}")
    print(f"Qualifying paras:   {num_paragraphs:,}")
    print(f"  Train:            {num_train:,}")
    print(f"  Val:              {len(val_samples):,}")
    print(f"Sentences/para:     mean={sentence_counts_t.mean():.1f}, "
          f"min={sentence_counts_t.min().long().item()}, "
          f"max={sentence_counts_t.max().long().item()}")
    print(f"Steps/epoch (bs={batch_size}): {steps_per_epoch:,}")
    print(f"{'='*50}\n")

    return num_train, val_samples


class WikiParagraphDataset(IterableDataset):
    """Streams Wikipedia paragraphs as bags of tokenized sentences."""

    def __init__(self, cfg):
        super().__init__()
        self.data_cfg = cfg["data"]
        self.min_sentences = self.data_cfg["min_sentences"]
        self.max_sentences = self.data_cfg["max_sentences"]
        self.max_tokens = self.data_cfg["max_tokens_per_sentence"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.data_cfg["tokenizer"], use_fast=True
        )

    def _load_dataset(self):
        """Load dataset from local path or HuggingFace Hub."""
        local_path = self.data_cfg.get("local_path")
        if local_path:
            # Pre-downloaded dataset saved with save_to_disk()
            ds = load_from_disk(local_path)
            # Convert to iterable with enough shards for multi-worker loading
            num_shards = max(self.data_cfg.get("num_workers", 4), 1)
            ds = ds.to_iterable_dataset(num_shards=num_shards)
        else:
            ds = load_dataset(
                self.data_cfg["dataset"],
                self.data_cfg["config"],
                streaming=self.data_cfg.get("streaming", True),
                split="train",
                trust_remote_code=True,
            )
        return ds

    def _stream_paragraphs(self):
        ds = self._load_dataset()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard the stream across workers
            ds = ds.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )

        for article in ds:
            text = article["text"]
            # Split on double newlines (or single newlines for Wikipedia structure)
            paragraphs = re.split(r'\n\n+', text)
            for para in paragraphs:
                para = para.strip()
                if len(para) < 20:
                    continue
                sentences = split_sentences(para)
                if len(sentences) < self.min_sentences:
                    continue
                sentences = sentences[:self.max_sentences]
                yield sentences

    def __iter__(self):
        for sentences in self._stream_paragraphs():
            # Tokenize each sentence independently
            encoded = self.tokenizer(
                sentences,
                padding=False,
                truncation=True,
                max_length=self.max_tokens,
                return_attention_mask=False,
            )
            input_ids = encoded["input_ids"]  # list of variable-length lists
            yield {
                "input_ids": input_ids,
                "num_sentences": len(input_ids),
            }


class PreprocessedDataset(Dataset):
    """Map-style dataset backed by a pre-processed .pt file."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_preprocessed(cfg):
    """Load pre-processed train/val splits and metadata.

    Returns:
        train_dataset: PreprocessedDataset
        val_samples: list of dicts
        num_train: int
    """
    path = cfg["data"]["preprocessed_path"]
    print(f"Loading preprocessed data from {path}/...")

    train_samples = torch.load(os.path.join(path, "train.pt"), weights_only=False)
    val_samples = torch.load(os.path.join(path, "val.pt"), weights_only=False)
    meta = torch.load(os.path.join(path, "meta.pt"), weights_only=False)

    batch_size = cfg["training"]["batch_size"]
    steps_per_epoch = len(train_samples) // batch_size

    print(f"\n{'='*50}")
    print(f"Preprocessed Dataset Summary")
    print(f"{'='*50}")
    for k, v in meta.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v:,}")
    print(f"  Steps/epoch (bs={batch_size}): {steps_per_epoch:,}")
    print(f"{'='*50}\n")

    return PreprocessedDataset(train_samples), val_samples, len(train_samples)


def collate_fn(batch):
    """Collate variable-length paragraph samples into padded tensors.

    Returns:
        input_ids: (B, S_max, T_max) padded token ids
        attention_mask: (B, S_max, T_max) 1=real token, 0=pad
        sentence_mask: (B, S_max) True=real sentence, False=pad
    """
    batch_size = len(batch)
    max_sentences = max(item["num_sentences"] for item in batch)
    max_tokens = max(
        len(ids)
        for item in batch
        for ids in item["input_ids"]
    )

    input_ids = torch.zeros(batch_size, max_sentences, max_tokens, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_sentences, max_tokens, dtype=torch.long)
    sentence_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, ids in enumerate(item["input_ids"]):
            seq_len = len(ids)
            input_ids[i, j, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, j, :seq_len] = 1
            sentence_mask[i, j] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentence_mask": sentence_mask,
    }
