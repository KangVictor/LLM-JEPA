import re

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
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
            # Pre-downloaded dataset: load from disk or local files
            ds = load_dataset(
                local_path,
                split="train",
                trust_remote_code=True,
            )
            # Convert to iterable for consistent interface
            streaming = self.data_cfg.get("streaming", True)
            if streaming:
                ds = ds.to_iterable_dataset()
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
