"""Utilities for the contextual injection benchmark."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_payloads(path: str | Path) -> list[str]:
    payloads = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            payload = line.strip()
            if payload and not payload.startswith("#"):
                payloads.append(payload)
    if not payloads:
        raise ValueError(f"No payloads found in {path}")
    return payloads


def clean_sentence_list(sentences: Iterable[str]) -> list[str]:
    return [" ".join(s.split()) for s in sentences if len(" ".join(s.split())) >= 3]


def paragraph_to_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return clean_sentence_list(sentences)


def choose_sentence_window(
    sentences: list[str],
    min_sentences: int,
    max_sentences: int,
    rng: random.Random,
) -> list[str] | None:
    """Choose a contiguous sentence window that fits the model limit."""
    if len(sentences) < min_sentences:
        return None
    if len(sentences) <= max_sentences:
        return sentences

    window_size = max(min_sentences, max_sentences)
    start = rng.randint(0, len(sentences) - window_size)
    return sentences[start : start + window_size]


def zscores(values: list[float], eps: float = 1e-8) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    if std < eps:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(1 for label in labels if label)
    if positives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Compute AUROC with average ranks for tied scores."""
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0 for _ in scores]
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def top_indices(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
