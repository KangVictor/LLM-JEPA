"""KNN probe on MTEB classification datasets.

This freezes the SentenceJEPA embedding model, encodes the MTEB train/test
splits, fits KNN classifiers on the train embeddings, and evaluates on test.

Usage:
    python -m finetune.knn_probe \
        --config configs/colab.yaml \
        --checkpoint /path/to/step_50000.pt \
        --task Banking77Classification \
        --embedding_mode document \
        --k 1 3 5 10 20

    python -m finetune.knn_probe \
        --config configs/colab.yaml \
        --checkpoint /path/to/step_50000.pt \
        --task ToxicConversationsClassification \
        --embedding_mode document_layer_mean \
        --metric cosine \
        --weights distance
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer

from finetune.linear_probe import (
    all_int_like,
    encode_texts,
    load_encoder,
    load_mteb_classification_data,
)


def subset_examples(texts, labels, max_samples, seed):
    """Deterministically subsample examples for faster diagnostic runs."""
    if max_samples is None or len(texts) <= max_samples:
        return texts, labels

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=generator)[:max_samples].tolist()
    return [texts[i] for i in indices], [labels[i] for i in indices]


def subset_examples_per_class(texts, labels, samples_per_class, seed):
    """Sample up to N train examples from each class."""
    if samples_per_class is None:
        return texts, labels
    if samples_per_class < 1:
        raise ValueError("--train_samples_per_class must be >= 1")

    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    selected = []
    class_counts = {}
    for label in sorted(by_class):
        indices = by_class[label]
        perm = torch.randperm(len(indices), generator=generator).tolist()
        keep = [indices[i] for i in perm[:samples_per_class]]
        selected.extend(keep)
        class_counts[label] = len(keep)

    selected.sort()
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    print(
        f"Train per-class sample cap: {samples_per_class} "
        f"({len(class_counts):,} classes, min={min_count}, max={max_count})"
    )
    return [texts[i] for i in selected], [labels[i] for i in selected]


def as_class_labels(train_labels, test_labels):
    """Convert MTEB labels into discrete class ids for KNN classification."""
    labels = list(train_labels) + list(test_labels)
    if not all_int_like(labels):
        raise ValueError(
            "KNN classification requires discrete integer-like labels. "
            "This task appears to expose continuous labels; use a regression "
            "probe instead."
        )
    return np.asarray(train_labels, dtype=np.int64), np.asarray(test_labels, dtype=np.int64)


def normalize_if_needed(embs, enabled):
    if not enabled:
        return embs
    return F.normalize(embs.float(), dim=1).cpu().numpy()


def evaluate_knn(train_embs, train_labels, test_embs, test_labels, label_names, args):
    rows = []
    reports = {}

    for k in args.k:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > len(train_labels):
            print(f"Skipping k={k}: only {len(train_labels):,} train examples")
            continue

        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights=args.weights,
            metric=args.metric,
            algorithm="brute",
            n_jobs=args.n_jobs,
        )
        classifier.fit(train_embs, train_labels)
        preds = classifier.predict(test_embs)

        row = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "embedding_mode": args.embedding_mode,
            "k": int(k),
            "metric": args.metric,
            "weights": args.weights,
            "train_samples": int(len(train_labels)),
            "test_samples": int(len(test_labels)),
            "accuracy": float(accuracy_score(test_labels, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, preds)),
            "macro_f1": float(f1_score(test_labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(test_labels, preds, average="weighted", zero_division=0)
            ),
            "macro_precision": float(
                precision_score(test_labels, preds, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(test_labels, preds, average="macro", zero_division=0)
            ),
        }
        rows.append(row)

        report_names = None
        unique_labels = sorted(set(train_labels.tolist()) | set(test_labels.tolist()))
        if label_names and len(label_names) == len(unique_labels):
            report_names = [str(name) for name in label_names]

        reports[str(k)] = classification_report(
            test_labels,
            preds,
            labels=unique_labels,
            target_names=report_names,
            output_dict=True,
            zero_division=0,
        )

        print(
            f"k={k:<4} acc={row['accuracy']:.4f} "
            f"bal_acc={row['balanced_accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f}"
        )

    if not rows:
        raise RuntimeError("No KNN results were produced.")

    best = max(rows, key=lambda row: row["accuracy"])
    return rows, reports, best


def write_outputs(rows, reports, best, args):
    os.makedirs(args.output, exist_ok=True)
    checkpoint_name = Path(args.checkpoint).stem
    base = f"{args.task}_{args.embedding_mode}_{checkpoint_name}_knn"

    json_path = os.path.join(args.output, f"{base}.json")
    csv_path = os.path.join(args.output, f"{base}.csv")

    payload = {
        "best": best,
        "results": rows,
        "classification_reports": reports,
        "args": vars(args),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBest k={best['k']} accuracy={best['accuracy']:.4f}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="KNN probe on MTEB classification tasks")
    parser.add_argument("--config", type=str, required=True, help="SentenceJEPA config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="SentenceJEPA checkpoint")
    parser.add_argument("--task", type=str, required=True, help="MTEB classification task")
    parser.add_argument(
        "--embedding_mode",
        choices=["document", "document_layer_mean", "single", "sentence_mean"],
        default="document",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=None,
        help="Batch size for frozen encoder inference. Defaults to --batch_size.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--train_samples_per_class",
        type=int,
        default=None,
        help=(
            "Use at most this many training examples per class, similar to "
            "MTEB's few-shot samples_per_label. Applied before "
            "--max_train_samples."
        ),
    )
    parser.add_argument(
        "--max_sentences_per_text",
        type=int,
        default=None,
        help="Sentence cap for document/sentence_mean modes. Defaults to data.max_sentences.",
    )
    parser.add_argument(
        "--no_normalize_embeddings",
        action="store_true",
        help="Disable L2 normalization before KNN. Usually keep normalization on.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/knn_probe")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    encode_batch_size = args.encode_batch_size or args.batch_size
    max_sentences = args.max_sentences_per_text
    if max_sentences is None:
        max_sentences = cfg["data"].get("max_sentences")

    print(f"Loading MTEB task: {args.task}")
    train_texts, train_labels, test_texts, test_labels, label_names = (
        load_mteb_classification_data(args.task)
    )
    train_texts, train_labels = subset_examples_per_class(
        train_texts,
        train_labels,
        args.train_samples_per_class,
        args.seed,
    )
    train_texts, train_labels = subset_examples(
        train_texts, train_labels, args.max_train_samples, args.seed
    )
    test_texts, test_labels = subset_examples(
        test_texts, test_labels, args.max_test_samples, args.seed
    )
    train_labels, test_labels = as_class_labels(train_labels, test_labels)

    print(f"Train samples: {len(train_texts):,}")
    print(f"Test samples:  {len(test_texts):,}")
    print(f"Classes:       {len(set(train_labels.tolist()) | set(test_labels.tolist())):,}")

    print(f"\nLoading frozen encoder: {args.checkpoint}")
    encoder = load_encoder(cfg, args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"], use_fast=True)
    max_length = cfg["encoder"]["max_seq_len"]

    print(
        f"\nEncoding with mode={args.embedding_mode}, "
        f"batch_size={encode_batch_size}, max_sentences={max_sentences}"
    )
    train_embs = encode_texts(
        encoder,
        tokenizer,
        train_texts,
        max_length,
        device,
        batch_size=encode_batch_size,
        embedding_mode=args.embedding_mode,
        max_sentences=max_sentences,
    )
    test_embs = encode_texts(
        encoder,
        tokenizer,
        test_texts,
        max_length,
        device,
        batch_size=encode_batch_size,
        embedding_mode=args.embedding_mode,
        max_sentences=max_sentences,
    )

    normalize = not args.no_normalize_embeddings
    train_embs_np = normalize_if_needed(train_embs, normalize)
    test_embs_np = normalize_if_needed(test_embs, normalize)

    print(
        f"Train embeddings: {train_embs_np.shape}, "
        f"test embeddings: {test_embs_np.shape}, normalize={normalize}"
    )
    print(f"\nRunning KNN: metric={args.metric}, weights={args.weights}, k={args.k}")
    rows, reports, best = evaluate_knn(
        train_embs_np,
        train_labels,
        test_embs_np,
        test_labels,
        label_names,
        args,
    )
    write_outputs(rows, reports, best, args)


if __name__ == "__main__":
    main()
