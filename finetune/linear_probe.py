"""Linear probe fine-tuning on MTEB classification datasets.

Freezes the SentenceJEPA encoder, trains a linear classifier on top.

Usage:
    python -m finetune.linear_probe \
        --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task Banking77Classification

    python -m finetune.linear_probe \
        --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task AmazonPolarityClassification \
        --epochs 10 --lr 1e-3 --batch_size 256

    python -m finetune.linear_probe \
        --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task AmazonPolarityClassification \
        --embedding_mode sentence_mean \
        --encode_batch_size 64

    python -m finetune.linear_probe \
        --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task TweetSentimentExtractionClassification \
        --label_mode ordinal
"""

import argparse
import json
import os
from contextlib import nullcontext

import mteb
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.data import split_sentences
from src.model import SentenceEncoder


def is_int_like(value):
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def all_int_like(values):
    return all(is_int_like(value) for value in values)


def get_embedding_dim(cfg):
    """Return the encoder output dimension used by the current projection head."""
    enc = cfg["encoder"]
    return enc.get("embedding_size", enc["hidden_size"])


def autocast_context(device):
    """Use the same bf16 inference path as eval_mteb when the backend supports it."""
    if device.type in ("cuda", "cpu"):
        return torch.amp.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def load_encoder(cfg, checkpoint_path, device):
    """Load frozen SentenceEncoder from a SentenceJEPA checkpoint."""
    encoder = SentenceEncoder(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    encoder_state = {
        k.removeprefix("encoder."): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    # Freeze all parameters
    for p in encoder.parameters():
        p.requires_grad = False

    step = ckpt.get("step", "?")
    print(f"Loaded frozen encoder from {checkpoint_path} (step {step})")
    return encoder


@torch.no_grad()
def encode_texts_single(encoder, tokenizer, texts, max_length, device, batch_size=256):
    """Encode each text as one truncated sequence: (B, T) -> (B, 1, T)."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # (B, T) -> (B, 1, T) for encoder
        input_ids = encoded["input_ids"].unsqueeze(1).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(1).to(device)

        with autocast_context(device):
            emb = encoder(input_ids, attention_mask)  # (B, 1, D)

        all_embeddings.append(emb.squeeze(1).float().cpu())

    return torch.cat(all_embeddings, dim=0)


def text_to_sentences(text, max_sentences=None):
    """Split one example into sentences, falling back to the raw text."""
    text = "" if text is None else str(text).strip()
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    return sentences or [text]


@torch.no_grad()
def encode_texts_sentence_mean(
    encoder,
    tokenizer,
    texts,
    max_length,
    device,
    batch_size=64,
    max_sentences=None,
):
    """Encode split sentences and mean-pool sentence embeddings per example."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        sentence_lists = [
            text_to_sentences(text, max_sentences=max_sentences)
            for text in batch
        ]
        flat_sentences = [
            sentence
            for sentences in sentence_lists
            for sentence in sentences
        ]

        encoded = tokenizer(
            flat_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        B = len(sentence_lists)
        S = max(len(sentences) for sentences in sentence_lists)
        T = encoded["input_ids"].size(1)

        input_ids = torch.zeros(B, S, T, dtype=torch.long)
        attention_mask = torch.zeros(B, S, T, dtype=torch.long)
        sentence_mask = torch.zeros(B, S, dtype=torch.bool)

        offset = 0
        for row, sentences in enumerate(sentence_lists):
            count = len(sentences)
            input_ids[row, :count] = encoded["input_ids"][offset : offset + count]
            attention_mask[row, :count] = encoded["attention_mask"][
                offset : offset + count
            ]
            sentence_mask[row, :count] = True
            offset += count

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        sentence_mask = sentence_mask.to(device)

        with autocast_context(device):
            sent_embs = encoder(input_ids, attention_mask)  # (B, S, D)
            weights = sentence_mask.unsqueeze(-1).to(dtype=sent_embs.dtype)
            emb = (sent_embs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

        all_embeddings.append(emb.float().cpu())

    return torch.cat(all_embeddings, dim=0)


def encode_texts(
    encoder,
    tokenizer,
    texts,
    max_length,
    device,
    batch_size=256,
    embedding_mode="single",
    max_sentences=None,
):
    """Encode strings with either single-sequence or sentence-mean pooling."""
    if embedding_mode == "single":
        return encode_texts_single(
            encoder, tokenizer, texts, max_length, device, batch_size
        )
    if embedding_mode == "sentence_mean":
        return encode_texts_sentence_mean(
            encoder,
            tokenizer,
            texts,
            max_length,
            device,
            batch_size,
            max_sentences=max_sentences,
        )
    raise ValueError(f"Unknown embedding mode: {embedding_mode}")


def first_dataset_split(dataset):
    """Unwrap common MTEB subset containers to a split dictionary."""
    if "train" in dataset or "training" in dataset:
        return dataset

    for subset in dataset.values():
        if "train" in subset or "training" in subset:
            return subset

    raise ValueError(f"No train split found. Available: {list(dataset.keys())}")


def load_mteb_classification_data(task_name):
    """Load an MTEB classification task and return texts + labels for train/test.

    Returns:
        train_texts, train_labels, test_texts, test_labels, label_names
    """
    tasks = mteb.get_tasks(tasks=[task_name])
    task = list(tasks)[0]
    task.load_data()

    ds = task.dataset
    ds = first_dataset_split(ds)

    # Find available splits
    if "train" in ds:
        train_split = ds["train"]
    elif "training" in ds:
        train_split = ds["training"]
    else:
        raise ValueError(f"No train split found. Available: {list(ds.keys())}")

    if "test" in ds:
        test_split = ds["test"]
    elif "validation" in ds:
        test_split = ds["validation"]
    else:
        raise ValueError(f"No test/validation split found. Available: {list(ds.keys())}")

    # Extract texts and labels
    text_col = "text"
    label_col = "label"

    # Some datasets use different column names
    train_cols = train_split.column_names
    if "text" not in train_cols:
        for candidate in ["sentence", "sentence1", "query", "question", "content"]:
            if candidate in train_cols:
                text_col = candidate
                break
        else:
            raise ValueError(
                f"No supported text column found. Available columns: {train_cols}"
            )

    if "label" not in train_cols:
        for candidate in ["label_text", "labels", "class"]:
            if candidate in train_cols:
                label_col = candidate
                break
        else:
            raise ValueError(
                f"No supported label column found. Available columns: {train_cols}"
            )

    train_texts = train_split[text_col]
    train_labels_raw = train_split[label_col]
    test_texts = test_split[text_col]
    test_labels_raw = test_split[label_col]

    # Convert string labels to ints if needed. Numeric labels are kept numeric;
    # integer-like labels receive class names, while non-integer labels can be
    # used by scalar regression probes.
    if isinstance(train_labels_raw[0], str):
        label_names = sorted(set(train_labels_raw) | set(test_labels_raw))
        label_to_id = {name: i for i, name in enumerate(label_names)}
        train_labels = [label_to_id[l] for l in train_labels_raw]
        test_labels = [label_to_id[l] for l in test_labels_raw]
    else:
        train_labels = list(train_labels_raw)
        test_labels = list(test_labels_raw)
        all_labels = train_labels + test_labels
        if all_int_like(all_labels):
            train_labels = [int(label) for label in train_labels]
            test_labels = [int(label) for label in test_labels]
            unique_labels = sorted(set(train_labels) | set(test_labels))
            if unique_labels == list(range(len(unique_labels))):
                label_names = [str(i) for i in unique_labels]
            else:
                label_names = [str(label) for label in unique_labels]
        else:
            train_labels = [float(label) for label in train_labels]
            test_labels = [float(label) for label in test_labels]
            label_names = []

    return train_texts, train_labels, test_texts, test_labels, label_names


def resolve_label_mode(label_mode, train_labels, test_labels, label_names):
    if label_mode != "auto":
        return label_mode

    labels = train_labels + test_labels
    if label_names and all_int_like(labels):
        return "classification"
    return "regression"


def scalar_probe_metrics(preds, labels, label_min=None, label_max=None):
    preds = preds.float()
    labels = labels.float()
    mse = F.mse_loss(preds, labels).item()
    mae = F.l1_loss(preds, labels).item()
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": mse ** 0.5,
    }
    if label_min is not None and label_max is not None:
        rounded = preds.round().clamp(label_min, label_max).long()
        metrics["rounded_accuracy"] = (
            rounded == labels.round().long()
        ).float().mean().item()
    return metrics


def clone_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def main():
    parser = argparse.ArgumentParser(description="Linear probe on MTEB classification tasks")
    parser.add_argument("--config", type=str, required=True, help="SentenceJEPA config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="SentenceJEPA checkpoint")
    parser.add_argument("--task", type=str, required=True, help="MTEB task name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label_mode",
        choices=["classification", "ordinal", "regression", "auto"],
        default="classification",
        help=(
            "classification: linear classifier with cross-entropy. ordinal: "
            "one-output scalar head for ordered integer labels. regression: "
            "one-output scalar head for continuous labels. auto chooses "
            "classification for integer-like class labels, regression otherwise."
        ),
    )
    parser.add_argument(
        "--scalar_loss",
        choices=["mse", "smooth_l1"],
        default="mse",
        help="Loss for ordinal/regression label modes.",
    )
    parser.add_argument(
        "--embedding_mode",
        choices=["single", "sentence_mean"],
        default="single",
        help=(
            "single: current one-sequence probe. sentence_mean: split each "
            "example into sentences, encode each sentence, then mean-pool."
        ),
    )
    parser.add_argument(
        "--max_sentences_per_text",
        type=int,
        default=None,
        help="Sentence cap for sentence_mean mode. Defaults to data.max_sentences.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=None,
        help="Batch size for frozen encoder inference. Defaults to --batch_size.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load frozen encoder
    encoder = load_encoder(cfg, args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"], use_fast=True)
    max_length = cfg["encoder"]["max_seq_len"]
    embedding_dim = get_embedding_dim(cfg)
    encode_batch_size = args.encode_batch_size or args.batch_size
    max_sentences_per_text = args.max_sentences_per_text
    if max_sentences_per_text is None:
        max_sentences_per_text = cfg["data"].get("max_sentences")

    # Load dataset
    print(f"\nLoading MTEB task: {args.task}")
    train_texts, train_labels, test_texts, test_labels, label_names = \
        load_mteb_classification_data(args.task)
    label_mode = resolve_label_mode(
        args.label_mode, train_labels, test_labels, label_names
    )
    scalar_mode = label_mode in ("ordinal", "regression")
    if label_mode == "classification" and not label_names:
        raise ValueError(
            "Classification mode requires discrete integer/string labels. Use "
            "--label_mode regression for continuous labels."
        )
    num_classes = len(label_names)
    all_labels = train_labels + test_labels
    label_min = min(all_labels) if scalar_mode else None
    label_max = max(all_labels) if scalar_mode else None
    rounded_scalar_metrics = scalar_mode and all_int_like(all_labels)

    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,} samples")
    print(f"  Label mode: {label_mode}")
    if scalar_mode:
        print(f"  Label range: [{label_min}, {label_max}]")
        print(f"  Scalar loss: {args.scalar_loss}")
        if rounded_scalar_metrics:
            print("  Rounded scalar accuracy will be reported.")
    else:
        print(f"  Classes: {num_classes}")
    print(f"  Embedding mode: {args.embedding_mode}")
    if args.embedding_mode == "sentence_mean":
        print(f"  Max sentences/text: {max_sentences_per_text}")
        print(f"  Encode batch size: {encode_batch_size}")

    # Encode all texts (one-time cost with frozen encoder)
    print("\nEncoding train set...")
    train_embs = encode_texts(
        encoder,
        tokenizer,
        train_texts,
        max_length,
        device,
        encode_batch_size,
        embedding_mode=args.embedding_mode,
        max_sentences=max_sentences_per_text,
    )
    print("Encoding test set...")
    test_embs = encode_texts(
        encoder,
        tokenizer,
        test_texts,
        max_length,
        device,
        encode_batch_size,
        embedding_mode=args.embedding_mode,
        max_sentences=max_sentences_per_text,
    )

    if train_embs.size(1) != embedding_dim:
        print(
            f"  NOTE: config embedding_dim={embedding_dim}, "
            f"but encoder returned {train_embs.size(1)}; using returned size."
        )
        embedding_dim = train_embs.size(1)

    # Embedding diagnostics
    all_embs = torch.cat([train_embs, test_embs], dim=0)
    norms = all_embs.norm(dim=1)
    dim_var = all_embs.var(dim=0)
    normed = F.normalize(all_embs, dim=1)
    # sample pairwise cosine sim (cap at 2000 to avoid OOM)
    idx = torch.randperm(len(normed))[:min(2000, len(normed))]
    sample = normed[idx]
    cos_sim = sample @ sample.T
    triu_mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_sim[triu_mask]

    print(f"\n  Embedding diagnostics:")
    print(f"    norm:       mean={norms.mean():.4f}  std={norms.std():.4f}")
    print(f"    dim var:    mean={dim_var.mean():.6f}  min={dim_var.min():.6f}")
    print(f"    cosine sim: mean={pairwise_cos.mean():.4f}  std={pairwise_cos.std():.4f}")
    if pairwise_cos.mean() > 0.9:
        print("    WARNING: embeddings are near-collapsed (very high cosine similarity)")
    if dim_var.min() < 1e-6:
        print("    WARNING: some dimensions have near-zero variance")

    # L2-normalize embeddings (standard practice for linear probes)
    train_embs = F.normalize(train_embs, dim=1)
    test_embs = F.normalize(test_embs, dim=1)

    if scalar_mode:
        train_labels_t = torch.tensor(train_labels, dtype=torch.float32)
        test_labels_t = torch.tensor(test_labels, dtype=torch.float32)
    else:
        train_labels_t = torch.tensor(train_labels, dtype=torch.long)
        test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    # Create data loaders (embeddings are on CPU, move per batch)
    train_ds = TensorDataset(train_embs, train_labels_t)
    test_ds = TensorDataset(test_embs, test_labels_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Linear probe
    output_dim = 1 if scalar_mode else num_classes
    linear = nn.Linear(embedding_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(
        linear.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining linear probe for {args.epochs} epochs...")
    print(f"  lr={args.lr}, batch_size={args.batch_size}, weight_decay={args.weight_decay}")

    best_acc = 0.0
    best_mae = float("inf")
    best_rmse = float("inf")
    best_rounded_acc = None
    best_epoch = 0
    best_state = clone_state_dict(linear)

    for epoch in range(args.epochs):
        # Train
        linear.train()
        total_loss = 0.0
        total_correct = 0
        total_abs_error = 0.0
        total_sq_error = 0.0
        total_rounded_correct = 0
        total_samples = 0

        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            outputs = linear(embs)
            if scalar_mode:
                preds = outputs.squeeze(-1)
                if args.scalar_loss == "smooth_l1":
                    loss = F.smooth_l1_loss(preds, labels)
                else:
                    loss = F.mse_loss(preds, labels)
            else:
                logits = outputs
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embs.size(0)
            if scalar_mode:
                errors = preds.detach() - labels
                total_abs_error += errors.abs().sum().item()
                total_sq_error += errors.pow(2).sum().item()
                if rounded_scalar_metrics:
                    rounded = preds.detach().round().clamp(label_min, label_max).long()
                    total_rounded_correct += (
                        rounded == labels.round().long()
                    ).sum().item()
            else:
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += embs.size(0)

        scheduler.step()
        train_loss = total_loss / total_samples
        if scalar_mode:
            train_mae = total_abs_error / total_samples
            train_rmse = (total_sq_error / total_samples) ** 0.5
            train_acc = (
                total_rounded_correct / total_samples
                if rounded_scalar_metrics
                else None
            )
        else:
            train_acc = total_correct / total_samples
            train_mae = None
            train_rmse = None

        # Evaluate
        linear.eval()
        test_correct = 0
        test_total = 0
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for embs, labels in test_loader:
                embs, labels = embs.to(device), labels.to(device)
                outputs = linear(embs)
                if scalar_mode:
                    preds = outputs.squeeze(-1)
                    test_preds.append(preds.cpu())
                    test_targets.append(labels.cpu())
                else:
                    logits = outputs
                    test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += embs.size(0)

        if scalar_mode:
            test_preds_t = torch.cat(test_preds)
            test_targets_t = torch.cat(test_targets)
            test_metrics = scalar_probe_metrics(
                test_preds_t,
                test_targets_t,
                label_min=label_min if rounded_scalar_metrics else None,
                label_max=label_max if rounded_scalar_metrics else None,
            )
            test_mae = test_metrics["mae"]
            test_rmse = test_metrics["rmse"]
            test_acc = test_metrics.get("rounded_accuracy")
            if label_mode == "ordinal" and test_acc is not None:
                improved = test_acc > best_acc
            else:
                improved = test_mae < best_mae
        else:
            test_acc = test_correct / test_total
            test_mae = None
            test_rmse = None
            improved = test_acc > best_acc

        if improved:
            if test_acc is not None:
                best_acc = test_acc
            if test_mae is not None:
                best_mae = test_mae
            if test_rmse is not None:
                best_rmse = test_rmse
            best_rounded_acc = test_acc
            best_epoch = epoch + 1
            best_state = clone_state_dict(linear)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if scalar_mode:
                train_acc_str = (
                    f" train_round_acc={train_acc:.4f}"
                    if train_acc is not None
                    else ""
                )
                test_acc_str = (
                    f" test_round_acc={test_acc:.4f}"
                    if test_acc is not None
                    else ""
                )
                print(
                    f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
                    f"train_rmse={train_rmse:.4f}{train_acc_str} | "
                    f"test_mae={test_mae:.4f} test_rmse={test_rmse:.4f}"
                    f"{test_acc_str} {'*' if improved else ''}"
                )
            else:
                print(
                    f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"test_acc={test_acc:.4f} {'*' if improved else ''}"
                )

    # Save best model
    os.makedirs(args.output, exist_ok=True)
    output_name = (
        args.task
        if args.embedding_mode == "single"
        else f"{args.task}_{args.embedding_mode}"
    )
    model_path = os.path.join(args.output, f"{output_name}_linear_probe.pt")
    torch.save({
        "linear_state_dict": best_state,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "label_names": label_names,
        "label_mode": label_mode,
        "scalar_loss": args.scalar_loss if scalar_mode else None,
        "label_min": label_min,
        "label_max": label_max,
        "task": args.task,
        "embedding_mode": args.embedding_mode,
        "max_sentences_per_text": max_sentences_per_text,
        "best_test_accuracy": best_acc if not scalar_mode else best_rounded_acc,
        "best_test_mae": None if best_mae == float("inf") else best_mae,
        "best_test_rmse": None if best_rmse == float("inf") else best_rmse,
        "best_rounded_accuracy": best_rounded_acc,
        "best_epoch": best_epoch,
        "encoder_checkpoint": args.checkpoint,
        "config": args.config,
    }, model_path)
    print(f"\nSaved best model to {model_path}")

    # Save results
    result = {
        "task": args.task,
        "label_mode": label_mode,
        "scalar_loss": args.scalar_loss if scalar_mode else None,
        "num_classes": num_classes,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "best_test_accuracy": best_acc if not scalar_mode else best_rounded_acc,
        "best_test_mae": None if best_mae == float("inf") else best_mae,
        "best_test_rmse": None if best_rmse == float("inf") else best_rmse,
        "best_rounded_accuracy": best_rounded_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "checkpoint": args.checkpoint,
        "embedding_dim": embedding_dim,
        "embedding_mode": args.embedding_mode,
        "max_sentences_per_text": max_sentences_per_text,
        "encode_batch_size": encode_batch_size,
    }

    output_path = os.path.join(args.output, f"{output_name}_linear_probe.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results: {args.task} (Linear Probe)")
    print(f"{'='*50}")
    if scalar_mode:
        print(f"  Best epoch: {best_epoch}")
        if best_rounded_acc is not None:
            print(f"  Best rounded accuracy: {best_rounded_acc:.4f}")
        if best_mae != float("inf"):
            print(f"  Best test MAE: {best_mae:.4f}")
        if best_rmse != float("inf"):
            print(f"  Best test RMSE: {best_rmse:.4f}")
    else:
        print(f"  Best test accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Saved to {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
