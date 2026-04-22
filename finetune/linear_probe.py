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
"""

import argparse
import json
import os

import mteb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.model import SentenceEncoder


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
def encode_texts(encoder, tokenizer, texts, max_length, device, batch_size=256):
    """Encode a list of strings into embeddings using the frozen encoder."""
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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = encoder(input_ids, attention_mask)  # (B, 1, H)

        all_embeddings.append(emb.squeeze(1).float().cpu())

    return torch.cat(all_embeddings, dim=0)


def load_mteb_classification_data(task_name):
    """Load an MTEB classification task and return texts + labels for train/test.

    Returns:
        train_texts, train_labels, test_texts, test_labels, label_names
    """
    tasks = mteb.get_tasks(tasks=[task_name])
    task = list(tasks)[0]
    task.load_data()

    ds = task.dataset

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
        for candidate in ["sentence", "sentence1", "query", "question"]:
            if candidate in train_cols:
                text_col = candidate
                break

    if "label" not in train_cols:
        for candidate in ["label_text", "labels", "class"]:
            if candidate in train_cols:
                label_col = candidate
                break

    train_texts = train_split[text_col]
    train_labels_raw = train_split[label_col]
    test_texts = test_split[text_col]
    test_labels_raw = test_split[label_col]

    # Convert string labels to ints if needed
    if isinstance(train_labels_raw[0], str):
        label_names = sorted(set(train_labels_raw) | set(test_labels_raw))
        label_to_id = {name: i for i, name in enumerate(label_names)}
        train_labels = [label_to_id[l] for l in train_labels_raw]
        test_labels = [label_to_id[l] for l in test_labels_raw]
    else:
        train_labels = list(train_labels_raw)
        test_labels = list(test_labels_raw)
        num_classes = max(max(train_labels), max(test_labels)) + 1
        label_names = [str(i) for i in range(num_classes)]

    return train_texts, train_labels, test_texts, test_labels, label_names


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load frozen encoder
    encoder = load_encoder(cfg, args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"], use_fast=True)
    max_length = cfg["encoder"]["max_seq_len"]
    hidden_size = cfg["encoder"]["hidden_size"]

    # Load dataset
    print(f"\nLoading MTEB task: {args.task}")
    train_texts, train_labels, test_texts, test_labels, label_names = \
        load_mteb_classification_data(args.task)
    num_classes = len(label_names)

    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,} samples")
    print(f"  Classes: {num_classes}")

    # Encode all texts (one-time cost with frozen encoder)
    print("\nEncoding train set...")
    train_embs = encode_texts(encoder, tokenizer, train_texts, max_length, device, args.batch_size)
    print("Encoding test set...")
    test_embs = encode_texts(encoder, tokenizer, test_texts, max_length, device, args.batch_size)

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

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    # Create data loaders (embeddings are on CPU, move per batch)
    train_ds = TensorDataset(train_embs, train_labels_t)
    test_ds = TensorDataset(test_embs, test_labels_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Linear probe
    linear = nn.Linear(hidden_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        linear.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining linear probe for {args.epochs} epochs...")
    print(f"  lr={args.lr}, batch_size={args.batch_size}, weight_decay={args.weight_decay}")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        linear.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            logits = linear(embs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embs.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += embs.size(0)

        scheduler.step()
        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # Evaluate
        linear.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for embs, labels in test_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = linear(embs)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += embs.size(0)

        test_acc = test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_state = linear.state_dict()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_acc={test_acc:.4f} {'*' if test_acc == best_acc else ''}"
            )

    # Save best model
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, f"{args.task}_linear_probe.pt")
    torch.save({
        "linear_state_dict": best_state,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "label_names": label_names,
        "task": args.task,
        "best_test_accuracy": best_acc,
        "best_epoch": best_epoch,
        "encoder_checkpoint": args.checkpoint,
        "config": args.config,
    }, model_path)
    print(f"\nSaved best model to {model_path}")

    # Save results
    result = {
        "task": args.task,
        "num_classes": num_classes,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "best_test_accuracy": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "checkpoint": args.checkpoint,
        "hidden_size": hidden_size,
    }

    output_path = os.path.join(args.output, f"{args.task}_linear_probe.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results: {args.task} (Linear Probe)")
    print(f"{'='*50}")
    print(f"  Best test accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Saved to {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
