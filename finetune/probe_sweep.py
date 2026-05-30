"""Probe every checkpoint in a folder and plot accuracy over training.

Usage:
    python -m finetune.probe_sweep \
        --config configs/colab.yaml \
        --checkpoint_dir /content/drive/MyDrive/SentenceJEPAModel \
        --task Banking77Classification \
        --device cuda \
        --epochs 20 \
        --batch_size 256

    python -m finetune.probe_sweep \
        --config configs/colab.yaml \
        --checkpoint_dir /content/drive/MyDrive/SentenceJEPAModel \
        --task AmazonPolarityClassification \
        --embedding_mode sentence_mean \
        --encode_batch_size 64 \
        --early_stop_patience 5 \
        --min_checkpoint_step 30000
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from finetune.linear_probe import (
    encode_texts,
    get_embedding_dim,
    load_mteb_classification_data,
)
from src.model import SentenceEncoder, SentenceJEPA


def checkpoint_step_from_name(path):
    """Extract step from filenames like step_10000.pt, or return None."""
    match = re.search(r"step[_-]?(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def discover_checkpoints(checkpoint_dir, pattern, recursive):
    """Find checkpoint files in a folder."""
    root = Path(checkpoint_dir)
    globber = root.rglob if recursive else root.glob
    checkpoints = [p for p in globber(pattern) if p.is_file()]
    checkpoints.sort(
        key=lambda p: (
            checkpoint_step_from_name(p) is None,
            checkpoint_step_from_name(p) or 0,
            str(p),
        )
    )
    return checkpoints


def filter_checkpoints_by_min_step(checkpoints, min_step):
    """Drop checkpoints whose filename step is known and below min_step."""
    if min_step is None:
        return checkpoints, 0, 0

    kept = []
    skipped = 0
    kept_unknown = 0
    for path in checkpoints:
        step = checkpoint_step_from_name(path)
        if step is None:
            kept.append(path)
            kept_unknown += 1
        elif step >= min_step:
            kept.append(path)
        else:
            skipped += 1
    return kept, skipped, kept_unknown


def normalize_step(step):
    """Convert checkpoint step metadata into a JSON/plot friendly integer."""
    if step is None:
        return None
    if isinstance(step, torch.Tensor):
        step = step.item()
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def load_encoder_and_step(cfg, checkpoint_path, device):
    """Load a frozen embedding model from a SentenceJEPA checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    if any(k.startswith("document_transformer.") for k in state_dict):
        encoder = SentenceJEPA(cfg).to(device)
        encoder.load_state_dict(state_dict, strict=False)
    else:
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        encoder = SentenceEncoder(cfg).to(device)
        encoder.load_state_dict(encoder_state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    step = ckpt.get("step", checkpoint_step_from_name(Path(checkpoint_path)))
    return encoder, step


def subset_examples(texts, labels, max_samples, seed):
    """Deterministically subsample a dataset for quick sweeps."""
    if max_samples is None or len(texts) <= max_samples:
        return texts, labels

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=generator)[:max_samples].tolist()
    return [texts[i] for i in indices], [labels[i] for i in indices]


def accuracy_from_logits(logits, labels):
    return (logits.argmax(dim=1) == labels).sum().item()


def train_probe(
    train_embs,
    train_labels,
    test_embs,
    test_labels,
    num_classes,
    epochs,
    lr,
    weight_decay,
    batch_size,
    device,
    seed,
    early_stop_patience=0,
    early_stop_min_delta=0.0,
    early_stop_min_epochs=0,
):
    """Train a linear classifier on frozen embeddings and return accuracy stats."""
    train_embs = F.normalize(train_embs, dim=1)
    test_embs = F.normalize(test_embs, dim=1)
    embedding_dim = train_embs.size(1)

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    train_ds = TensorDataset(train_embs, train_labels_t)
    test_ds = TensorDataset(test_embs, test_labels_t)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    linear = nn.Linear(embedding_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        linear.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    final_train_acc = 0.0
    final_test_acc = 0.0
    epochs_without_improvement = 0
    stopped_early = False
    history = []

    for epoch in range(epochs):
        linear.train()
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for embs, labels in train_loader:
            embs = embs.to(device)
            labels = labels.to(device)
            logits = linear(embs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * embs.size(0)
            train_correct += accuracy_from_logits(logits, labels)
            train_total += embs.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        train_loss = train_loss_total / train_total

        linear.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for embs, labels in test_loader:
                embs = embs.to(device)
                labels = labels.to(device)
                logits = linear(embs)
                test_correct += accuracy_from_logits(logits, labels)
                test_total += embs.size(0)

        test_acc = test_correct / test_total
        final_train_acc = train_acc
        final_test_acc = test_acc
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

        if test_acc > best_test_acc + early_stop_min_delta:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stop_patience > 0
            and epoch + 1 >= early_stop_min_epochs
            and epochs_without_improvement >= early_stop_patience
        ):
            stopped_early = True
            break

    return {
        "best_test_accuracy": best_test_acc,
        "train_accuracy_at_best": best_train_acc,
        "best_epoch": best_epoch,
        "final_train_accuracy": final_train_acc,
        "final_test_accuracy": final_test_acc,
        "epochs_trained": len(history),
        "stopped_early": stopped_early,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_min_epochs": early_stop_min_epochs,
        "epoch_history": history,
    }


def embedding_diagnostics(train_embs, test_embs):
    """Small embedding health snapshot for each checkpoint."""
    all_embs = torch.cat([train_embs, test_embs], dim=0)
    norms = all_embs.norm(dim=1)
    dim_var = all_embs.var(dim=0)
    normed = F.normalize(all_embs, dim=1)
    idx = torch.randperm(len(normed))[:min(2000, len(normed))]
    sample = normed[idx]
    cos_sim = sample @ sample.T
    triu_mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_sim[triu_mask]

    return {
        "emb_norm_mean": norms.mean().item(),
        "emb_norm_std": norms.std().item(),
        "emb_var_mean": dim_var.mean().item(),
        "emb_var_min": dim_var.min().item(),
        "emb_cosine_mean": pairwise_cos.mean().item(),
        "emb_cosine_std": pairwise_cos.std().item(),
        "emb_cosine_p95": pairwise_cos.quantile(0.95).item(),
    }


def save_results(results, output_dir, task):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{task}_checkpoint_probe.json")
    csv_path = os.path.join(output_dir, f"{task}_checkpoint_probe.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    flat_rows = []
    for row in results:
        flat_rows.append({
            k: v
            for k, v in row.items()
            if k != "epoch_history"
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    return json_path, csv_path


def plot_results(results, output_dir, task):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{task}_checkpoint_probe.png")

    has_steps = all(row["step"] is not None for row in results)
    x = [row["step"] if has_steps else i for i, row in enumerate(results)]
    xlabel = "checkpoint step" if has_steps else "checkpoint index"

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        [row["best_test_accuracy"] for row in results],
        marker="o",
        label="best test accuracy",
    )
    plt.plot(
        x,
        [row["train_accuracy_at_best"] for row in results],
        marker="o",
        label="train accuracy at best test",
    )
    plt.plot(
        x,
        [row["final_test_accuracy"] for row in results],
        marker="x",
        linestyle="--",
        label="final test accuracy",
    )
    plt.xlabel(xlabel)
    plt.ylabel("accuracy")
    plt.title(f"{task} linear probe over checkpoints")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probes for every checkpoint in a folder."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.pt")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--task", type=str, default="Banking77Classification")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/probe_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min_checkpoint_step",
        type=int,
        default=None,
        help=(
            "Skip checkpoints whose filename step is below this value. "
            "Checkpoints without a parseable step are kept."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Stop linear-probe training after this many epochs without a test "
            "accuracy improvement greater than --early_stop_min_delta. 0 disables."
        ),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum test accuracy improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=0,
        help="Do not early-stop before this many probe epochs have completed.",
    )
    parser.add_argument(
        "--embedding_mode",
        choices=["document", "document_layer_mean", "single", "sentence_mean"],
        default="document",
        help=(
            "document: contextual sentence mean from hierarchical Paragraph-JEPA. "
            "document_layer_mean: mean over sentence outputs from all document "
            "transformer layers. single: one-sequence probe. sentence_mean: "
            "independent sentence mean."
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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    checkpoints = discover_checkpoints(
        args.checkpoint_dir, args.pattern, args.recursive
    )
    checkpoints, skipped_early, kept_unknown = filter_checkpoints_by_min_step(
        checkpoints, args.min_checkpoint_step
    )
    if args.min_checkpoint_step is not None:
        print(
            f"Checkpoint step filter: kept {len(checkpoints)}, "
            f"skipped {skipped_early} below step {args.min_checkpoint_step}, "
            f"kept {kept_unknown} with unknown step"
        )
    if args.limit is not None:
        checkpoints = checkpoints[:args.limit]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {args.checkpoint_dir} with pattern {args.pattern}"
        )

    print(f"Found {len(checkpoints)} checkpoints:")
    for path in checkpoints:
        print(f"  {path}")

    print(f"\nLoading MTEB task: {args.task}")
    train_texts, train_labels, test_texts, test_labels, label_names = (
        load_mteb_classification_data(args.task)
    )
    train_texts, train_labels = subset_examples(
        train_texts, train_labels, args.max_train_samples, args.seed
    )
    test_texts, test_labels = subset_examples(
        test_texts, test_labels, args.max_test_samples, args.seed
    )
    num_classes = len(label_names)

    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,} samples")
    print(f"  Classes: {num_classes}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"], use_fast=True)
    max_length = cfg["encoder"]["max_seq_len"]
    config_embedding_dim = get_embedding_dim(cfg)
    encode_batch_size = args.encode_batch_size or args.batch_size
    max_sentences_per_text = args.max_sentences_per_text
    if max_sentences_per_text is None:
        max_sentences_per_text = cfg["data"].get("max_sentences")
    output_name = (
        args.task
        if args.embedding_mode == "single"
        else f"{args.task}_{args.embedding_mode}"
    )

    print(f"  Embedding mode: {args.embedding_mode}")
    if args.embedding_mode == "sentence_mean":
        print(f"  Max sentences/text: {max_sentences_per_text}")
        print(f"  Encode batch size: {encode_batch_size}")

    results = []
    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"\n[{idx}/{len(checkpoints)}] Probing {checkpoint_path}")
        encoder, step = load_encoder_and_step(cfg, checkpoint_path, device)

        print("  Encoding train set...")
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
        print("  Encoding test set...")
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

        embedding_dim = train_embs.size(1)
        if embedding_dim != config_embedding_dim:
            print(
                f"  NOTE: config embedding_dim={config_embedding_dim}, "
                f"encoder returned {embedding_dim}; using returned size."
            )

        diagnostics = embedding_diagnostics(train_embs, test_embs)
        probe_result = train_probe(
            train_embs=train_embs,
            train_labels=train_labels,
            test_embs=test_embs,
            test_labels=test_labels,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_min_epochs=args.early_stop_min_epochs,
        )

        row = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "step": normalize_step(step),
            "task": args.task,
            "embedding_dim": embedding_dim,
            "train_samples": len(train_texts),
            "test_samples": len(test_texts),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "encode_batch_size": encode_batch_size,
            "weight_decay": args.weight_decay,
            "embedding_mode": args.embedding_mode,
            "max_sentences_per_text": max_sentences_per_text,
            **diagnostics,
            **probe_result,
        }
        results.append(row)

        print(
            "  "
            f"best_test={row['best_test_accuracy']:.4f} "
            f"train_at_best={row['train_accuracy_at_best']:.4f} "
            f"final_test={row['final_test_accuracy']:.4f} "
            f"best_epoch={row['best_epoch']} "
            f"epochs_trained={row['epochs_trained']}/{args.epochs} "
            f"stopped_early={row['stopped_early']}"
        )

        del encoder, train_embs, test_embs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        json_path, csv_path = save_results(results, args.output, output_name)
        plot_path = plot_results(results, args.output, output_name)
        print(f"  Updated: {json_path}")
        print(f"  Updated: {csv_path}")
        print(f"  Updated: {plot_path}")

    best = max(results, key=lambda row: row["best_test_accuracy"])
    print("\nBest checkpoint:")
    print(f"  {best['checkpoint_name']}")
    print(f"  step={best['step']}")
    print(f"  best_test_accuracy={best['best_test_accuracy']:.4f}")
    print(f"  train_accuracy_at_best={best['train_accuracy_at_best']:.4f}")


if __name__ == "__main__":
    main()
