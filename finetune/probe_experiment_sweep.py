"""Sweep linear probes across experiment folders and summarize the best runs.

Each experiment folder is expected to contain checkpoints from one training run
or hyperparameter setting. The script probes every checkpoint for one or more
MTEB classification tasks, then writes:

  - all_probe_results.csv/json: every checkpoint-task result
  - best_linear_probe_by_folder.csv/json: best checkpoint per folder/task
  - linear_probe_summary.md: compact per-task leaderboard

Usage:
    python -m finetune.probe_experiment_sweep \
        --config configs/colab.yaml \
        --root_dir /content/drive/MyDrive/SentenceJEPARuns \
        --tasks Banking77Classification EmotionClassification \
        --device cuda \
        --epochs 20 \
        --batch_size 256

    python -m finetune.probe_experiment_sweep \
        --config configs/colab.yaml \
        --root_dir /content/drive/MyDrive/SentenceJEPARuns \
        --tasks AmazonPolarityClassification \
        --embedding_mode sentence_mean \
        --encode_batch_size 512 \
        --precision auto
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from finetune.linear_probe import (
    get_embedding_dim,
    load_mteb_classification_data,
    text_to_sentences,
)
from finetune.probe_sweep import (
    checkpoint_step_from_name,
    discover_checkpoints,
    normalize_step,
    subset_examples,
    train_probe,
)
from src.model import SentenceEncoder


PREFERRED_FIELDS = [
    "task",
    "experiment_name",
    "experiment_dir",
    "checkpoint_name",
    "checkpoint",
    "step",
    "best_test_accuracy",
    "train_accuracy_at_best",
    "best_epoch",
    "final_test_accuracy",
    "final_train_accuracy",
    "embedding_mode",
    "embedding_dim",
    "objective_mode",
    "sigreg_weight",
    "sigreg_enabled",
    "mask_ratio_min",
    "mask_ratio_max",
    "max_mask_count",
    "multi_mask",
    "encoder_num_layers",
    "encoder_hidden_size",
    "encoder_embedding_size",
    "predictor_num_layers",
    "predictor_hidden_size",
    "training_lr",
    "training_batch_size",
    "training_precision",
    "epochs",
    "probe_lr",
    "probe_batch_size",
    "probe_weight_decay",
    "encode_batch_size",
    "precision",
    "max_sentences_per_text",
    "train_samples",
    "test_samples",
    "diagnostic_samples",
    "emb_norm_mean",
    "emb_norm_std",
    "emb_var_mean",
    "emb_var_min",
    "emb_cosine_mean",
    "emb_cosine_std",
    "emb_cosine_p95",
]


def get_nested(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def discover_experiment_dirs(root_dir, pattern, recursive_folders):
    """Return directories that contain checkpoint files."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    if recursive_folders:
        dirs = sorted({path.parent for path in root.rglob(pattern) if path.is_file()})
    else:
        dirs = []
        if any(path.is_file() for path in root.glob(pattern)):
            dirs.append(root)
        for child in sorted(root.iterdir()):
            if child.is_dir() and any(path.is_file() for path in child.glob(pattern)):
                dirs.append(child)

    return dirs


def experiment_name(root_dir, experiment_dir):
    root = Path(root_dir).resolve()
    directory = Path(experiment_dir).resolve()
    try:
        return directory.relative_to(root).as_posix() or directory.name
    except ValueError:
        return directory.name


def extract_hparams(cfg):
    """Flatten the most useful pretraining hyperparameters for the summary CSV."""
    return {
        "objective_mode": get_nested(cfg, ["objective", "mode"]),
        "sigreg_weight": get_nested(cfg, ["sigreg", "weight"]),
        "sigreg_enabled": get_nested(cfg, ["sigreg", "enabled"]),
        "sigreg_num_projections": get_nested(cfg, ["sigreg", "num_projections"]),
        "mask_ratio_min": get_nested(cfg, ["masking", "mask_ratio_min"]),
        "mask_ratio_max": get_nested(cfg, ["masking", "mask_ratio_max"]),
        "max_mask_count": get_nested(cfg, ["masking", "max_mask_count"]),
        "multi_mask": get_nested(cfg, ["masking", "multi_mask"]),
        "encoder_num_layers": get_nested(cfg, ["encoder", "num_layers"]),
        "encoder_hidden_size": get_nested(cfg, ["encoder", "hidden_size"]),
        "encoder_embedding_size": get_nested(cfg, ["encoder", "embedding_size"]),
        "predictor_num_layers": get_nested(cfg, ["predictor", "num_layers"]),
        "predictor_hidden_size": get_nested(cfg, ["predictor", "hidden_size"]),
        "training_lr": get_nested(cfg, ["training", "lr"]),
        "training_batch_size": get_nested(cfg, ["training", "batch_size"]),
        "training_precision": get_nested(cfg, ["training", "precision"]),
        "training_epochs": get_nested(cfg, ["training", "epochs"]),
        "data_max_sentences": get_nested(cfg, ["data", "max_sentences"]),
        "data_max_tokens_per_sentence": get_nested(
            cfg, ["data", "max_tokens_per_sentence"]
        ),
    }


def load_encoder_and_config(fallback_cfg, checkpoint_path, device, use_checkpoint_config):
    """Load a frozen encoder and return checkpoint step/config metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("config")
    cfg = (
        checkpoint_cfg
        if use_checkpoint_config and isinstance(checkpoint_cfg, dict)
        else fallback_cfg
    )
    config_source = "checkpoint" if cfg is checkpoint_cfg else "cli_config"

    state_dict = checkpoint["model_state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* weights found in {checkpoint_path}")

    encoder = SentenceEncoder(cfg).to(device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    step = checkpoint.get("step", checkpoint_step_from_name(Path(checkpoint_path)))
    return encoder, normalize_step(step), cfg, config_source


def load_task_data(task_names, max_train_samples, max_test_samples, seed):
    """Load and optionally subsample all requested MTEB classification tasks."""
    task_data = {}
    for task_name in task_names:
        print(f"\nLoading MTEB task: {task_name}")
        train_texts, train_labels, test_texts, test_labels, label_names = (
            load_mteb_classification_data(task_name)
        )
        train_texts, train_labels = subset_examples(
            train_texts, train_labels, max_train_samples, seed
        )
        test_texts, test_labels = subset_examples(
            test_texts, test_labels, max_test_samples, seed
        )
        task_data[task_name] = {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_labels": test_labels,
            "label_names": label_names,
            "num_classes": len(label_names),
        }
        print(
            f"  Train: {len(train_texts):,} | "
            f"Test: {len(test_texts):,} | "
            f"Classes: {len(label_names)}"
        )
    return task_data


def choose_default_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def configure_torch(device, allow_tf32):
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_precision(device, precision):
    """Return autocast settings for fast encoder inference."""
    if precision == "fp32" or device.type != "cuda":
        return False, torch.float32, "fp32"
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if precision == "bf16":
        if not is_bf16_supported():
            print("Requested bf16, but this CUDA device does not support it; using fp16.")
            return True, torch.float16, "fp16"
        return True, torch.bfloat16, "bf16"
    if precision == "fp16":
        return True, torch.float16, "fp16"

    if torch.cuda.is_available() and is_bf16_supported():
        return True, torch.bfloat16, "bf16"
    return True, torch.float16, "fp16"


def autocast_context(device, enabled, dtype):
    if enabled and device.type in ("cuda", "cpu"):
        return torch.amp.autocast(device.type, dtype=dtype)
    return nullcontext()


def batch_items(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def maybe_pin(tensor, pin_memory):
    if pin_memory:
        return tensor.pin_memory()
    return tensor


def parallel_sentence_lists(texts, max_sentences, num_workers):
    if num_workers <= 1 or len(texts) < 1024:
        return [
            text_to_sentences(text, max_sentences=max_sentences)
            for text in texts
        ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(
            executor.map(
                lambda text: text_to_sentences(text, max_sentences=max_sentences),
                texts,
            )
        )


def prepare_single_batches(texts, tokenizer, max_length, batch_size, pin_memory):
    """Tokenize texts once as (B, 1, T) encoder batches."""
    batches = []
    for text_batch in batch_items(texts, batch_size):
        encoded = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].unsqueeze(1).contiguous()
        attention_mask = encoded["attention_mask"].unsqueeze(1).contiguous()
        sentence_mask = torch.ones(input_ids.size(0), 1, dtype=torch.bool)
        batches.append(
            {
                "input_ids": maybe_pin(input_ids, pin_memory),
                "attention_mask": maybe_pin(attention_mask, pin_memory),
                "sentence_mask": maybe_pin(sentence_mask, pin_memory),
            }
        )
    return batches


def prepare_sentence_mean_batches(
    texts,
    tokenizer,
    max_length,
    batch_size,
    max_sentences,
    pin_memory,
    num_workers,
):
    """Split/tokenize texts once as (B, S, T) encoder batches."""
    sentence_lists = parallel_sentence_lists(texts, max_sentences, num_workers)
    batches = []

    for list_batch in batch_items(sentence_lists, batch_size):
        flat_sentences = [
            sentence
            for sentences in list_batch
            for sentence in sentences
        ]
        encoded = tokenizer(
            flat_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        B = len(list_batch)
        S = max(len(sentences) for sentences in list_batch)
        T = encoded["input_ids"].size(1)

        input_ids = torch.zeros(B, S, T, dtype=torch.long)
        attention_mask = torch.zeros(B, S, T, dtype=torch.long)
        sentence_mask = torch.zeros(B, S, dtype=torch.bool)

        offset = 0
        for row, sentences in enumerate(list_batch):
            count = len(sentences)
            input_ids[row, :count] = encoded["input_ids"][offset : offset + count]
            attention_mask[row, :count] = encoded["attention_mask"][
                offset : offset + count
            ]
            sentence_mask[row, :count] = True
            offset += count

        batches.append(
            {
                "input_ids": maybe_pin(input_ids, pin_memory),
                "attention_mask": maybe_pin(attention_mask, pin_memory),
                "sentence_mask": maybe_pin(sentence_mask, pin_memory),
            }
        )

    return batches


def prepare_batches(
    texts,
    tokenizer,
    max_length,
    embedding_mode,
    max_sentences,
    batch_size,
    pin_memory,
    num_workers,
):
    if embedding_mode == "single":
        return prepare_single_batches(
            texts,
            tokenizer,
            max_length,
            batch_size,
            pin_memory,
        )
    if embedding_mode == "sentence_mean":
        return prepare_sentence_mean_batches(
            texts,
            tokenizer,
            max_length,
            batch_size,
            max_sentences,
            pin_memory,
            num_workers,
        )
    raise ValueError(f"Unknown embedding mode: {embedding_mode}")


def prepared_cache_key(
    task_name,
    split_name,
    tokenizer_name,
    max_length,
    embedding_mode,
    max_sentences,
    batch_size,
):
    return (
        task_name,
        split_name,
        tokenizer_name,
        int(max_length),
        embedding_mode,
        int(max_sentences) if max_sentences is not None else None,
        int(batch_size),
    )


def get_prepared_batches(
    cache,
    task_name,
    split_name,
    texts,
    tokenizer_name,
    tokenizer,
    max_length,
    embedding_mode,
    max_sentences,
    batch_size,
    pin_memory,
    num_workers,
):
    key = prepared_cache_key(
        task_name,
        split_name,
        tokenizer_name,
        max_length,
        embedding_mode,
        max_sentences,
        batch_size,
    )
    if key not in cache:
        print(
            f"      Preparing {task_name}/{split_name}: "
            f"{len(texts):,} examples, mode={embedding_mode}, "
            f"batch={batch_size}"
        )
        cache[key] = prepare_batches(
            texts,
            tokenizer,
            max_length,
            embedding_mode,
            max_sentences,
            batch_size,
            pin_memory,
            num_workers,
        )
    return cache[key]


def encode_prepared_batches(
    encoder,
    batches,
    device,
    embedding_mode,
    amp_enabled,
    amp_dtype,
):
    """Run frozen encoder over pre-tokenized CPU batches."""
    embeddings = []
    with torch.inference_mode():
        for batch in batches:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            sentence_mask = batch["sentence_mask"].to(device, non_blocking=True)

            with autocast_context(device, amp_enabled, amp_dtype):
                sent_embs = encoder(input_ids, attention_mask)
                if embedding_mode == "single":
                    emb = sent_embs.squeeze(1)
                else:
                    weights = sentence_mask.unsqueeze(-1).to(dtype=sent_embs.dtype)
                    emb = (
                        (sent_embs * weights).sum(dim=1)
                        / weights.sum(dim=1).clamp(min=1)
                    )

            embeddings.append(emb.float().cpu())

    return torch.cat(embeddings, dim=0)


def embedding_diagnostics_fast(train_embs, test_embs, sample_size, seed):
    if sample_size <= 1:
        return {}

    all_embs = torch.cat([train_embs, test_embs], dim=0)
    norms = all_embs.norm(dim=1)
    dim_var = all_embs.var(dim=0)
    normed = F.normalize(all_embs, dim=1)

    sample_count = min(sample_size, len(normed))
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(normed), generator=generator)[:sample_count]
    sample = normed[idx]
    cos_sim = sample @ sample.T
    triu_mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_sim[triu_mask]

    diagnostics = {
        "emb_norm_mean": norms.mean().item(),
        "emb_norm_std": norms.std().item(),
        "emb_var_mean": dim_var.mean().item(),
        "emb_var_min": dim_var.min().item(),
    }
    if pairwise_cos.numel() > 0:
        diagnostics.update(
            {
                "emb_cosine_mean": pairwise_cos.mean().item(),
                "emb_cosine_std": pairwise_cos.std().item(),
                "emb_cosine_p95": pairwise_cos.quantile(0.95).item(),
            }
        )
    return diagnostics


def train_probe_fast(
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
):
    """Train the linear probe with embeddings resident on the selected device."""
    train_embs = F.normalize(train_embs, dim=1).to(device, non_blocking=True)
    test_embs = F.normalize(test_embs, dim=1).to(device, non_blocking=True)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)
    embedding_dim = train_embs.size(1)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    linear = nn.Linear(embedding_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        linear.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    final_train_acc = 0.0
    final_test_acc = 0.0
    history = []

    for epoch in range(epochs):
        linear.train()
        permutation = torch.randperm(train_embs.size(0), device=device)
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for start in range(0, train_embs.size(0), batch_size):
            idx = permutation[start : start + batch_size]
            embs = train_embs.index_select(0, idx)
            labels = train_labels.index_select(0, idx)

            logits = linear(embs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * embs.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += embs.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        train_loss = train_loss_total / train_total

        linear.eval()
        test_correct = 0
        test_total = 0
        with torch.inference_mode():
            for start in range(0, test_embs.size(0), batch_size):
                embs = test_embs[start : start + batch_size]
                labels = test_labels[start : start + batch_size]
                logits = linear(embs)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += embs.size(0)

        test_acc = test_correct / test_total
        final_train_acc = train_acc
        final_test_acc = test_acc
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_epoch = epoch + 1

    return {
        "best_test_accuracy": best_test_acc,
        "train_accuracy_at_best": best_train_acc,
        "best_epoch": best_epoch,
        "final_train_accuracy": final_train_acc,
        "final_test_accuracy": final_test_acc,
        "epoch_history": history,
    }


def csv_value(value):
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def flat_rows_for_csv(rows):
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: csv_value(value)
                for key, value in row.items()
                if key != "epoch_history"
            }
        )
    return flat_rows


def write_csv(rows, path):
    flat_rows = flat_rows_for_csv(rows)
    if not flat_rows:
        return None

    keys = {key for row in flat_rows for key in row}
    fieldnames = [key for key in PREFERRED_FIELDS if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return path


def best_rows_by_folder(rows):
    best = {}
    for row in rows:
        key = (row["task"], row["experiment_name"], row["embedding_mode"])
        score = row.get("best_test_accuracy", float("-inf"))
        step = row.get("step")
        tie_break = step if step is not None else -1
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_score = current.get("best_test_accuracy", float("-inf"))
        current_step = current.get("step")
        current_tie_break = current_step if current_step is not None else -1
        if (score, tie_break) > (current_score, current_tie_break):
            best[key] = row

    return sorted(
        best.values(),
        key=lambda row: (
            row["task"],
            -row.get("best_test_accuracy", 0.0),
            row["experiment_name"],
        ),
    )


def write_summary_markdown(best_rows, path):
    tasks = sorted({row["task"] for row in best_rows})
    lines = ["# Linear Probe Summary", ""]
    for task in tasks:
        task_rows = [
            row for row in best_rows
            if row["task"] == task
        ]
        task_rows.sort(key=lambda row: row.get("best_test_accuracy", 0.0), reverse=True)

        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            "| Rank | Experiment | Best Test | Train At Best | Step | "
            "Objective | SIGReg | Mask | Mode |"
        )
        lines.append(
            "| ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |"
        )
        for rank, row in enumerate(task_rows, start=1):
            mask = (
                f"{row.get('mask_ratio_min')}-"
                f"{row.get('mask_ratio_max')}"
            )
            lines.append(
                f"| {rank} | {row['experiment_name']} | "
                f"{row.get('best_test_accuracy', 0.0):.4f} | "
                f"{row.get('train_accuracy_at_best', 0.0):.4f} | "
                f"{row.get('step')} | "
                f"{row.get('objective_mode')} | "
                f"{row.get('sigreg_weight')} | "
                f"{mask} | "
                f"{row.get('embedding_mode')} |"
            )
        lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def write_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_json_path = os.path.join(output_dir, "all_probe_results.json")
    all_csv_path = os.path.join(output_dir, "all_probe_results.csv")
    best_json_path = os.path.join(output_dir, "best_linear_probe_by_folder.json")
    best_csv_path = os.path.join(output_dir, "best_linear_probe_by_folder.csv")
    summary_path = os.path.join(output_dir, "linear_probe_summary.md")

    with open(all_json_path, "w") as handle:
        json.dump(rows, handle, indent=2)
    write_csv(rows, all_csv_path)

    best_rows = best_rows_by_folder(rows)
    with open(best_json_path, "w") as handle:
        json.dump(best_rows, handle, indent=2)
    write_csv(best_rows, best_csv_path)
    write_summary_markdown(best_rows, summary_path)

    return {
        "all_json": all_json_path,
        "all_csv": all_csv_path,
        "best_json": best_json_path,
        "best_csv": best_csv_path,
        "summary": summary_path,
    }


def print_task_leaders(best_rows, top_k):
    tasks = sorted({row["task"] for row in best_rows})
    for task in tasks:
        print(f"\nTop {top_k} for {task}:")
        task_rows = [
            row for row in best_rows
            if row["task"] == task
        ]
        task_rows.sort(key=lambda row: row.get("best_test_accuracy", 0.0), reverse=True)
        for rank, row in enumerate(task_rows[:top_k], start=1):
            print(
                f"  {rank:>2}. {row['experiment_name']} | "
                f"best_test={row.get('best_test_accuracy', 0.0):.4f} | "
                f"train_at_best={row.get('train_accuracy_at_best', 0.0):.4f} | "
                f"step={row.get('step')} | "
                f"sigreg={row.get('sigreg_weight')} | "
                f"objective={row.get('objective_mode')}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run linear probes across all checkpoint folders under a root directory "
            "and summarize the best checkpoint per folder/task."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--pattern", type=str, default="*.pt")
    parser.add_argument(
        "--recursive_folders",
        action="store_true",
        help="Treat every nested directory containing checkpoints as an experiment.",
    )
    parser.add_argument(
        "--recursive_checkpoints",
        action="store_true",
        help="Find checkpoints recursively within each experiment folder.",
    )
    parser.add_argument("--folder_limit", type=int, default=None)
    parser.add_argument("--checkpoint_limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/probe_experiment_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--embedding_mode",
        choices=["single", "sentence_mean"],
        default="single",
    )
    parser.add_argument(
        "--max_sentences_per_text",
        type=int,
        default=None,
        help="Sentence cap for sentence_mean mode. Defaults to checkpoint data.max_sentences.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=None,
        help="Batch size for frozen encoder inference. Defaults to --batch_size.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Autocast precision for encoder inference. auto uses bf16 if supported, else fp16 on CUDA.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="CPU workers for sentence splitting. Defaults to min(8, cpu_count).",
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_true",
        help="Disable pinned CPU batches before GPU transfer.",
    )
    parser.add_argument(
        "--no_tf32",
        action="store_true",
        help="Disable TF32 matmul on CUDA devices that support it.",
    )
    parser.add_argument(
        "--compile_encoder",
        action="store_true",
        help="Compile each encoder with torch.compile. Useful only when many batches share shapes.",
    )
    parser.add_argument(
        "--probe_on_cpu",
        action="store_true",
        help="Train the linear probe through the older CPU-batch DataLoader path.",
    )
    parser.add_argument(
        "--diagnostic_samples",
        type=int,
        default=1000,
        help="Pairwise cosine sample size for embedding diagnostics. Use 0 to skip.",
    )
    parser.add_argument(
        "--write_every",
        type=int,
        default=10,
        help="Write CSV/JSON progress every N completed rows. Final output is always written.",
    )
    parser.add_argument(
        "--no_checkpoint_config",
        action="store_true",
        help="Use --config for all encoders instead of the config saved in each checkpoint.",
    )
    parser.add_argument(
        "--save_history",
        action="store_true",
        help="Store per-epoch linear-probe history in the JSON output.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of leaders to print per task at the end.",
    )
    args = parser.parse_args()

    with open(args.config) as handle:
        fallback_cfg = yaml.safe_load(handle)

    device = torch.device(args.device)
    encode_batch_size = args.encode_batch_size or args.batch_size
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = choose_default_workers()
    pin_memory = (device.type == "cuda") and not args.no_pin_memory
    configure_torch(device, allow_tf32=not args.no_tf32)
    amp_enabled, amp_dtype, resolved_precision = resolve_precision(
        device, args.precision
    )
    use_checkpoint_config = not args.no_checkpoint_config

    print(
        "Runtime: "
        f"device={device}, precision={resolved_precision}, "
        f"encode_batch_size={encode_batch_size}, "
        f"probe_batch_size={args.batch_size}, "
        f"pin_memory={pin_memory}, workers={num_workers}, "
        f"tf32={not args.no_tf32}"
    )

    experiment_dirs = discover_experiment_dirs(
        args.root_dir,
        args.pattern,
        recursive_folders=args.recursive_folders,
    )
    if args.folder_limit is not None:
        experiment_dirs = experiment_dirs[: args.folder_limit]
    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment folders with {args.pattern} found under {args.root_dir}"
        )

    print(f"Found {len(experiment_dirs)} experiment folder(s):")
    for directory in experiment_dirs:
        print(f"  {experiment_name(args.root_dir, directory)} -> {directory}")

    task_data = load_task_data(
        args.tasks,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    tokenizer_cache = {}
    prepared_cache = {}
    rows = []

    for folder_idx, directory in enumerate(experiment_dirs, start=1):
        exp_name = experiment_name(args.root_dir, directory)
        checkpoints = discover_checkpoints(
            directory,
            args.pattern,
            recursive=args.recursive_checkpoints,
        )
        if args.checkpoint_limit is not None:
            checkpoints = checkpoints[: args.checkpoint_limit]
        if not checkpoints:
            continue

        print(
            f"\n[{folder_idx}/{len(experiment_dirs)}] {exp_name}: "
            f"{len(checkpoints)} checkpoint(s)"
        )

        for ckpt_idx, checkpoint_path in enumerate(checkpoints, start=1):
            print(f"\n  [{ckpt_idx}/{len(checkpoints)}] {checkpoint_path.name}")
            encoder, step, cfg, config_source = load_encoder_and_config(
                fallback_cfg,
                checkpoint_path,
                device,
                use_checkpoint_config=use_checkpoint_config,
            )
            if args.compile_encoder:
                encoder = torch.compile(encoder)

            tokenizer_name = get_nested(
                cfg,
                ["data", "tokenizer"],
                get_nested(fallback_cfg, ["data", "tokenizer"]),
            )
            if tokenizer_name not in tokenizer_cache:
                tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    use_fast=True,
                )
            tokenizer = tokenizer_cache[tokenizer_name]

            max_length = get_nested(
                cfg,
                ["encoder", "max_seq_len"],
                get_nested(fallback_cfg, ["encoder", "max_seq_len"]),
            )
            max_sentences_per_text = args.max_sentences_per_text
            if max_sentences_per_text is None:
                max_sentences_per_text = get_nested(
                    cfg,
                    ["data", "max_sentences"],
                    get_nested(fallback_cfg, ["data", "max_sentences"]),
                )

            hparams = extract_hparams(cfg)
            config_embedding_dim = get_embedding_dim(cfg)

            for task_name in args.tasks:
                data = task_data[task_name]
                print(f"    Probing {task_name}")

                train_batches = get_prepared_batches(
                    prepared_cache,
                    task_name,
                    "train",
                    data["train_texts"],
                    tokenizer_name,
                    tokenizer,
                    max_length,
                    args.embedding_mode,
                    max_sentences_per_text,
                    encode_batch_size,
                    pin_memory,
                    num_workers,
                )
                test_batches = get_prepared_batches(
                    prepared_cache,
                    task_name,
                    "test",
                    data["test_texts"],
                    tokenizer_name,
                    tokenizer,
                    max_length,
                    args.embedding_mode,
                    max_sentences_per_text,
                    encode_batch_size,
                    pin_memory,
                    num_workers,
                )

                train_embs = encode_prepared_batches(
                    encoder,
                    train_batches,
                    device,
                    args.embedding_mode,
                    amp_enabled,
                    amp_dtype,
                )
                test_embs = encode_prepared_batches(
                    encoder,
                    test_batches,
                    device,
                    args.embedding_mode,
                    amp_enabled,
                    amp_dtype,
                )

                embedding_dim = train_embs.size(1)
                if embedding_dim != config_embedding_dim:
                    print(
                        f"      NOTE: config embedding_dim={config_embedding_dim}, "
                        f"encoder returned {embedding_dim}; using returned size."
                    )

                diagnostics = embedding_diagnostics_fast(
                    train_embs,
                    test_embs,
                    sample_size=args.diagnostic_samples,
                    seed=args.seed,
                )
                if args.probe_on_cpu:
                    probe_result = train_probe(
                        train_embs=train_embs,
                        train_labels=data["train_labels"],
                        test_embs=test_embs,
                        test_labels=data["test_labels"],
                        num_classes=data["num_classes"],
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        device=device,
                        seed=args.seed,
                    )
                else:
                    probe_result = train_probe_fast(
                        train_embs=train_embs,
                        train_labels=data["train_labels"],
                        test_embs=test_embs,
                        test_labels=data["test_labels"],
                        num_classes=data["num_classes"],
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        device=device,
                        seed=args.seed,
                    )
                if not args.save_history:
                    probe_result.pop("epoch_history", None)

                row = {
                    "task": task_name,
                    "experiment_name": exp_name,
                    "experiment_dir": str(directory),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "step": step,
                    "checkpoint_index": ckpt_idx,
                    "checkpoints_in_folder": len(checkpoints),
                    "config_source": config_source,
                    "tokenizer": tokenizer_name,
                    "embedding_mode": args.embedding_mode,
                    "embedding_dim": embedding_dim,
                    "max_sentences_per_text": max_sentences_per_text,
                    "train_samples": len(data["train_texts"]),
                    "test_samples": len(data["test_texts"]),
                    "num_classes": data["num_classes"],
                    "epochs": args.epochs,
                    "probe_lr": args.lr,
                    "probe_batch_size": args.batch_size,
                    "probe_weight_decay": args.weight_decay,
                    "encode_batch_size": encode_batch_size,
                    "precision": resolved_precision,
                    "diagnostic_samples": args.diagnostic_samples,
                    "max_train_samples": args.max_train_samples,
                    "max_test_samples": args.max_test_samples,
                    **hparams,
                    **diagnostics,
                    **probe_result,
                }
                rows.append(row)

                print(
                    f"      best_test={row['best_test_accuracy']:.4f} | "
                    f"train_at_best={row['train_accuracy_at_best']:.4f} | "
                    f"final_test={row['final_test_accuracy']:.4f} | "
                    f"best_epoch={row['best_epoch']}"
                )

                del train_embs, test_embs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if args.write_every > 0 and len(rows) % args.write_every == 0:
                    paths = write_outputs(rows, args.output)
                    print(f"      Updated CSV: {paths['best_csv']}")

            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No probe results were produced.")

    paths = write_outputs(rows, args.output)
    best_rows = best_rows_by_folder(rows)
    print_task_leaders(best_rows, top_k=args.top_k)

    print("\nWrote:")
    print(f"  All results CSV: {paths['all_csv']}")
    print(f"  Best summary CSV: {paths['best_csv']}")
    print(f"  Markdown summary: {paths['summary']}")


if __name__ == "__main__":
    main()
