"""Evaluate SentenceJEPA leave-one-sentence-out injection localization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import SentenceJEPA  # noqa: E402
from utils import (  # noqa: E402
    auroc,
    average_precision,
    ensure_dir,
    json_default,
    read_jsonl,
    top_indices,
    zscores,
)


METHODS = ("jepa", "neighbor", "centroid")


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(device, precision):
    if precision == "fp32" or device.type not in ("cuda", "cpu"):
        return False, torch.float32, "fp32"
    if precision == "bf16":
        return True, torch.bfloat16, "bf16"
    if precision == "fp16":
        return True, torch.float16, "fp16"
    if precision != "auto":
        raise ValueError(f"Unknown precision: {precision}")

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, "bf16"
        return True, torch.float16, "fp16"
    return False, torch.float32, "fp32"


def autocast_context(device, enabled, dtype):
    if enabled:
        return torch.amp.autocast(device.type, dtype=dtype)
    return nullcontext()


def load_model_and_tokenizer(args, device):
    fallback_cfg = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("config")
    cfg = (
        fallback_cfg
        if args.no_checkpoint_config or not isinstance(checkpoint_cfg, dict)
        else checkpoint_cfg
    )
    config_source = "cli_config" if cfg is fallback_cfg else "checkpoint"

    model = SentenceJEPA(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"], use_fast=True)
    step = checkpoint.get("step", "?")
    print(
        f"Loaded SentenceJEPA checkpoint {args.checkpoint} "
        f"(step {step}, config={config_source})"
    )
    return model, tokenizer, cfg


def encode_sentences(tokenizer, sentences, cfg, device):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=cfg["data"]["max_tokens_per_sentence"],
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
    sentence_mask = torch.ones(
        (1, len(sentences)), dtype=torch.bool, device=device
    )
    return input_ids, attention_mask, sentence_mask


def cosine_error(pred, target):
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)


def baseline_neighbor_scores(targets):
    scores = []
    for idx in range(targets.size(0)):
        neighbors = []
        if idx > 0:
            neighbors.append(targets[idx - 1])
        if idx + 1 < targets.size(0):
            neighbors.append(targets[idx + 1])
        if not neighbors:
            scores.append(0.0)
            continue
        context = torch.stack(neighbors, dim=0).mean(dim=0)
        score = cosine_error(targets[idx].unsqueeze(0), context.unsqueeze(0))[0]
        scores.append(float(score.item()))
    return scores


def baseline_centroid_scores(targets):
    scores = []
    n_sentences = targets.size(0)
    if n_sentences <= 1:
        return [0.0 for _ in range(n_sentences)]

    total = targets.sum(dim=0)
    for idx in range(n_sentences):
        context = (total - targets[idx]) / (n_sentences - 1)
        score = cosine_error(targets[idx].unsqueeze(0), context.unsqueeze(0))[0]
        scores.append(float(score.item()))
    return scores


@torch.no_grad()
def score_example(model, tokenizer, cfg, example, device, amp_enabled, amp_dtype):
    sentences = example["sentences"]
    max_sentences = cfg["data"]["max_sentences"]
    if len(sentences) > max_sentences:
        raise ValueError(
            f"Example {example.get('example_id')} has {len(sentences)} sentences, "
            f"but model config allows {max_sentences}."
        )

    input_ids, attention_mask, sentence_mask = encode_sentences(
        tokenizer, sentences, cfg, device
    )
    n_sentences = len(sentences)

    with autocast_context(device, amp_enabled, amp_dtype):
        enc_out = model.encoder(input_ids, attention_mask)
        encoder_sequence = enc_out.repeat(n_sentences, 1, 1)
        sentence_mask_repeated = sentence_mask.repeat(n_sentences, 1)
        mask_indices = torch.eye(n_sentences, dtype=torch.bool, device=device)
        pred_out = model.predictor(
            encoder_sequence, sentence_mask_repeated, mask_indices
        )

    targets = enc_out.squeeze(0)
    jepa_scores = cosine_error(pred_out, targets).detach().cpu().tolist()
    targets = targets.float().detach()
    neighbor_scores = baseline_neighbor_scores(targets)
    centroid_scores = baseline_centroid_scores(targets)

    return {
        "jepa": jepa_scores,
        "neighbor": neighbor_scores,
        "centroid": centroid_scores,
    }


def empty_metrics():
    return {
        "examples": 0,
        "sentences": 0,
        "injected_examples": 0,
        "clean_examples": 0,
        "top1_localization_accuracy": None,
        "recall_at_2": None,
        "auroc_sentence": None,
        "auprc_sentence": None,
        "clean_sentence_fpr_z_gt_2": None,
        "clean_sentence_fpr_z_gt_3": None,
        "clean_paragraph_fpr_any_z_gt_2": None,
        "clean_paragraph_fpr_any_z_gt_3": None,
    }


def compute_metrics(examples, sentence_rows):
    metrics = {}
    for method in METHODS:
        method_metrics = empty_metrics()
        method_metrics["examples"] = len(examples)
        method_metrics["sentences"] = len(sentence_rows)

        injected_total = 0
        top1_hits = 0
        recall2_hits = 0
        clean_examples = 0
        clean_sentence_total = 0
        clean_z2_hits = 0
        clean_z3_hits = 0
        clean_para_z2_hits = 0
        clean_para_z3_hits = 0

        all_scores = []
        all_labels = []

        for example in examples:
            labels = example["labels"]
            scores = example["scores"][method]
            zs = example["zscores"][method]
            injected_indices = set(example["injected_indices"])

            all_scores.extend(scores)
            all_labels.extend(labels)

            if injected_indices:
                injected_total += 1
                ranked = top_indices(scores, min(2, len(scores)))
                if ranked and ranked[0] in injected_indices:
                    top1_hits += 1
                if any(idx in injected_indices for idx in ranked):
                    recall2_hits += 1
            else:
                clean_examples += 1
                clean_sentence_total += len(zs)
                clean_z2_hits += sum(1 for z in zs if z > 2.0)
                clean_z3_hits += sum(1 for z in zs if z > 3.0)
                clean_para_z2_hits += int(any(z > 2.0 for z in zs))
                clean_para_z3_hits += int(any(z > 3.0 for z in zs))

        method_metrics["injected_examples"] = injected_total
        method_metrics["clean_examples"] = clean_examples
        if injected_total:
            method_metrics["top1_localization_accuracy"] = top1_hits / injected_total
            method_metrics["recall_at_2"] = recall2_hits / injected_total
        method_metrics["auroc_sentence"] = auroc(all_scores, all_labels)
        method_metrics["auprc_sentence"] = average_precision(all_scores, all_labels)
        if clean_sentence_total:
            method_metrics["clean_sentence_fpr_z_gt_2"] = (
                clean_z2_hits / clean_sentence_total
            )
            method_metrics["clean_sentence_fpr_z_gt_3"] = (
                clean_z3_hits / clean_sentence_total
            )
        if clean_examples:
            method_metrics["clean_paragraph_fpr_any_z_gt_2"] = (
                clean_para_z2_hits / clean_examples
            )
            method_metrics["clean_paragraph_fpr_any_z_gt_3"] = (
                clean_para_z3_hits / clean_examples
            )
        metrics[method] = method_metrics

    return metrics


def write_scores_csv(sentence_rows, path):
    fieldnames = [
        "example_id",
        "doc_id",
        "attack_type",
        "sentence_index",
        "label",
        "sentence",
    ]
    for method in METHODS:
        fieldnames.extend([f"{method}_score", f"{method}_z"])

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sentence_rows:
            writer.writerow(row)


def write_suspicious_markdown(examples, path, top_k_examples=10):
    ranked_examples = sorted(
        examples,
        key=lambda example: max(example["zscores"]["jepa"] or [0.0]),
        reverse=True,
    )[:top_k_examples]

    lines = [
        "# Contextual Injection Suspicious Sentences",
        "",
        "Ranked by the maximum paragraph-normalized JEPA prediction-error z-score.",
        "",
    ]
    for example in ranked_examples:
        lines.append(f"## {example['example_id']}")
        lines.append("")
        lines.append(
            f"- attack_type: `{example['attack_type']}`"
        )
        lines.append(
            f"- injected_indices: `{example['injected_indices']}`"
        )
        lines.append("")
        ranked = top_indices(example["scores"]["jepa"], min(3, len(example["sentences"])))
        for rank, idx in enumerate(ranked, start=1):
            label = example["labels"][idx]
            score = example["scores"]["jepa"][idx]
            z = example["zscores"]["jepa"][idx]
            sentence = example["sentences"][idx]
            lines.append(
                f"{rank}. idx={idx}, label={label}, score={score:.4f}, z={z:.2f}: "
                f"{sentence}"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def evaluate(args):
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    amp_enabled, amp_dtype, precision_name = resolve_precision(device, args.precision)
    print(f"Using device={device}, precision={precision_name}")

    model, tokenizer, cfg = load_model_and_tokenizer(args, device)
    raw_examples = list(read_jsonl(args.benchmark_path))
    if args.max_examples is not None:
        raw_examples = raw_examples[: args.max_examples]

    scored_examples = []
    sentence_rows = []
    for example_index, example in enumerate(raw_examples, start=1):
        if example_index % args.log_every == 0 or example_index == 1:
            print(f"Scoring example {example_index:,}/{len(raw_examples):,}")

        scores = score_example(
            model,
            tokenizer,
            cfg,
            example,
            device,
            amp_enabled,
            amp_dtype,
        )
        method_zscores = {method: zscores(scores[method]) for method in METHODS}
        scored_example = {
            "example_id": example["example_id"],
            "doc_id": example.get("doc_id"),
            "attack_type": example.get("attack_type", "unknown"),
            "sentences": example["sentences"],
            "labels": example["labels"],
            "injected_indices": example.get("injected_indices", []),
            "scores": scores,
            "zscores": method_zscores,
        }
        scored_examples.append(scored_example)

        for idx, sentence in enumerate(example["sentences"]):
            row = {
                "example_id": example["example_id"],
                "doc_id": example.get("doc_id"),
                "attack_type": example.get("attack_type", "unknown"),
                "sentence_index": idx,
                "label": example["labels"][idx],
                "sentence": sentence,
            }
            for method in METHODS:
                row[f"{method}_score"] = scores[method][idx]
                row[f"{method}_z"] = method_zscores[method][idx]
            sentence_rows.append(row)

    metrics = compute_metrics(scored_examples, sentence_rows)

    scores_path = output_dir / "scores_per_sentence.csv"
    metrics_path = output_dir / "metrics.json"
    markdown_path = output_dir / "top_suspicious_sentences.md"

    write_scores_csv(sentence_rows, scores_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_suspicious_markdown(
        scored_examples,
        markdown_path,
        top_k_examples=args.top_k_report,
    )

    print(f"Wrote sentence scores to {scores_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote suspicious sentence report to {markdown_path}")
    print("\nJEPA metrics:")
    print(json.dumps(metrics["jepa"], indent=2, default=json_default))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate contextual injection localization with SentenceJEPA."
    )
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=("auto", "bf16", "fp16", "fp32"),
    )
    parser.add_argument(
        "--no_checkpoint_config",
        action="store_true",
        help="Use --config instead of the config saved inside the checkpoint.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--top_k_report", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
