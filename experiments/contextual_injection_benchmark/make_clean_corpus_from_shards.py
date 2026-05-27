"""Decode preprocessed SentenceJEPA shards into clean paragraph JSONL.

The generated rows are compatible with build_benchmark.py:

    {"doc_id": "...", "text": "paragraph text here"}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INSTRUCTION_LIKE_PATTERNS = (
    r"\bignore (all )?(previous|prior|above)\b",
    r"\bsystem\s*:",
    r"\bassistant\s*:",
    r"\bdeveloper message\b",
    r"\bhidden system\b",
    r"\bprompt injection\b",
    r"\breveal hidden\b",
    r"\boverride (the )?(previous|prior|above|system)\b",
)


def load_tokenizer_name(config_path, tokenizer_override):
    if tokenizer_override:
        return tokenizer_override
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["tokenizer"]


def discover_shard_files(input_path, split, pattern):
    root = Path(input_path)
    if root.is_file():
        return [root]

    files = []
    if pattern:
        files.extend(sorted(root.glob(pattern)))
    else:
        if split in ("train", "both"):
            train_shards = root / "train_shards"
            if train_shards.exists():
                files.extend(sorted(train_shards.glob("train_*.pt")))
            train_pt = root / "train.pt"
            if train_pt.exists():
                files.append(train_pt)
        if split in ("val", "both"):
            val_pt = root / "val.pt"
            if val_pt.exists():
                files.append(val_pt)
        if split == "source_shards":
            source_root = root / "shards"
            search_root = source_root if source_root.exists() else root
            files.extend(sorted(search_root.glob("*/*.pt")))

    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No shard files found under {root}. Try --pattern, --split, or pass a .pt file directly."
        )
    return files


def read_samples(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError(f"Unsupported shard payload format: {path}")


def clean_decoded_sentence(sentence):
    sentence = " ".join(sentence.split())
    sentence = sentence.replace(" n't", "n't")
    sentence = sentence.replace(" 's", "'s")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" !", "!")
    sentence = sentence.replace(" ?", "?")
    sentence = sentence.replace(" ;", ";")
    sentence = sentence.replace(" :", ":")
    return sentence.strip()


def decode_sample(tokenizer, sample, max_sentences=None):
    input_ids = sample["input_ids"]
    if max_sentences is not None:
        input_ids = input_ids[:max_sentences]
    sentences = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    sentences = [clean_decoded_sentence(sentence) for sentence in sentences]
    return [sentence for sentence in sentences if len(sentence) >= 3]


def is_instruction_like(text):
    lower_text = text.lower()
    return any(re.search(pattern, lower_text) for pattern in INSTRUCTION_LIKE_PATTERNS)


def source_from_path(path):
    if path.parent.name == "train_shards":
        return "train"
    if path.parent.parent.name == "shards":
        return path.parent.name
    return path.stem


def sample_to_row(tokenizer, sample, source, shard_path, sample_index, args):
    sentences = decode_sample(
        tokenizer,
        sample,
        max_sentences=args.max_sentences,
    )
    if len(sentences) < args.min_sentences:
        return None

    text = " ".join(sentences)
    if args.reject_instruction_like and is_instruction_like(text):
        return None

    return {
        "doc_id": f"{source}:{shard_path.stem}:{sample_index}",
        "text": text,
        "sentences": sentences,
        "metadata": {
            "source": source,
            "shard_path": str(shard_path),
            "sample_index": sample_index,
            "num_sentences": len(sentences),
            "decoded_from_token_ids": True,
        },
    }


def write_rows(rows, output_path):
    with Path(output_path).open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(args):
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    tokenizer_name = load_tokenizer_name(args.config, args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    shard_files = discover_shard_files(args.input_path, args.split, args.pattern)
    if args.shuffle_files:
        rng.shuffle(shard_files)

    allowed_sources = None
    if args.sources:
        allowed_sources = {source.strip() for source in args.sources.split(",") if source.strip()}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    scanned = 0
    skipped = 0
    started = time.time()

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Input files: {len(shard_files):,}")
    print(f"Output: {output_path}")
    if args.max_examples is not None and args.sample_strategy == "reservoir":
        print(
            "Sampling strategy: reservoir; scanning every shard to draw an "
            "approximately uniform random clean corpus."
        )
    else:
        print(
            "Sampling strategy: first; use --sample_strategy reservoir with "
            "--max_examples for an unbiased random subset."
        )

    if args.max_examples is not None and args.sample_strategy == "reservoir":
        reservoir = []
        eligible = 0
        for file_index, shard_path in enumerate(shard_files, start=1):
            samples = read_samples(shard_path)
            fallback_source = source_from_path(shard_path)
            print(
                f"[{file_index:,}/{len(shard_files):,}] {shard_path} "
                f"({len(samples):,} samples)"
            )

            for sample_index, sample in enumerate(samples):
                scanned += 1
                source = sample.get("source", fallback_source)
                if allowed_sources is not None and source not in allowed_sources:
                    skipped += 1
                    continue

                row = sample_to_row(
                    tokenizer,
                    sample,
                    source,
                    shard_path,
                    sample_index,
                    args,
                )
                if row is None:
                    skipped += 1
                    continue

                eligible += 1
                if len(reservoir) < args.max_examples:
                    reservoir.append(row)
                else:
                    replace_index = rng.randint(0, eligible - 1)
                    if replace_index < args.max_examples:
                        reservoir[replace_index] = row

                if eligible % args.log_every == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"  eligible {eligible:,} | reservoir {len(reservoir):,} | "
                        f"scanned {scanned:,} | skipped {skipped:,} | "
                        f"{eligible / elapsed:.1f} eligible rows/s"
                    )

        rng.shuffle(reservoir)
        write_rows(reservoir, output_path)
        written = len(reservoir)
        print(f"Random sample written to {output_path}")
        print_summary(written, scanned, skipped, started)
        return

    with output_path.open("w", encoding="utf-8") as out:
        for file_index, shard_path in enumerate(shard_files, start=1):
            samples = read_samples(shard_path)
            if args.shuffle_within_shard:
                samples = list(samples)
                rng.shuffle(samples)
            fallback_source = source_from_path(shard_path)
            print(
                f"[{file_index:,}/{len(shard_files):,}] {shard_path} "
                f"({len(samples):,} samples)"
            )

            for sample_index, sample in enumerate(samples):
                scanned += 1
                source = sample.get("source", fallback_source)
                if allowed_sources is not None and source not in allowed_sources:
                    skipped += 1
                    continue

                row = sample_to_row(
                    tokenizer,
                    sample,
                    source,
                    shard_path,
                    sample_index,
                    args,
                )
                if row is None:
                    skipped += 1
                    continue

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                if written % args.log_every == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"  wrote {written:,} rows | scanned {scanned:,} | "
                        f"skipped {skipped:,} | {written / elapsed:.1f} rows/s"
                    )

                if args.max_examples is not None and written >= args.max_examples:
                    print(f"Reached --max_examples={args.max_examples:,}")
                    print_summary(written, scanned, skipped, started)
                    return

    print_summary(written, scanned, skipped, started)


def print_summary(written, scanned, skipped, started):
    elapsed = max(time.time() - started, 1e-9)
    print("\nDone.")
    print(f"  written: {written:,}")
    print(f"  scanned: {scanned:,}")
    print(f"  skipped: {skipped:,}")
    print(f"  elapsed: {elapsed / 60:.1f} min")
    print(f"  speed:   {written / elapsed:.1f} rows/s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode SentenceJEPA .pt shards into clean paragraph JSONL."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Preprocessed dataset root, shard directory, or one .pt shard file.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer override. Defaults to data.tokenizer from --config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "both", "source_shards"),
        help=(
            "Which default files to read. train reads train_shards/train_*.pt or train.pt; "
            "source_shards reads shards/*/*.pt."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional glob pattern relative to --input_path, e.g. 'train_shards/train_*.pt'.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--min_sentences", type=int, default=4)
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=11,
        help="Decode at most this many clean sentences per row. Default 11 leaves room for one injection under a 12-sentence model cap.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Optional comma-separated source filter, e.g. 'wikipedia,c4,openwebtext'.",
    )
    parser.add_argument(
        "--reject_instruction_like",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter clean rows that already contain obvious instruction/prompt-injection phrases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="reservoir",
        choices=("first", "reservoir"),
        help=(
            "With --max_examples, reservoir scans all shards and writes a random "
            "subset. first stops after the first eligible examples."
        ),
    )
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--shuffle_within_shard", action="store_true")
    parser.add_argument("--log_every", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
