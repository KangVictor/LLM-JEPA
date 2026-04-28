"""Generate a mixed preprocessed paragraph dataset for SentenceJEPA.

The output format matches ``preprocess.py`` and can be used with:

    python train.py --config configs/default.yaml \
        --override data.preprocessed_path=data/mixed

Example:
    python generate_dataset.py \
        --config configs/colab.yaml \
        --output /content/drive/MyDrive/SentenceJEPA/mixed_owt_books \
        --total_paragraphs 1000000 \
        --mix wikipedia=0.5,openwebtext=0.3,bookcorpus=0.2

Supported sources:
    wikipedia    -> wikimedia/wikipedia 20231101.en, English articles.
    openwebtext  -> Skylion007/openwebtext, English web documents.
    bookcorpus   -> kd13/bookcorpus-clean, English ordered book sentences.

Excluded source:
    opensubtitles -> intentionally unsupported for this JEPA setup. Subtitle
    corpora are dialogue/translation fragments without stable paragraph
    boundaries, so mixing them into paragraph-level prediction would change
    the task semantics.
"""

import argparse
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass

torch = None
load_dataset = None
AutoTokenizer = None
split_sentences = None


def import_runtime_deps():
    """Import heavy training dependencies only when generation actually runs."""
    global torch, load_dataset, AutoTokenizer, split_sentences

    if torch is not None:
        return

    import torch as torch_module
    from datasets import load_dataset as load_dataset_fn
    from transformers import AutoTokenizer as auto_tokenizer_cls

    from src.data import split_sentences as split_sentences_fn

    torch = torch_module
    load_dataset = load_dataset_fn
    AutoTokenizer = auto_tokenizer_cls
    split_sentences = split_sentences_fn


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str | None
    split: str
    text_column: str
    mode: str
    description: str
    supported: bool = True
    config: str | None = None
    reason: str | None = None


SOURCE_SPECS = {
    "wikipedia": SourceSpec(
        name="wikipedia",
        dataset="wikimedia/wikipedia",
        split="train",
        text_column="text",
        mode="documents",
        description=(
            "English Wikipedia articles; paragraph blocks are split on blank "
            "lines, matching the existing preprocessing path."
        ),
        config="20231101.en",
    ),
    "openwebtext": SourceSpec(
        name="openwebtext",
        dataset="Skylion007/openwebtext",
        split="train",
        text_column="text",
        mode="documents",
        description="English web documents with text field; paragraph/sentence windows are extracted.",
        config="plain_text",
    ),
    "bookcorpus": SourceSpec(
        name="bookcorpus",
        dataset="kd13/bookcorpus-clean",
        split="train",
        text_column="text",
        mode="ordered_sentences",
        description=(
            "English BookCorpus-clean rows with doc_id/sent_id; contiguous "
            "sentences are grouped into pseudo-paragraphs."
        ),
    ),
    "opensubtitles": SourceSpec(
        name="opensubtitles",
        dataset=None,
        split="train",
        text_column="content",
        mode="excluded",
        description="Excluded from this paragraph-level generator.",
        supported=False,
        reason=(
            "OpenSubtitles is subtitle dialogue/translation data. The common HF "
            "versions are either sentence-aligned translation pairs or full SRT "
            "files with timing/dialogue fragments, not clean English paragraphs. "
            "That is not faithful to the current paragraph -> sentence JEPA task."
        ),
    ),
}


def parse_mix(mix):
    """Parse 'wikipedia=0.5,openwebtext=0.3,bookcorpus=0.2'."""
    proportions = {}
    for item in mix.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Mix item must be name=proportion, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip().lower()
        if name not in SOURCE_SPECS:
            choices = ", ".join(sorted(SOURCE_SPECS))
            raise ValueError(f"Unknown source '{name}'. Available: {choices}")

        spec = SOURCE_SPECS[name]
        if not spec.supported:
            raise ValueError(f"Source '{name}' is excluded: {spec.reason}")

        prop = float(value)
        if prop <= 0:
            raise ValueError(f"Proportion for '{name}' must be > 0, got {prop}")
        proportions[name] = proportions.get(name, 0.0) + prop

    if not proportions:
        raise ValueError(
            "No sources selected. Example: "
            "--mix wikipedia=0.5,openwebtext=0.3,bookcorpus=0.2"
        )

    total = sum(proportions.values())
    return {name: prop / total for name, prop in proportions.items()}


def compute_quotas(proportions, total_paragraphs):
    """Turn proportions into exact integer quotas summing to total_paragraphs."""
    names = list(proportions)
    raw = {name: proportions[name] * total_paragraphs for name in names}
    quotas = {name: int(raw[name]) for name in names}
    remainder = total_paragraphs - sum(quotas.values())

    names_by_fraction = sorted(
        names, key=lambda name: raw[name] - quotas[name], reverse=True
    )
    for name in names_by_fraction[:remainder]:
        quotas[name] += 1
    return quotas


def clean_sentence(sentence):
    sentence = re.sub(r"\s+", " ", str(sentence)).strip()
    return sentence


def is_usable_sentence(sentence, min_chars):
    if len(sentence) < min_chars:
        return False
    # Keep English-like text and drop rows that are mostly markup/numbers.
    alpha = sum(ch.isalpha() for ch in sentence)
    return alpha >= max(3, len(sentence) * 0.45)


def sentence_windows(sentences, min_sentences, max_sentences, stride=None):
    """Yield fixed-size sentence windows that satisfy min/max sentence counts."""
    if stride is None:
        stride = max_sentences
    if len(sentences) < min_sentences:
        return

    start = 0
    while start < len(sentences):
        window = sentences[start : start + max_sentences]
        if len(window) >= min_sentences:
            yield window
        start += stride


def extract_document_paragraphs(text, min_sentences, max_sentences, min_chars):
    """Extract paragraph-like sentence blocks from a document string."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return

    blocks = re.split(r"\n\s*\n+", text)
    if len(blocks) == 1:
        # Some HF text dumps flatten articles into one line. Treat the whole
        # document as one discourse stream and chunk it into sentence windows.
        blocks = [text]

    for block in blocks:
        block = clean_sentence(block)
        if len(block) < min_chars:
            continue
        sentences = [
            clean_sentence(sentence)
            for sentence in split_sentences(block)
        ]
        sentences = [
            sentence
            for sentence in sentences
            if is_usable_sentence(sentence, min_chars)
        ]
        yield from sentence_windows(sentences, min_sentences, max_sentences)


def iter_document_paragraphs(spec, min_sentences, max_sentences, min_chars):
    ds = load_dataset(
        spec.dataset,
        spec.config,
        split=spec.split,
        streaming=True,
        trust_remote_code=False,
    )
    for row in ds:
        text = row.get(spec.text_column, "")
        yield from extract_document_paragraphs(
            text, min_sentences, max_sentences, min_chars
        )


def iter_bookcorpus_paragraphs(spec, min_sentences, max_sentences, min_chars):
    ds = load_dataset(
        spec.dataset,
        spec.config,
        split=spec.split,
        streaming=True,
        trust_remote_code=False,
    )

    current_doc_id = None
    buffer = []

    def flush_buffer(force=False):
        nonlocal buffer
        if len(buffer) >= min_sentences and (force or len(buffer) >= max_sentences):
            window = buffer[:max_sentences]
            buffer = buffer[max_sentences:]
            return window
        return None

    for row in ds:
        doc_id = row.get("doc_id")
        if current_doc_id is None:
            current_doc_id = doc_id
        elif doc_id != current_doc_id:
            while len(buffer) >= min_sentences:
                window = buffer[:max_sentences]
                buffer = buffer[max_sentences:]
                if len(window) >= min_sentences:
                    yield window
            buffer = []
            current_doc_id = doc_id

        sentence = clean_sentence(row.get(spec.text_column, ""))
        if not is_usable_sentence(sentence, min_chars):
            continue
        buffer.append(sentence)

        window = flush_buffer()
        if window is not None:
            yield window

    while len(buffer) >= min_sentences:
        window = buffer[:max_sentences]
        buffer = buffer[max_sentences:]
        if len(window) >= min_sentences:
            yield window


def iter_source_paragraphs(source_name, min_sentences, max_sentences, min_chars):
    spec = SOURCE_SPECS[source_name]
    if spec.mode == "documents":
        return iter_document_paragraphs(
            spec, min_sentences, max_sentences, min_chars
        )
    if spec.mode == "ordered_sentences":
        return iter_bookcorpus_paragraphs(
            spec, min_sentences, max_sentences, min_chars
        )
    raise ValueError(f"Unsupported source mode for {source_name}: {spec.mode}")


def tokenize_paragraph_batch(tokenizer, paragraphs, max_tokens):
    """Tokenize a batch of sentence-list paragraphs and return train samples."""
    flat_sentences = []
    boundaries = []
    for sentences in paragraphs:
        start = len(flat_sentences)
        flat_sentences.extend(sentences)
        boundaries.append((start, len(flat_sentences)))

    encoded = tokenizer(
        flat_sentences,
        padding=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
    )

    samples = []
    for start, end in boundaries:
        ids = encoded["input_ids"][start:end]
        samples.append({
            "input_ids": ids,
            "num_sentences": len(ids),
        })
    return samples


def collect_source_samples(
    source_name,
    quota,
    cfg,
    tokenizer,
    tokenizer_batch_paragraphs,
    min_chars,
):
    data_cfg = cfg["data"]
    min_sentences = data_cfg["min_sentences"]
    max_sentences = data_cfg["max_sentences"]
    max_tokens = data_cfg["max_tokens_per_sentence"]

    samples = []
    sentence_counts = []
    paragraph_batch = []
    scanned_paragraphs = 0
    t0 = time.time()

    iterator = iter_source_paragraphs(
        source_name, min_sentences, max_sentences, min_chars
    )
    for sentences in iterator:
        scanned_paragraphs += 1
        paragraph_batch.append(sentences)

        if len(paragraph_batch) >= tokenizer_batch_paragraphs:
            new_samples = tokenize_paragraph_batch(
                tokenizer, paragraph_batch, max_tokens
            )
            samples.extend(new_samples)
            sentence_counts.extend(sample["num_sentences"] for sample in new_samples)
            paragraph_batch = []

            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"  {source_name}: {len(samples):,}/{quota:,} samples "
                f"({len(samples) / elapsed:.1f} samples/s)"
            )
            if len(samples) >= quota:
                break

    if paragraph_batch and len(samples) < quota:
        new_samples = tokenize_paragraph_batch(tokenizer, paragraph_batch, max_tokens)
        samples.extend(new_samples)
        sentence_counts.extend(sample["num_sentences"] for sample in new_samples)

    samples = samples[:quota]
    sentence_counts = sentence_counts[:quota]
    if len(samples) < quota:
        raise RuntimeError(
            f"Source '{source_name}' ended after {len(samples):,} usable samples, "
            f"but quota was {quota:,}."
        )

    return samples, {
        "requested": quota,
        "collected": len(samples),
        "scanned_usable_paragraphs": scanned_paragraphs,
        "sentence_counts": sentence_counts,
    }


def list_sources():
    print("Available dataset sources:")
    for name, spec in SOURCE_SPECS.items():
        status = "supported" if spec.supported else "excluded"
        print(f"  {name:14s} {status:10s} {spec.description}")
        if spec.reason:
            print(f"    reason: {spec.reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed SentenceJEPA preprocessed train/val data."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--total_paragraphs", type=int, required=False)
    parser.add_argument(
        "--mix",
        type=str,
        default="wikipedia=0.5,openwebtext=0.3,bookcorpus=0.2",
        help=(
            "Comma-separated proportions, e.g. "
            "wikipedia=0.5,openwebtext=0.3,bookcorpus=0.2"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_paragraphs", type=int, default=None)
    parser.add_argument("--tokenizer_batch_paragraphs", type=int, default=4096)
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list_sources", action="store_true")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if args.output is None:
        raise ValueError("--output is required unless --list_sources is used")
    if args.total_paragraphs is None or args.total_paragraphs <= 0:
        raise ValueError("--total_paragraphs must be a positive integer")
    if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {args.output}. "
            "Use --overwrite to replace train.pt/val.pt/meta.pt."
        )

    proportions = parse_mix(args.mix)

    import_runtime_deps()
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    val_size = (
        args.val_paragraphs
        if args.val_paragraphs is not None
        else cfg["training"].get("val_paragraphs", 1000)
    )
    if val_size >= args.total_paragraphs:
        raise ValueError(
            f"val_paragraphs={val_size} must be smaller than "
            f"total_paragraphs={args.total_paragraphs}"
        )

    quotas = compute_quotas(proportions, args.total_paragraphs)

    print("Dataset generation plan:")
    print(f"  total_paragraphs: {args.total_paragraphs:,}")
    print(f"  val_paragraphs:   {val_size:,}")
    for name, quota in quotas.items():
        spec = SOURCE_SPECS[name]
        print(
            f"  {name:14s} {quota:,} paragraphs "
            f"({proportions[name] * 100:.1f}%) from {spec.dataset}"
        )
    print("Excluded:")
    for name, spec in SOURCE_SPECS.items():
        if not spec.supported:
            print(f"  {name}: {spec.reason}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["data"]["tokenizer"], use_fast=True
    )

    all_samples = []
    source_meta = {}
    t0 = time.time()

    for name, quota in quotas.items():
        print(f"\nCollecting {name}...")
        samples, stats = collect_source_samples(
            source_name=name,
            quota=quota,
            cfg=cfg,
            tokenizer=tokenizer,
            tokenizer_batch_paragraphs=args.tokenizer_batch_paragraphs,
            min_chars=args.min_chars,
        )
        for sample in samples:
            sample["source"] = name
        all_samples.extend(samples)
        source_meta[name] = {
            k: v for k, v in stats.items() if k != "sentence_counts"
        }

    random.seed(args.seed)
    random.shuffle(all_samples)

    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    sentence_counts = torch.tensor(
        [sample["num_sentences"] for sample in all_samples], dtype=torch.float32
    )

    source_counts = defaultdict(int)
    for sample in all_samples:
        source_counts[sample["source"]] += 1

    meta = {
        "num_paragraphs": len(all_samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "source_counts": dict(source_counts),
        "source_meta": source_meta,
        "mix": proportions,
        "quotas": quotas,
        "excluded_sources": {
            name: spec.reason
            for name, spec in SOURCE_SPECS.items()
            if not spec.supported
        },
        "sentence_count_mean": sentence_counts.mean().item(),
        "sentence_count_min": sentence_counts.min().item(),
        "sentence_count_max": sentence_counts.max().item(),
        "tokenizer": cfg["data"]["tokenizer"],
        "max_tokens_per_sentence": cfg["data"]["max_tokens_per_sentence"],
        "min_sentences": cfg["data"]["min_sentences"],
        "max_sentences": cfg["data"]["max_sentences"],
    }

    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.pt")
    val_path = os.path.join(args.output, "val.pt")
    meta_path = os.path.join(args.output, "meta.pt")

    print("\nSaving...")
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    torch.save(meta, meta_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Saved mixed dataset to {args.output}/")
    print(f"{'=' * 60}")
    print(f"  train.pt: {len(train_samples):,} paragraphs")
    print(f"  val.pt:   {len(val_samples):,} paragraphs")
    print(f"  meta.pt:  dataset statistics")
    print(f"  elapsed:  {elapsed:.0f}s")
    print("  source counts:")
    for name, count in sorted(source_counts.items()):
        print(f"    {name}: {count:,}")
    print("\nTo train:")
    print(
        "  python train.py --config "
        f"{args.config} --override data.preprocessed_path={args.output}"
    )


if __name__ == "__main__":
    main()
