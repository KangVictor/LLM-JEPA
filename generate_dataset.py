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
    c4           -> allenai/c4 en, cleaned English Common Crawl documents.

Excluded source:
    opensubtitles -> intentionally unsupported for this JEPA setup. Subtitle
    corpora are dialogue/translation fragments without stable paragraph
    boundaries, so mixing them into paragraph-level prediction would change
    the task semantics.
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
    "c4": SourceSpec(
        name="c4",
        dataset="allenai/c4",
        split="train",
        text_column="text",
        mode="documents",
        description=(
            "C4 English cleaned Common Crawl documents; sentence windows are "
            "extracted from each text field."
        ),
        config="en",
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


def source_shard_dir(output, source_name):
    return Path(output) / "shards" / source_name


def shard_files(output, source_name=None):
    root = Path(output) / "shards"
    if source_name is not None:
        root = root / source_name
    if not root.exists():
        return []
    return sorted(root.rglob("shard_*.pt"))


def read_shard(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return payload, {"path": str(path), "num_samples": len(payload)}
    return payload["samples"], payload.get("meta", {})


def count_existing_samples(output, source_name):
    total = 0
    paths = shard_files(output, source_name)
    for path in paths:
        samples, _ = read_shard(path)
        total += len(samples)
    return total, paths


def save_shard(output, source_name, shard_index, samples, meta):
    out_dir = source_shard_dir(output, source_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_index:06d}.pt"
    tmp_path = out_dir / f"shard_{shard_index:06d}.tmp"
    torch.save({"samples": samples, "meta": meta}, tmp_path)
    os.replace(tmp_path, path)
    return path


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def generate_source_shards(
    source_name,
    quota,
    cfg,
    tokenizer,
    output,
    tokenizer_batch_paragraphs,
    min_chars,
    shard_size,
    resume,
):
    data_cfg = cfg["data"]
    min_sentences = data_cfg["min_sentences"]
    max_sentences = data_cfg["max_sentences"]
    max_tokens = data_cfg["max_tokens_per_sentence"]

    existing_count, existing_paths = count_existing_samples(output, source_name)
    if existing_count > quota:
        print(
            f"  {source_name}: found {existing_count:,} existing samples, "
            f"which already exceeds quota {quota:,}."
        )
        return {
            "requested": quota,
            "existing": existing_count,
            "collected": 0,
            "total": existing_count,
            "shards": [str(path) for path in existing_paths],
        }
    if existing_count and not resume:
        raise FileExistsError(
            f"Found {existing_count:,} existing {source_name} shard samples in "
            f"{source_shard_dir(output, source_name)}. Use --resume or choose a new output."
        )

    print(
        f"  {source_name}: existing={existing_count:,}, "
        f"remaining={quota - existing_count:,}"
    )

    iterator = iter_source_paragraphs(
        source_name, min_sentences, max_sentences, min_chars
    )
    skipped = 0
    paragraph_batch = []
    shard_samples = []
    sentence_counts = []
    collected = 0
    scanned = 0
    t0 = time.time()
    shard_index = len(existing_paths)

    def flush_token_batch():
        nonlocal paragraph_batch, shard_samples, collected
        if not paragraph_batch:
            return
        new_samples = tokenize_paragraph_batch(tokenizer, paragraph_batch, max_tokens)
        paragraph_batch = []
        for sample in new_samples:
            if existing_count + collected >= quota:
                break
            sample["source"] = source_name
            shard_samples.append(sample)
            sentence_counts.append(sample["num_sentences"])
            collected += 1

    def flush_shard(force=False):
        nonlocal shard_samples, shard_index
        if not shard_samples:
            return None
        if not force and len(shard_samples) < shard_size:
            return None
        path = save_shard(
            output,
            source_name,
            shard_index,
            shard_samples,
            {
                "source": source_name,
                "shard_index": shard_index,
                "num_samples": len(shard_samples),
                "created_at": time.time(),
                "min_sentences": min_sentences,
                "max_sentences": max_sentences,
                "max_tokens_per_sentence": max_tokens,
            },
        )
        print(f"  saved {path} ({len(shard_samples):,} samples)")
        shard_index += 1
        shard_samples = []
        return path

    for sentences in iterator:
        scanned += 1
        if skipped < existing_count:
            skipped += 1
            continue

        paragraph_batch.append(sentences)
        if len(paragraph_batch) >= tokenizer_batch_paragraphs:
            flush_token_batch()
            flush_shard()

            elapsed = max(time.time() - t0, 1e-6)
            total_now = existing_count + collected
            print(
                f"  {source_name}: {total_now:,}/{quota:,} samples "
                f"({collected / elapsed:.1f} new samples/s)"
            )
            if total_now >= quota:
                break

    if existing_count + collected < quota:
        flush_token_batch()
    flush_shard(force=True)

    total = existing_count + collected
    if total < quota:
        raise RuntimeError(
            f"Source '{source_name}' ended at {total:,} samples, "
            f"but quota was {quota:,}."
        )

    meta = {
        "source": source_name,
        "requested": quota,
        "existing": existing_count,
        "collected": collected,
        "total": total,
        "skipped_for_resume": skipped,
        "scanned_usable_paragraphs": scanned,
        "sentence_count_mean": (
            sum(sentence_counts) / len(sentence_counts) if sentence_counts else None
        ),
        "shard_size": shard_size,
    }
    write_json(source_shard_dir(output, source_name) / "source_meta.json", meta)
    return meta


def load_samples_from_shards(output, quotas=None, log_every=10):
    samples_by_source = defaultdict(list)
    all_paths = shard_files(output)
    if not all_paths:
        raise FileNotFoundError(f"No shard_*.pt files found under {Path(output) / 'shards'}")

    print(f"  Found {len(all_paths):,} shard files under {Path(output) / 'shards'}")
    by_source = defaultdict(int)
    for path in all_paths:
        by_source[path.parent.name] += 1
    for source, count in sorted(by_source.items()):
        target = quotas.get(source) if quotas is not None else None
        target_text = f", quota={target:,}" if target is not None else ""
        print(f"    {source}: {count:,} shards{target_text}")

    t0 = time.time()
    for index, path in enumerate(all_paths, start=1):
        samples, _ = read_shard(path)
        kept = 0
        skipped = 0
        for sample in samples:
            source = sample.get("source", path.parent.name)
            if quotas is not None and source not in quotas:
                skipped += 1
                continue
            if quotas is not None and source in quotas:
                if len(samples_by_source[source]) >= quotas[source]:
                    skipped += 1
                    continue
            samples_by_source[source].append(sample)
            kept += 1

        if index == 1 or index == len(all_paths) or index % log_every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            loaded = sum(len(items) for items in samples_by_source.values())
            print(
                f"  loaded shard {index:,}/{len(all_paths):,}: {path.name} "
                f"kept={kept:,} skipped={skipped:,} total_loaded={loaded:,} "
                f"({loaded / elapsed:.1f} samples/s)"
            )
            for source in sorted(samples_by_source):
                if quotas is not None and source in quotas:
                    print(
                        f"    {source}: {len(samples_by_source[source]):,}/"
                        f"{quotas[source]:,}"
                    )
                else:
                    print(f"    {source}: {len(samples_by_source[source]):,}")

        if quotas is not None and all(
            len(samples_by_source[source]) >= quotas[source]
            for source in quotas
        ):
            print("  Reached all requested source quotas; stopping shard load.")
            break

    if quotas is not None:
        missing = {
            source: quotas[source] - len(samples_by_source[source])
            for source in quotas
            if len(samples_by_source[source]) < quotas[source]
        }
        if missing:
            raise RuntimeError(f"Not enough shard samples for requested mix: {missing}")

    all_samples = []
    for source, samples in samples_by_source.items():
        if quotas is not None and source in quotas:
            samples = samples[: quotas[source]]
        all_samples.extend(samples)
    print(f"  Total loaded samples for combine: {len(all_samples):,}")
    return all_samples


def iter_samples_from_shards(output, quotas=None, log_every=10):
    """Yield samples from saved source shards without retaining them all."""
    all_paths = shard_files(output)
    if not all_paths:
        raise FileNotFoundError(f"No shard_*.pt files found under {Path(output) / 'shards'}")

    print(f"  Found {len(all_paths):,} shard files under {Path(output) / 'shards'}")
    by_source = defaultdict(int)
    for path in all_paths:
        by_source[path.parent.name] += 1
    for source, count in sorted(by_source.items()):
        target = quotas.get(source) if quotas is not None else None
        target_text = f", quota={target:,}" if target is not None else ""
        print(f"    {source}: {count:,} shards{target_text}")

    counts = defaultdict(int)
    total_yielded = 0
    t0 = time.time()
    for index, path in enumerate(all_paths, start=1):
        samples, _ = read_shard(path)
        kept = 0
        skipped = 0
        for sample in samples:
            source = sample.get("source", path.parent.name)
            if quotas is not None and source not in quotas:
                skipped += 1
                continue
            if quotas is not None and counts[source] >= quotas[source]:
                skipped += 1
                continue
            counts[source] += 1
            total_yielded += 1
            kept += 1
            yield total_yielded - 1, sample

        if index == 1 or index == len(all_paths) or index % log_every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"  streamed shard {index:,}/{len(all_paths):,}: {path.name} "
                f"kept={kept:,} skipped={skipped:,} total={total_yielded:,} "
                f"({total_yielded / elapsed:.1f} samples/s)"
            )
            for source in sorted(counts):
                if quotas is not None and source in quotas:
                    print(f"    {source}: {counts[source]:,}/{quotas[source]:,}")
                else:
                    print(f"    {source}: {counts[source]:,}")

        if quotas is not None and all(counts[source] >= quotas[source] for source in quotas):
            print("  Reached all requested source quotas; stopping shard stream.")
            break

    if quotas is not None:
        missing = {
            source: quotas[source] - counts[source]
            for source in quotas
            if counts[source] < quotas[source]
        }
        if missing:
            raise RuntimeError(f"Not enough shard samples for requested mix: {missing}")


def save_final_dataset(samples, output, val_size, seed, meta_extra=None):
    print(f"  Shuffling {len(samples):,} samples with seed={seed}...")
    random.seed(seed)
    random.shuffle(samples)

    print(f"  Splitting val={val_size:,}, train={len(samples) - val_size:,}...")
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    print("  Computing final metadata...")
    sentence_counts = torch.tensor(
        [sample["num_sentences"] for sample in samples], dtype=torch.float32
    )
    source_counts = defaultdict(int)
    for sample in samples:
        source_counts[sample.get("source", "unknown")] += 1

    meta = {
        "num_paragraphs": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "source_counts": dict(source_counts),
        "sentence_count_mean": sentence_counts.mean().item(),
        "sentence_count_min": sentence_counts.min().item(),
        "sentence_count_max": sentence_counts.max().item(),
    }
    if meta_extra:
        meta.update(meta_extra)

    os.makedirs(output, exist_ok=True)
    train_path = os.path.join(output, "train.pt")
    val_path = os.path.join(output, "val.pt")
    meta_path = os.path.join(output, "meta.pt")

    print("\nSaving final dataset...")
    print(f"  writing {train_path} ...")
    torch.save(train_samples, train_path)
    print(f"  writing {val_path} ...")
    torch.save(val_samples, val_path)
    print(f"  writing {meta_path} ...")
    torch.save(meta, meta_path)

    print(f"  train.pt: {len(train_samples):,} paragraphs")
    print(f"  val.pt:   {len(val_samples):,} paragraphs")
    print(f"  meta.pt:  dataset statistics")
    print("  source counts:")
    for name, count in sorted(source_counts.items()):
        print(f"    {name}: {count:,}")
    return train_path, val_path, meta_path


def save_sharded_final_dataset(
    output,
    quotas,
    val_size,
    seed,
    train_shard_size,
    log_every=10,
    meta_extra=None,
):
    """Build val.pt and train_shards/*.pt without loading all train samples."""
    if quotas is None:
        raise ValueError("Sharded combine requires --total_paragraphs and --mix quotas.")
    total_samples = sum(quotas.values())
    if val_size >= total_samples:
        raise ValueError(
            f"val_paragraphs={val_size} must be smaller than sample count {total_samples:,}"
        )

    rng = random.Random(seed)
    val_indices = set(rng.sample(range(total_samples), val_size))
    train_dir = Path(output) / "train_shards"
    if train_dir.exists() and any(train_dir.glob("train_*.pt")):
        raise FileExistsError(
            f"Train shard directory already contains train_*.pt files: {train_dir}. "
            "Choose a new output directory or remove old final train shards."
        )
    train_dir.mkdir(parents=True, exist_ok=True)

    val_samples = []
    train_buffer = []
    train_paths = []
    source_counts = defaultdict(int)
    sentence_count_sum = 0
    sentence_count_min = None
    sentence_count_max = None
    num_train = 0
    num_val = 0

    def update_meta(sample):
        nonlocal sentence_count_sum, sentence_count_min, sentence_count_max
        source_counts[sample.get("source", "unknown")] += 1
        count = int(sample["num_sentences"])
        sentence_count_sum += count
        sentence_count_min = count if sentence_count_min is None else min(sentence_count_min, count)
        sentence_count_max = count if sentence_count_max is None else max(sentence_count_max, count)

    def flush_train_shard(force=False):
        nonlocal train_buffer
        if not train_buffer:
            return
        if not force and len(train_buffer) < train_shard_size:
            return
        shard_index = len(train_paths)
        path = train_dir / f"train_{shard_index:06d}.pt"
        tmp_path = train_dir / f"train_{shard_index:06d}.tmp"
        torch.save(train_buffer, tmp_path)
        os.replace(tmp_path, path)
        train_paths.append(str(path))
        print(f"  wrote {path} ({len(train_buffer):,} samples)")
        train_buffer = []

    print(f"  Selecting exactly {val_size:,} validation samples by seeded global index.")
    print(f"  Writing train shards of {train_shard_size:,} samples to {train_dir}")
    t0 = time.time()
    for global_index, sample in iter_samples_from_shards(output, quotas, log_every=log_every):
        update_meta(sample)
        if global_index in val_indices:
            val_samples.append(sample)
            num_val += 1
        else:
            train_buffer.append(sample)
            num_train += 1
            flush_train_shard()

    flush_train_shard(force=True)
    if num_val != val_size:
        raise RuntimeError(f"Expected {val_size:,} val samples, got {num_val:,}")

    meta = {
        "format": "sharded_train",
        "num_paragraphs": num_train + num_val,
        "num_train": num_train,
        "num_val": num_val,
        "train_shards": train_paths,
        "train_shard_size": train_shard_size,
        "source_counts": dict(source_counts),
        "sentence_count_mean": sentence_count_sum / max(1, num_train + num_val),
        "sentence_count_min": sentence_count_min,
        "sentence_count_max": sentence_count_max,
    }
    if meta_extra:
        meta.update(meta_extra)

    val_path = os.path.join(output, "val.pt")
    meta_path = os.path.join(output, "meta.pt")
    print("\nSaving sharded final dataset...")
    print(f"  writing {val_path} ({len(val_samples):,} samples) ...")
    torch.save(val_samples, val_path)
    print(f"  writing {meta_path} ...")
    torch.save(meta, meta_path)

    elapsed = time.time() - t0
    print("\nSharded final dataset saved.")
    print(f"  Train samples: {num_train:,} in {len(train_paths):,} shards")
    print(f"  Val samples:   {num_val:,}")
    print(f"  Elapsed:       {elapsed:.0f}s")


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
        "--mode",
        choices=("full", "generate_shards", "combine_shards"),
        default="full",
        help=(
            "full writes train.pt/val.pt directly; generate_shards writes "
            "resumable source shards; combine_shards builds train.pt/val.pt "
            "from saved shards."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="With --mode generate_shards, process only one source.",
    )
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
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument(
        "--combine_log_every",
        type=int,
        default=10,
        help="Print combine progress every N shard files.",
    )
    parser.add_argument(
        "--combine_format",
        choices=("monolithic", "sharded"),
        default="monolithic",
        help=(
            "monolithic writes train.pt; sharded writes train_shards/*.pt and "
            "keeps combine memory much lower."
        ),
    )
    parser.add_argument(
        "--final_train_shard_size",
        type=int,
        default=100000,
        help="Number of train samples per final train shard when --combine_format=sharded.",
    )
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list_sources", action="store_true")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if args.output is None:
        raise ValueError("--output is required unless --list_sources is used")
    if args.mode != "combine_shards" and (
        args.total_paragraphs is None or args.total_paragraphs <= 0
    ):
        raise ValueError("--total_paragraphs must be a positive integer")
    if (
        args.mode == "full"
        and os.path.exists(args.output)
        and os.listdir(args.output)
        and not args.overwrite
    ):
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
    if args.total_paragraphs is not None and val_size >= args.total_paragraphs:
        raise ValueError(
            f"val_paragraphs={val_size} must be smaller than "
            f"total_paragraphs={args.total_paragraphs}"
        )

    quotas = (
        compute_quotas(proportions, args.total_paragraphs)
        if args.total_paragraphs is not None
        else None
    )

    if args.source is not None:
        args.source = args.source.lower()
        if args.source not in SOURCE_SPECS:
            choices = ", ".join(sorted(SOURCE_SPECS))
            raise ValueError(f"Unknown source '{args.source}'. Available: {choices}")
        if not SOURCE_SPECS[args.source].supported:
            raise ValueError(
                f"Source '{args.source}' is excluded: {SOURCE_SPECS[args.source].reason}"
            )
        if args.mode != "generate_shards":
            raise ValueError("--source is only valid with --mode generate_shards")
        if args.source not in quotas:
            raise ValueError(
                f"--source {args.source} is not present in --mix {args.mix}. "
                f"Use --mix {args.source}=1 to generate only that source."
            )

    if args.mode == "combine_shards":
        print("Combining existing shards...")
        if quotas is None:
            print("  Using all available shard samples.")
        else:
            print("  Requested mix quotas:")
            for name, quota in quotas.items():
                print(f"    {name}: {quota:,}")
        meta_extra = {
            "mix": proportions if quotas is not None else None,
            "quotas": quotas,
            "excluded_sources": {
                name: spec.reason
                for name, spec in SOURCE_SPECS.items()
                if not spec.supported
            },
            "tokenizer": cfg["data"]["tokenizer"],
            "max_tokens_per_sentence": cfg["data"]["max_tokens_per_sentence"],
            "min_sentences": cfg["data"]["min_sentences"],
            "max_sentences": cfg["data"]["max_sentences"],
        }
        if args.combine_format == "sharded":
            save_sharded_final_dataset(
                args.output,
                quotas,
                val_size,
                args.seed,
                args.final_train_shard_size,
                log_every=max(1, args.combine_log_every),
                meta_extra=meta_extra,
            )
            return
        all_samples = load_samples_from_shards(
            args.output,
            quotas,
            log_every=max(1, args.combine_log_every),
        )
        if val_size >= len(all_samples):
            raise ValueError(
                f"val_paragraphs={val_size} must be smaller than available "
                f"sample count {len(all_samples):,}"
            )
        save_final_dataset(
            all_samples,
            args.output,
            val_size,
            args.seed,
            meta_extra,
        )
        return

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

    if args.mode == "generate_shards":
        selected_sources = [args.source] if args.source else list(quotas)
        print("\nGenerating resumable shards only.")
        print(f"  shard_size: {args.shard_size:,}")
        print(f"  resume:     {args.resume}")
        shard_meta = {}
        for name in selected_sources:
            print(f"\nCollecting {name} shards...")
            shard_meta[name] = generate_source_shards(
                source_name=name,
                quota=quotas[name],
                cfg=cfg,
                tokenizer=tokenizer,
                output=args.output,
                tokenizer_batch_paragraphs=args.tokenizer_batch_paragraphs,
                min_chars=args.min_chars,
                shard_size=args.shard_size,
                resume=args.resume,
            )

        os.makedirs(args.output, exist_ok=True)
        write_json(
            Path(args.output) / "shard_plan.json",
            {
                "mode": "generate_shards",
                "mix": proportions,
                "quotas": quotas,
                "selected_sources": selected_sources,
                "shard_size": args.shard_size,
                "meta": shard_meta,
            },
        )
        print("\nShard generation complete.")
        print("To combine later:")
        print(
            "  python generate_dataset.py --mode combine_shards "
            f"--config {args.config} --output {args.output} "
            f"--total_paragraphs {args.total_paragraphs} --mix {args.mix} "
            f"--val_paragraphs {val_size}"
        )
        return

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

    save_final_dataset(
        all_samples,
        args.output,
        val_size,
        args.seed,
        {
            "source_meta": source_meta,
            "mix": proportions,
            "quotas": quotas,
            "excluded_sources": {
                name: spec.reason
                for name, spec in SOURCE_SPECS.items()
                if not spec.supported
            },
            "tokenizer": cfg["data"]["tokenizer"],
            "max_tokens_per_sentence": cfg["data"]["max_tokens_per_sentence"],
            "min_sentences": cfg["data"]["min_sentences"],
            "max_sentences": cfg["data"]["max_sentences"],
        },
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Saved mixed dataset to {args.output}/")
    print(f"{'=' * 60}")
    print(f"  elapsed:  {elapsed:.0f}s")
    print("\nTo train:")
    print(
        "  python train.py --config "
        f"{args.config} --override data.preprocessed_path={args.output}"
    )


if __name__ == "__main__":
    main()
