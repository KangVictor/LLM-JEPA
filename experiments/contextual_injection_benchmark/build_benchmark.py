"""Build a sentence-injection benchmark from clean paragraph JSONL."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from utils import (
    choose_sentence_window,
    clean_sentence_list,
    load_payloads,
    paragraph_to_sentences,
    read_jsonl,
    write_jsonl,
)


DEFAULT_PAYLOADS = Path(__file__).resolve().parent / "payloads" / "default_payloads.txt"


def build_examples(args):
    rng = random.Random(args.seed)
    payloads = load_payloads(args.payloads_path)

    clean_limit = args.max_sentences - 1
    if clean_limit < args.min_sentences:
        raise ValueError(
            "--max_sentences must be at least --min_sentences + 1 so an injected "
            "sentence can fit inside the model sentence limit."
        )

    built = 0
    skipped = 0
    rows = []

    for row in read_jsonl(args.clean_corpus):
        doc_id = str(row.get("doc_id", f"doc_{built + skipped}"))
        if isinstance(row.get("sentences"), list):
            sentences = clean_sentence_list(row["sentences"])
        else:
            sentences = paragraph_to_sentences(row.get("text", ""))
        sentences = choose_sentence_window(
            sentences,
            min_sentences=args.min_sentences,
            max_sentences=clean_limit,
            rng=rng,
        )
        if sentences is None or len(sentences) < args.min_sentences:
            skipped += 1
            continue

        if args.include_clean:
            rows.append(
                {
                    "example_id": f"{doc_id}::clean::{built}",
                    "doc_id": doc_id,
                    "attack_type": "clean",
                    "sentences": sentences,
                    "labels": [0 for _ in sentences],
                    "injected_indices": [],
                    "metadata": {
                        "source": str(args.clean_corpus),
                        "clean_sentence_count": len(sentences),
                        "seed": args.seed,
                    },
                }
            )

        payload_index = rng.randrange(len(payloads))
        payload = payloads[payload_index]
        insertion_index = rng.randint(1, len(sentences) - 1)
        injected_sentences = (
            sentences[:insertion_index] + [payload] + sentences[insertion_index:]
        )
        labels = [0 for _ in injected_sentences]
        labels[insertion_index] = 1

        rows.append(
            {
                "example_id": f"{doc_id}::injected_sentence::{built}",
                "doc_id": doc_id,
                "attack_type": "injected_sentence",
                "sentences": injected_sentences,
                "labels": labels,
                "injected_indices": [insertion_index],
                "metadata": {
                    "source": str(args.clean_corpus),
                    "payload": payload,
                    "payload_index": payload_index,
                    "insertion_index": insertion_index,
                    "clean_sentence_count": len(sentences),
                    "seed": args.seed,
                },
            }
        )

        built += 1
        if built >= args.num_examples:
            break

    if built < args.num_examples:
        print(
            f"Warning: requested {args.num_examples:,} examples but only built "
            f"{built:,}; skipped {skipped:,} short or invalid paragraphs."
        )

    written = write_jsonl(rows, args.output_path)
    print(f"Wrote {written:,} benchmark rows to {args.output_path}")
    print(
        f"Base clean paragraphs used: {built:,}; skipped: {skipped:,}; "
        f"clean controls included: {args.include_clean}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build contextual sentence-injection benchmark JSONL."
    )
    parser.add_argument("--clean_corpus", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--payloads_path", type=str, default=str(DEFAULT_PAYLOADS))
    parser.add_argument("--min_sentences", type=int, default=4)
    parser.add_argument("--max_sentences", type=int, default=12)
    parser.add_argument(
        "--include_clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write clean control examples for false-positive-rate metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_examples(parse_args())
