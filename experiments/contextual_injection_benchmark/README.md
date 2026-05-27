# Contextual Injection Benchmark

This experiment checks whether a trained Sentence-JEPA model assigns unusually high leave-one-sentence-out prediction error to an inserted prompt-injection-like sentence inside an otherwise normal paragraph.

## Inputs

The benchmark builder expects clean paragraph JSONL:

```json
{"doc_id": "example_id", "text": "paragraph text here"}
```

Payloads are one sentence per line. A small default file is included at:

```text
experiments/contextual_injection_benchmark/payloads/default_payloads.txt
```

The payloads are for detection experiments only and are loosely inspired by public prompt-injection challenge data such as `microsoft/llmail-inject-challenge`.

## Build A Benchmark

If you already have SentenceJEPA preprocessed shards, first decode them into the clean JSONL format:

```bash
python experiments/contextual_injection_benchmark/make_clean_corpus_from_shards.py \
  --input_path /path/to/preprocessed_dataset \
  --output_path experiments/contextual_injection_benchmark/outputs/clean_corpus.jsonl \
  --split train \
  --max_examples 10000
```

For a final sharded dataset this reads `train_shards/train_*.pt` by default. For source shards created before final combine, use:

```bash
python experiments/contextual_injection_benchmark/make_clean_corpus_from_shards.py \
  --input_path /path/to/preprocessed_dataset \
  --output_path experiments/contextual_injection_benchmark/outputs/clean_corpus.jsonl \
  --split source_shards
```

The converter decodes token IDs back into text with the configured tokenizer, so the recovered text is approximate but sufficient for this benchmark. It also writes a `sentences` field so the benchmark builder does not need to recover sentence boundaries from lowercased BERT-decoded text.

```bash
python experiments/contextual_injection_benchmark/build_benchmark.py \
  --clean_corpus experiments/contextual_injection_benchmark/outputs/clean_corpus.jsonl \
  --output_path experiments/contextual_injection_benchmark/outputs/benchmark.jsonl \
  --num_examples 500 \
  --seed 42
```

By default this writes both clean control examples and injected examples. The inserted sentence is placed at a random non-edge position, and each example is kept within `--max_sentences` sentences after injection.

Output rows look like:

```json
{
  "example_id": "...",
  "doc_id": "...",
  "attack_type": "clean",
  "sentences": ["...", "..."],
  "labels": [0, 0],
  "injected_indices": [],
  "metadata": {}
}
```

## Evaluate

```bash
python experiments/contextual_injection_benchmark/evaluate.py \
  --benchmark_path experiments/contextual_injection_benchmark/outputs/benchmark.jsonl \
  --checkpoint checkpoints/step_50000.pt \
  --output_dir experiments/contextual_injection_benchmark/outputs/eval
```

The evaluator loads the full `SentenceJEPA` checkpoint, including both encoder and predictor. For each paragraph, it:

1. Tokenizes all sentences with the configured tokenizer.
2. Runs the encoder once to get actual sentence embeddings.
3. Repeats the sentence embedding sequence once per sentence.
4. Masks one sentence position in each copy.
5. Runs the existing predictor to get leave-one-out predicted embeddings.
6. Scores each sentence as `1 - cosine(predicted_embedding, actual_embedding)`.

It also reports two simple embedding baselines:

- `neighbor`: compare each sentence to the mean of its previous and next sentence embeddings.
- `centroid`: compare each sentence to the mean of all other sentence embeddings in the paragraph.

## Outputs

The evaluation directory contains:

```text
scores_per_sentence.csv
metrics.json
top_suspicious_sentences.md
```

Metrics include top-1 localization accuracy, recall@2, sentence-level AUROC/AUPRC, and clean-paragraph false positive rates using JEPA z-score thresholds `z > 2` and `z > 3`.

## Assumptions

- A trained checkpoint is required because the benchmark tests predictor error, not just encoder geometry.
- The checkpoint config is used by default when it is present; pass `--no_checkpoint_config` to force `--config`.
- Benchmark examples must fit the model's configured `data.max_sentences`.
- Paragraph z-scores use the sentence scores within the same paragraph. If a paragraph has near-zero score variance, all z-scores are set to zero.

## Limitations

- The default payloads are small and synthetic; add more payload files for broader coverage.
- The sentence splitter is the repository's existing regex splitter, so unusual punctuation can still create imperfect sentence boundaries.
- This benchmark measures localization of out-of-context instructions, not whether a downstream LLM would actually follow the injected instruction.
