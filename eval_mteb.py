"""Evaluate the SentenceJEPA encoder on MTEB tasks.

Usage:
    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task STS12

    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task Banking77Classification \
        --batch_size 128

    # Old independent sentence-mean behavior:
    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task AskUbuntuDupQuestions \
        --embedding_mode sentence_mean

    # Old single-sequence behavior:
    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task AskUbuntuDupQuestions \
        --embedding_mode single
"""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import mteb
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType, Array
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.data import split_sentences
from src.model import SentenceJEPA


def get_embedding_dim(cfg):
    enc = cfg["encoder"]
    return enc.get("embedding_size", enc["hidden_size"])


def resolve_precision(device, precision):
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


class SentenceJEPAWrapper:
    """Wraps SentenceJEPA for MTEB evaluation, implementing EncoderProtocol."""

    def __init__(
        self,
        cfg,
        checkpoint_path,
        device="cuda",
        batch_size=64,
        embedding_mode="document",
        max_sentences_per_text=None,
        precision="auto",
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = cfg["encoder"]["max_seq_len"]
        self.embedding_mode = embedding_mode
        self.max_sentences_per_text = max_sentences_per_text
        if self.max_sentences_per_text is None:
            self.max_sentences_per_text = cfg["data"].get("max_sentences")
        self.use_amp, self.amp_dtype, self.precision = resolve_precision(
            self.device, precision
        )
        checkpoint_id = hashlib.sha1(
            str(Path(checkpoint_path).resolve()).encode("utf-8")
        ).hexdigest()[:8]

        # Load full hierarchical encoder. The predictor is present in the
        # checkpoint but is not used by encode_document during inference.
        self.model = SentenceJEPA(cfg).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        has_document_weights = any(
            key.startswith("document_transformer.") for key in state_dict
        )
        document_modes = {"document", "document_layer_mean"}
        if self.embedding_mode in document_modes and not has_document_weights:
            raise ValueError(
                f"embedding_mode={self.embedding_mode!r} requires a hierarchical "
                "Paragraph-JEPA checkpoint with document_transformer weights. "
                "Use --embedding_mode sentence_mean for older checkpoints."
            )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                "Warning: checkpoint is missing hierarchical model keys; "
                f"first missing keys: {missing[:5]}"
            )
        if unexpected:
            print(
                "Warning: checkpoint has unexpected keys; "
                f"first unexpected keys: {unexpected[:5]}"
            )
        self.model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["data"]["tokenizer"], use_fast=True
        )
        step = ckpt.get("step", checkpoint_id)
        revision = f"step-{step}_{self.embedding_mode}_{checkpoint_id}"

        self._mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": "SentenceJEPA-Small",
            "revision": revision,
            "framework": ["PyTorch"],
            "embed_dim": get_embedding_dim(cfg),
            "max_tokens": cfg["encoder"]["max_seq_len"],
            "similarity_fn_name": "cosine",
            "open_weights": True,
        })

        print(f"Loaded encoder from {checkpoint_path} (step {step})")
        print(f"MTEB cache identity: SentenceJEPA-Small / {revision}")
        print(
            f"Embedding mode: {self.embedding_mode}, "
            f"max_sentences_per_text={self.max_sentences_per_text}, "
            f"precision={self.precision}"
        )

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    def text_to_sentences(self, text):
        text = "" if text is None else str(text).strip()
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]
        if self.max_sentences_per_text is not None:
            sentences = sentences[: self.max_sentences_per_text]
        return sentences or [text]

    def encode_single_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].unsqueeze(1).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(1).to(self.device)
        sentence_mask = torch.ones(
            input_ids.shape[:2],
            dtype=torch.bool,
            device=self.device,
        )

        with autocast_context(self.device, self.use_amp, self.amp_dtype):
            if self.embedding_mode in ("document", "document_layer_mean"):
                emb = self.model.encode_document(
                    input_ids,
                    attention_mask,
                    sentence_mask,
                    normalize=True,
                    layer_pooling=(
                        "concat_mean"
                        if self.embedding_mode == "document_layer_mean"
                        else "final"
                    ),
                )
            else:
                emb = self.model.encoder(input_ids, attention_mask).squeeze(1)

        return emb

    def encode_sentence_mean_batch(
        self,
        texts,
        contextual=False,
        layer_pooling="final",
    ):
        sentence_lists = [self.text_to_sentences(text) for text in texts]
        flat_sentences = [
            sentence
            for sentences in sentence_lists
            for sentence in sentences
        ]

        encoded = self.tokenizer(
            flat_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sentence_mask = sentence_mask.to(self.device)

        with autocast_context(self.device, self.use_amp, self.amp_dtype):
            if contextual:
                emb = self.model.encode_document(
                    input_ids,
                    attention_mask,
                    sentence_mask,
                    normalize=True,
                    layer_pooling=layer_pooling,
                )
            else:
                sent_embs = self.model.encoder(input_ids, attention_mask)  # (B, S, D)
                weights = sentence_mask.unsqueeze(-1).to(dtype=sent_embs.dtype)
                emb = (sent_embs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

        return emb

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> Array:
        """Encode sentences for MTEB.

        Args:
            inputs: DataLoader yielding batches of sentences.
            task_metadata: The task metadata.
            hf_split: The dataset split.
            hf_subset: The dataset subset.
            prompt_type: The prompt type to use.

        Returns:
            Embeddings as numpy array of shape (N, H).
        """
        all_embeddings = []

        with torch.inference_mode():
            for batch in inputs:
                # batch is a TypedDict: TextInput{"text": list[str]},
                # CorpusInput{"text": ..., "title": ..., "body": ...},
                # or QueryInput{"text": ..., "query": ...}
                # "text" is always present and is the primary field
                texts = batch["text"]
                if isinstance(texts, str):
                    texts = [texts]

                if self.embedding_mode == "single":
                    emb = self.encode_single_batch(texts)
                elif self.embedding_mode == "document":
                    emb = self.encode_sentence_mean_batch(
                        texts,
                        contextual=True,
                        layer_pooling="final",
                    )
                elif self.embedding_mode == "document_layer_mean":
                    emb = self.encode_sentence_mean_batch(
                        texts,
                        contextual=True,
                        layer_pooling="concat_mean",
                    )
                elif self.embedding_mode == "sentence_mean":
                    emb = self.encode_sentence_mean_batch(texts)
                else:
                    raise ValueError(f"Unknown embedding mode: {self.embedding_mode}")

                all_embeddings.append(emb.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute cosine similarity between two sets of embeddings."""
        e1 = torch.from_numpy(np.asarray(embeddings1, dtype=np.float32))
        e2 = torch.from_numpy(np.asarray(embeddings2, dtype=np.float32))
        e1 = torch.nn.functional.normalize(e1, dim=1)
        e2 = torch.nn.functional.normalize(e2, dim=1)
        return (e1 @ e2.T).numpy()

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute pairwise cosine similarity."""
        e1 = torch.from_numpy(np.asarray(embeddings1, dtype=np.float32))
        e2 = torch.from_numpy(np.asarray(embeddings2, dtype=np.float32))
        e1 = torch.nn.functional.normalize(e1, dim=1)
        e2 = torch.nn.functional.normalize(e2, dim=1)
        return (e1 * e2).sum(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SentenceJEPA on MTEB")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--task", type=str, required=True, help="MTEB task name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="results/mteb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--embedding_mode",
        choices=["document", "document_layer_mean", "single", "sentence_mean"],
        default="document",
        help=(
            "document: split text into sentences, contextualize them with the "
            "final document-transformer layer, then normalized mean-pool. "
            "document_layer_mean: concatenate all document-transformer layer "
            "outputs along the sentence axis, then normalized mean-pool. "
            "sentence_mean: old independent sentence mean. single: old "
            "first-max_seq_len-token behavior."
        ),
    )
    parser.add_argument(
        "--max_sentences_per_text",
        type=int,
        default=None,
        help="Sentence cap for sentence_mean mode. Defaults to data.max_sentences.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Autocast precision for encoder inference.",
    )
    parser.add_argument(
        "--overwrite_strategy",
        choices=["always", "never", "only-missing", "only-cache"],
        default="always",
        help=(
            "MTEB cache overwrite policy. Default 'always' avoids accidentally "
            "reusing results from a different checkpoint."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = SentenceJEPAWrapper(
        cfg,
        args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        embedding_mode=args.embedding_mode,
        max_sentences_per_text=args.max_sentences_per_text,
        precision=args.precision,
    )

    tasks = mteb.get_tasks(tasks=[args.task])
    print(f"\nRunning MTEB task: {args.task}")

    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy=args.overwrite_strategy,
    )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    checkpoint_name = Path(args.checkpoint).stem
    output_name = (
        f"{args.task}_{checkpoint_name}"
        if args.embedding_mode == "single"
        else f"{args.task}_{args.embedding_mode}_{checkpoint_name}"
    )
    output_path = os.path.join(args.output, f"{output_name}.json")

    results_serializable = []
    for task_result in results:
        results_serializable.append({
            "task_name": task_result.task_name,
            "scores": task_result.scores,
        })

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)

    # Print results summary
    print(f"\n{'='*50}")
    print(f"Results: {args.task}")
    print(f"{'='*50}")
    for task_result in results:
        for split, scores in task_result.scores.items():
            for score_set in scores:
                main_score = score_set.get("main_score", None)
                if main_score is not None:
                    print(f"  {split}: main_score = {main_score:.4f}")
    print(f"Saved to {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
