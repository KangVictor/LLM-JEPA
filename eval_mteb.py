"""Evaluate the SentenceJEPA encoder on MTEB tasks.

Usage:
    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task STS12

    python eval_mteb.py --config configs/default.yaml \
        --checkpoint checkpoints/step_50000.pt \
        --task Banking77Classification \
        --batch_size 128
"""

import argparse
import json
import os

import mteb
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType, Array
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.model import SentenceEncoder


class SentenceJEPAWrapper:
    """Wraps the SentenceEncoder for MTEB evaluation."""

    mteb_model_meta = ModelMeta.create_empty(overwrites={
        "name": "SentenceJEPA-Small",
        "revision": "1",
        "framework": ["PyTorch"],
        "embed_dim": 256,
        "max_tokens": 48,
        "similarity_fn_name": "cosine",
        "open_weights": True,
    })

    def __init__(self, cfg, checkpoint_path, device="cuda", batch_size=64):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = cfg["encoder"]["max_seq_len"]

        # Load encoder
        self.encoder = SentenceEncoder(cfg).to(self.device)

        # Extract encoder weights from full SentenceJEPA checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        self.encoder.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["data"]["tokenizer"], use_fast=True
        )

        step = ckpt.get("step", "?")
        print(f"Loaded encoder from {checkpoint_path} (step {step})")

    def encode(
        self,
        inputs,
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

        for batch in inputs:
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Reshape (B, T) -> (B, 1, T) to match encoder's expected (B, S, T)
            input_ids = encoded["input_ids"].unsqueeze(1).to(self.device)
            attention_mask = encoded["attention_mask"].unsqueeze(1).to(self.device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = self.encoder(input_ids, attention_mask)  # (B, 1, H)

            all_embeddings.append(emb.squeeze(1).float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SentenceJEPA on MTEB")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--task", type=str, required=True, help="MTEB task name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="results/mteb")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = SentenceJEPAWrapper(
        cfg, args.checkpoint, device=args.device, batch_size=args.batch_size
    )

    tasks = mteb.get_tasks(tasks=[args.task])
    print(f"\nRunning MTEB task: {args.task}")

    results = mteb.evaluate(model, tasks=tasks)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{args.task}.json")

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
