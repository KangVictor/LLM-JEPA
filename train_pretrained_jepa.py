"""Fine-tune an existing Hugging Face encoder with the SentenceJEPA objective.

This keeps the current paragraph data pipeline, predictor, masking, SIGReg,
logging, validation, and checkpoint format, but replaces the scratch sentence
encoder with AutoModel.from_pretrained(...).

Example:
    python train_pretrained_jepa.py \
        --config configs/pretrained_jepa.yaml \
        --model_name_or_path bert-base-uncased \
        --override data.preprocessed_path=/path/to/processed objective.mode=masked
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModel

from src.data import (
    WikiParagraphDataset,
    collate_fn,
    load_preprocessed,
    summarize_dataset,
)
from src.logging_utils import compute_metrics, log_step, log_val
from src.model import Predictor, ProjectionHead
from src.sigreg import SIGReg
from train import (
    compute_losses,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    validate,
)


class HFSentenceEncoder(nn.Module):
    """Sentence encoder backed by a pretrained Hugging Face transformer."""

    def __init__(self, cfg):
        super().__init__()
        pre_cfg = cfg.get("pretrained_encoder", {})
        enc_cfg = cfg["encoder"]

        self.model_name_or_path = pre_cfg.get(
            "model_name_or_path",
            pre_cfg.get("model_name", "bert-base-uncased"),
        )
        self.pooling = pre_cfg.get("pooling", "mean")
        self.normalize_output = bool(pre_cfg.get("normalize_output", False))

        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=pre_cfg.get("trust_remote_code", False),
        )
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.embedding_size = int(
            enc_cfg.get(
                "embedding_size",
                pre_cfg.get("embedding_size", self.hidden_size),
            )
        )
        self.vocab_size = getattr(self.backbone.config, "vocab_size", None)

        self.projector = ProjectionHead(
            input_dim=self.hidden_size,
            hidden_dim=int(
                pre_cfg.get(
                    "projector_hidden_size",
                    enc_cfg.get("projector_hidden_size", max(2048, self.hidden_size)),
                )
            ),
            output_dim=self.embedding_size,
        )

        if pre_cfg.get("gradient_checkpointing", False):
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            else:
                print("Warning: backbone does not support gradient checkpointing.")

        if pre_cfg.get("freeze_base", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        if pre_cfg.get("freeze_embeddings", False):
            embeddings = getattr(self.backbone, "embeddings", None)
            if embeddings is not None:
                for param in embeddings.parameters():
                    param.requires_grad = False

        freeze_layers = int(pre_cfg.get("freeze_layers", 0))
        if freeze_layers > 0:
            self.freeze_first_layers(freeze_layers)

    def freeze_first_layers(self, num_layers):
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            layers = getattr(encoder, "layers", None)
        if layers is None:
            print("Warning: could not find backbone encoder layers to freeze.")
            return
        for layer in list(layers)[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, S, T)
            attention_mask: (B, S, T)
        Returns:
            sentence embeddings: (B, S, D)
        """
        B, S, T = input_ids.shape
        ids_flat = input_ids.reshape(B * S, T)
        mask_flat = attention_mask.reshape(B * S, T)
        sentence_flat = mask_flat.any(dim=1)

        if self.vocab_size is not None and ids_flat.numel() > 0:
            max_id = int(ids_flat.max().item())
            if max_id >= self.vocab_size:
                raise ValueError(
                    f"Found token id {max_id}, but pretrained model vocab_size is "
                    f"{self.vocab_size}. If you are using preprocessed shards, "
                    "they must have been tokenized with a tokenizer compatible "
                    "with this pretrained model."
                )

        # Some HF models dislike all-zero attention masks. Give empty padded
        # sentence rows one harmless attended pad token; they are zeroed later.
        model_attention_mask = mask_flat.clone()
        if (~sentence_flat).any():
            model_attention_mask[~sentence_flat, 0] = 1

        outputs = self.backbone(
            input_ids=ids_flat,
            attention_mask=model_attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = token_embeddings[:, 0]
        elif self.pooling == "pooler":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is None:
                raise ValueError("pooling='pooler' requested, but model has no pooler_output")
            pooled = pooler_output
        elif self.pooling == "mean":
            weights = mask_flat.unsqueeze(-1).to(dtype=token_embeddings.dtype)
            token_counts = weights.sum(dim=1).clamp(min=1)
            pooled = (token_embeddings * weights).sum(dim=1) / token_counts
        else:
            raise ValueError(f"Unknown pretrained_encoder.pooling: {self.pooling}")

        pooled = pooled.reshape(B, S, self.hidden_size)
        sentence_mask = sentence_flat.reshape(B, S)
        embeddings = self.projector(pooled, sentence_mask)
        if self.normalize_output:
            embeddings = F.normalize(embeddings.float(), dim=-1).to(embeddings.dtype)
        return embeddings


class PretrainedSentenceJEPA(nn.Module):
    """SentenceJEPA with a pretrained HF sentence encoder."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = HFSentenceEncoder(cfg)
        self.predictor = Predictor(cfg)

    def forward_masked(self, input_ids, attention_mask, sentence_mask, mask_indices):
        enc_out = self.encoder(input_ids, attention_mask)
        pred_out = self.predictor(enc_out, sentence_mask, mask_indices)
        targets = enc_out[mask_indices]
        return pred_out, targets, enc_out, mask_indices

    def forward_next_sentence(self, input_ids, attention_mask, sentence_mask):
        enc_out = self.encoder(input_ids, attention_mask)
        pred_seq = self.predictor.forward_sequence(enc_out, sentence_mask, causal=True)
        pair_mask = sentence_mask[:, :-1] & sentence_mask[:, 1:]
        pred_out = pred_seq[:, :-1][pair_mask]
        targets = enc_out[:, 1:][pair_mask]
        return pred_out, targets, enc_out, pair_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        sentence_mask,
        mask_indices=None,
        mode="masked",
    ):
        if mode == "next_sentence":
            return self.forward_next_sentence(input_ids, attention_mask, sentence_mask)
        if mode == "masked":
            if mask_indices is None:
                raise ValueError("mask_indices is required for masked objective")
            return self.forward_masked(
                input_ids,
                attention_mask,
                sentence_mask,
                mask_indices,
            )
        raise ValueError(f"Unknown objective mode: {mode}")


def configure_pretrained_args(cfg, args):
    pre_cfg = cfg.setdefault("pretrained_encoder", {})
    if args.model_name_or_path is not None:
        pre_cfg["model_name_or_path"] = args.model_name_or_path
    if args.tokenizer_name is not None:
        pre_cfg["tokenizer_name"] = args.tokenizer_name
    if args.pooling is not None:
        pre_cfg["pooling"] = args.pooling
    if args.embedding_size is not None:
        cfg["encoder"]["embedding_size"] = int(args.embedding_size)
    if args.encoder_lr is not None:
        pre_cfg["encoder_lr"] = float(args.encoder_lr)
    if args.head_lr is not None:
        pre_cfg["head_lr"] = float(args.head_lr)
    if args.freeze_base:
        pre_cfg["freeze_base"] = True
    if args.trust_remote_code:
        pre_cfg["trust_remote_code"] = True

    model_name = pre_cfg.get("model_name_or_path", "bert-base-uncased")
    tokenizer_name = pre_cfg.get("tokenizer_name", model_name)

    # Raw streaming/local datasets are tokenized at runtime, so use the
    # pretrained tokenizer by default. Preprocessed shards are already token IDs;
    # they must have been generated with a compatible tokenizer.
    if cfg["data"].get("preprocessed_path"):
        print(
            "Using preprocessed token IDs. Make sure the shards were tokenized "
            f"with a tokenizer compatible with {model_name!r}."
        )
    else:
        cfg["data"]["tokenizer"] = tokenizer_name

    # Predictor reads cfg['encoder']['embedding_size']; keep it explicit.
    cfg["encoder"]["embedding_size"] = int(cfg["encoder"].get("embedding_size", 256))
    return cfg


def build_loaders(cfg):
    batch_size = cfg["training"]["batch_size"]
    if cfg["data"].get("preprocessed_path"):
        train_dataset, val_samples, num_train = load_preprocessed(cfg)
        is_iterable_train = isinstance(train_dataset, IterableDataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=not is_iterable_train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    else:
        num_train, val_samples = summarize_dataset(cfg)
        train_dataset = WikiParagraphDataset(cfg)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_samples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, num_train


def parameter_count(params):
    return sum(p.numel() for p in params)


def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    pre_cfg = cfg.get("pretrained_encoder", {})

    encoder_lr = float(pre_cfg.get("encoder_lr", train_cfg.get("encoder_lr", 2.0e-5)))
    head_lr = float(pre_cfg.get("head_lr", train_cfg.get("head_lr", train_cfg["lr"])))
    weight_decay = train_cfg["weight_decay"]

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.backbone."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "pretrained_encoder",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "jepa_heads",
            }
        )

    print(
        f"Trainable pretrained encoder params: {parameter_count(encoder_params):,} "
        f"(lr={encoder_lr:g})"
    )
    print(
        f"Trainable JEPA head/predictor params: {parameter_count(head_params):,} "
        f"(lr={head_lr:g})"
    )
    return torch.optim.AdamW(param_groups)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained encoder with JEPA")
    parser.add_argument("--config", type=str, default="configs/pretrained_jepa.yaml")
    parser.add_argument("--override", nargs="*", help="Config overrides: key.subkey=value")
    parser.add_argument("--resume_from", "--resume-from", dest="resume_from", default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--pooling", choices=("mean", "cls", "pooler"), default=None)
    parser.add_argument("--embedding_size", type=int, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    cfg = configure_pretrained_args(load_config(args), args)
    train_cfg = cfg["training"]
    set_seed(train_cfg["seed"])

    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb

        wandb_run = wandb.init(project=cfg["wandb"]["project"], config=cfg)

    train_loader, val_loader, num_train = build_loaders(cfg)
    batch_size = train_cfg["batch_size"]
    steps_per_epoch = num_train // batch_size
    total_steps = steps_per_epoch * train_cfg["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedSentenceJEPA(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Pretrained backbone: {model.encoder.model_name_or_path}")
    print(f"Sentence pooling: {model.encoder.pooling}")
    print(f"JEPA embedding size: {model.encoder.embedding_size}")

    sig_cfg = cfg["sigreg"]
    sigreg = (
        SIGReg(
            knots=sig_cfg.get("knots", 17),
            num_projections=sig_cfg["num_projections"],
        ).to(device)
        if sig_cfg["enabled"]
        else None
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        train_cfg["warmup_steps"],
        total_steps,
    )

    resume_from = args.resume_from or train_cfg.get("resume_from")
    step = 0
    if resume_from:
        step = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        if step >= total_steps:
            raise ValueError(
                f"Checkpoint is already at step {step:,}, but this run is "
                f"configured for {total_steps:,} steps."
            )

    use_amp = train_cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if train_cfg["precision"] == "bf16" else torch.float16

    model.train()
    start_step = step
    start_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
    mode = cfg.get("objective", {}).get("mode", "masked")
    mask_combinations = int(cfg["masking"].get("combinations_per_sample", 1))

    print(f"Starting pretrained JEPA fine-tuning for {train_cfg['epochs']} epochs")
    print(f"Steps per epoch: {steps_per_epoch:,}, total steps: {total_steps:,}")
    print(f"Batch size: {batch_size}, precision: {train_cfg['precision']}")
    print(f"Objective: {mode}")
    print(f"Mask combinations per paragraph: {mask_combinations}")
    print(f"SIGReg: {sig_cfg['enabled']}, weight={sig_cfg['weight']}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{train_cfg['epochs']} ---")
        epoch_end_step = min((epoch + 1) * steps_per_epoch, total_steps)

        for batch in train_loader:
            if step >= epoch_end_step:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            sentence_mask = batch["sentence_mask"].to(device, non_blocking=True)

            result = compute_losses(
                model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "sentence_mask": sentence_mask,
                },
                cfg,
                sigreg,
                amp_dtype,
                use_amp,
                mask_combinations=mask_combinations,
            )
            if result is None:
                continue
            (
                loss_total,
                loss_pred,
                loss_sig,
                metric_out,
                metric_mask,
                mask_counts,
                sig_losses,
            ) = result

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step = int(scheduler.last_epoch)

            if step % train_cfg["log_every"] == 0:
                losses = {
                    "total": loss_total.item(),
                    "prediction": loss_pred.item(),
                    "sigreg": loss_sig.item() if torch.is_tensor(loss_sig) else loss_sig,
                    "sigreg_document": sig_losses["document"].item(),
                    "sigreg_contextual": sig_losses["contextual"].item(),
                }
                metrics = compute_metrics(metric_out.detach(), metric_mask)
                log_step(step, losses, metrics, mask_counts, wandb_run)

            if step > 0 and step % train_cfg["val_every"] == 0:
                val_result = validate(
                    model,
                    val_loader,
                    cfg,
                    device,
                    amp_dtype,
                    use_amp,
                    sigreg,
                )
                if val_result is not None:
                    val_losses, val_metrics = val_result
                    log_val(step, val_losses, val_metrics, wandb_run)

            if step > 0 and step % train_cfg["save_every"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    cfg,
                    steps_per_epoch=steps_per_epoch,
                    total_steps=total_steps,
                )

        if step == start_step:
            continue
        val_result = validate(model, val_loader, cfg, device, amp_dtype, use_amp, sigreg)
        if val_result is not None:
            val_losses, val_metrics = val_result
            log_val(step, val_losses, val_metrics, wandb_run)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            cfg,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
        )
        print(f"Epoch {epoch + 1} complete at step {step}")

        if step >= total_steps:
            break

    print(f"\nPretrained JEPA fine-tuning complete: step {step}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
