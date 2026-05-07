import argparse
import os
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import WikiParagraphDataset, collate_fn, summarize_dataset, load_preprocessed
from src.logging_utils import compute_metrics, log_step, log_val
from src.masking import sample_masks
from src.model import SentenceJEPA
from src.sigreg import SIGReg


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d[p]
            # Parse value type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            d[parts[-1]] = value

    return cfg


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay learning rate schedule with linear warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    cfg,
    steps_per_epoch=None,
    total_steps=None,
):
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step}.pt")
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }
    if steps_per_epoch is not None:
        checkpoint["steps_per_epoch"] = steps_per_epoch
    if total_steps is not None:
        checkpoint["total_steps"] = total_steps
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Restore training state and return the global completed step."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain optimizer_state_dict; cannot resume "
            "with the exact optimizer/LR state."
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ckpt_step = int(checkpoint.get("step", 0))
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is None:
        raise KeyError(
            "Checkpoint does not contain scheduler_state_dict; cannot resume "
            "with a learning rate consistent with the previous step."
        )

    scheduler.load_state_dict(scheduler_state)
    scheduler_step = int(scheduler_state.get("last_epoch", ckpt_step))
    resume_step = scheduler_step
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Loaded checkpoint: {path}")
    print(
        f"Resuming from global step {resume_step:,} "
        f"(checkpoint step={ckpt_step:,}, scheduler step={scheduler_step:,}, "
        f"lr={current_lr:.6g})"
    )
    return resume_step


def compute_losses(model, batch, cfg, sigreg, amp_dtype, use_amp):
    """Run one batch and compute prediction + SIGReg losses."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    sentence_mask = batch["sentence_mask"]
    mode = cfg.get("objective", {}).get("mode", "next_sentence")

    mask_counts = None
    device_type = input_ids.device.type
    if device_type in ("cuda", "cpu"):
        autocast_ctx = torch.amp.autocast(
            device_type, dtype=amp_dtype, enabled=use_amp
        )
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        if mode == "next_sentence":
            pred_out, targets, enc_out, pred_mask = model(
                input_ids,
                attention_mask,
                sentence_mask,
                mode="next_sentence",
            )
            mask_counts = pred_mask.sum(dim=1)
        elif mode == "masked":
            mask_indices, mask_counts = sample_masks(sentence_mask, cfg["masking"])
            if mask_indices.sum() == 0:
                return None
            pred_out, targets, enc_out, _ = model(
                input_ids,
                attention_mask,
                sentence_mask,
                mask_indices=mask_indices,
                mode="masked",
            )
        else:
            raise ValueError(f"Unknown objective mode: {mode}")

        if pred_out.numel() == 0:
            return None

        loss_pred = F.mse_loss(pred_out, targets)

        if sigreg is not None:
            sigreg_embs = enc_out.transpose(0, 1)  # (S, B, D)
            sigreg_mask = sentence_mask.transpose(0, 1)  # (S, B)
            loss_sig = sigreg(sigreg_embs, sigreg_mask)
            loss_total = loss_pred + cfg["sigreg"]["weight"] * loss_sig
        else:
            loss_sig = enc_out.new_tensor(0.0)
            loss_total = loss_pred

    return loss_total, loss_pred, loss_sig, enc_out, mask_counts


@torch.no_grad()
def validate(model, val_loader, cfg, device, amp_dtype, use_amp, sigreg=None):
    """Run validation and return averaged losses + metrics."""
    model.eval()
    total_pred, total_sig, total_loss, num_batches = 0.0, 0.0, 0.0, 0

    all_enc = []
    all_smask = []

    for batch in val_loader:
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
        )
        if result is None:
            continue
        loss_total, loss_pred, loss_sig, enc_out, _ = result

        total_pred += loss_pred.item()
        total_sig += loss_sig.item()
        total_loss += loss_total.item()
        num_batches += 1
        all_enc.append(enc_out)
        all_smask.append(sentence_mask)

    model.train()

    if num_batches == 0:
        return None

    losses = {
        "total": total_loss / num_batches,
        "prediction": total_pred / num_batches,
        "sigreg": total_sig / num_batches,
    }
    # Metrics from last few batches (cap memory usage)
    enc_cat = torch.cat(all_enc[-4:], dim=0)
    smask_cat = torch.cat(all_smask[-4:], dim=0)
    metrics = compute_metrics(enc_cat, smask_cat)

    return losses, metrics


def main():
    parser = argparse.ArgumentParser(description="Train SentenceJEPA")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--override", nargs="*", help="Config overrides: key.subkey=value"
    )
    parser.add_argument(
        "--resume_from",
        "--resume-from",
        dest="resume_from",
        type=str,
        default=None,
        help="Path to a training checkpoint to resume from.",
    )
    args = parser.parse_args()

    cfg = load_config(args)
    train_cfg = cfg["training"]

    set_seed(train_cfg["seed"])

    # Wandb
    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            config=cfg,
        )

    # Load data — preprocessed (fast) or raw (streaming)
    batch_size = train_cfg["batch_size"]

    if cfg["data"].get("preprocessed_path"):
        train_dataset, val_samples, num_train = load_preprocessed(cfg)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    else:
        num_train, val_samples = summarize_dataset(cfg)
        dataset = WikiParagraphDataset(cfg)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    steps_per_epoch = num_train // batch_size
    num_epochs = train_cfg["epochs"]
    total_steps = steps_per_epoch * num_epochs

    # Validation loader (in-memory, fixed set)
    val_loader = DataLoader(
        val_samples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceJEPA(cfg).to(device)

    # SIGReg module (buffers need to be on device)
    sig_cfg = cfg["sigreg"]
    sigreg = (
        SIGReg(
            knots=sig_cfg.get("knots", 17),
            num_projections=sig_cfg["num_projections"],
        ).to(device)
        if sig_cfg["enabled"]
        else None
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg["warmup_steps"], total_steps
    )

    resume_from = args.resume_from or train_cfg.get("resume_from")
    step = 0
    if resume_from:
        step = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        if step >= total_steps:
            raise ValueError(
                f"Checkpoint is already at step {step:,}, but this run is "
                f"configured for only {total_steps:,} total steps. Increase "
                "training.epochs if you want to continue training longer."
            )

    # Mixed precision
    use_amp = train_cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if train_cfg["precision"] == "bf16" else torch.float16

    # Training loop
    model.train()
    start_step = step
    start_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

    print(f"Starting training for {num_epochs} epochs ({total_steps:,} steps)...")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    if resume_from:
        print(f"Remaining steps: {total_steps - step:,}")
    print(f"Batch size: {batch_size}, Precision: {train_cfg['precision']}")
    mode = cfg.get("objective", {}).get("mode", "next_sentence")
    print(f"Objective: {mode}")
    print(
        f"SIGReg: {sig_cfg['enabled']}, SIGReg Lambda: {sig_cfg['weight']} "
        f"Multi-mask: {cfg['masking']['multi_mask']}"
    )
    print(f"Predictor layers: {cfg['predictor']['num_layers']}")
    print(
        f"Mask ratio: "
        f"[{cfg['masking']['mask_ratio_min']}, {cfg['masking']['mask_ratio_max']}]"
    )

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_end_step = min((epoch + 1) * steps_per_epoch, total_steps)

        for batch in loader:
            if step >= epoch_end_step:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            sentence_mask = batch["sentence_mask"].to(device, non_blocking=True)

            # Forward pass
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
            )
            if result is None:
                continue
            loss_total, loss_pred, loss_sig, enc_out, mask_counts = result

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step = int(scheduler.last_epoch)

            # Logging
            if step % train_cfg["log_every"] == 0:
                losses = {
                    "total": loss_total.item(),
                    "prediction": loss_pred.item(),
                    "sigreg": loss_sig.item() if torch.is_tensor(loss_sig) else loss_sig,
                }
                metrics = compute_metrics(enc_out.detach(), sentence_mask)
                log_step(step, losses, metrics, mask_counts, wandb_run)

            # Validation
            if step > 0 and step % train_cfg["val_every"] == 0:
                val_result = validate(
                    model, val_loader, cfg, device, amp_dtype, use_amp, sigreg
                )
                if val_result is not None:
                    val_losses, val_metrics = val_result
                    log_val(step, val_losses, val_metrics, wandb_run)

            # Checkpointing
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

        # End-of-epoch validation + checkpoint
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

    print(f"\nTraining complete: {num_epochs} epochs, {step} steps")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
