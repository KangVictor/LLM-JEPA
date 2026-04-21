import argparse
import os
import random

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import WikiParagraphDataset, collate_fn
from src.logging_utils import compute_metrics, log_step
from src.masking import sample_masks
from src.model import SentenceJEPA
from src.sigreg import sigreg_loss


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


def save_checkpoint(model, optimizer, scheduler, step, cfg):
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SentenceJEPA")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--override", nargs="*", help="Config overrides: key.subkey=value"
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

    # Data
    dataset = WikiParagraphDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        collate_fn=collate_fn,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceJEPA(cfg).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg["warmup_steps"], train_cfg["max_steps"]
    )

    # Mixed precision
    use_amp = train_cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if train_cfg["precision"] == "bf16" else torch.float16

    # Training loop
    model.train()
    step = 0
    max_steps = train_cfg["max_steps"]

    print(f"Starting training for {max_steps} steps...")
    print(f"Batch size: {train_cfg['batch_size']}, Precision: {train_cfg['precision']}")
    print(f"SIGReg: {cfg['sigreg']['enabled']}, Multi-mask: {cfg['masking']['multi_mask']}")
    print(f"Predictor layers: {cfg['predictor']['num_layers']}")
    print(f"Mask ratio: [{cfg['masking']['mask_ratio_min']}, {cfg['masking']['mask_ratio_max']}]")

    data_iter = iter(loader)
    while step < max_steps:
        # Get next batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sentence_mask = batch["sentence_mask"].to(device, non_blocking=True)

        # Sample masks
        mask_indices, mask_counts = sample_masks(sentence_mask, cfg["masking"])

        # Skip if no masks (shouldn't happen, but safety)
        if mask_indices.sum() == 0:
            continue

        # Forward pass
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred_out, targets, enc_out = model(
                input_ids, attention_mask, sentence_mask, mask_indices
            )
            loss_pred = F.mse_loss(pred_out, targets)

            if cfg["sigreg"]["enabled"]:
                real_embs = enc_out[sentence_mask]  # (N, H)
                loss_sig = sigreg_loss(real_embs, cfg["sigreg"]["num_projections"])
                loss_total = loss_pred + cfg["sigreg"]["weight"] * loss_sig
            else:
                loss_sig = torch.tensor(0.0, device=device)
                loss_total = loss_pred

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        # Logging
        if step % train_cfg["log_every"] == 0:
            losses = {
                "total": loss_total.item(),
                "prediction": loss_pred.item(),
                "sigreg": loss_sig.item() if torch.is_tensor(loss_sig) else loss_sig,
            }
            metrics = compute_metrics(enc_out.detach(), sentence_mask)
            log_step(step, losses, metrics, mask_counts, wandb_run)

        # Checkpointing
        if step > 0 and step % train_cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, step, cfg)

        step += 1

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, cfg)
    print(f"Training complete at step {step}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
