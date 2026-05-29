import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_metrics(enc_out, sentence_mask):
    """Compute embedding quality metrics.

    Args:
        enc_out: (B, S, H) encoder embeddings
        sentence_mask: (B, S) bool — True for real sentences

    Returns:
        dict of metric name -> value
    """
    # Flatten to real embeddings only: (N, H)
    embs = enc_out[sentence_mask].float()
    N, H = embs.shape

    if N < 2:
        return {}

    metrics = {}
    metrics["emb/sample_count"] = float(N)

    # Embedding norms
    norms = embs.norm(dim=1)
    metrics["emb/norm_mean"] = norms.mean().item()
    metrics["emb/norm_std"] = norms.std(unbiased=False).item()
    metrics["emb/norm_min"] = norms.min().item()
    metrics["emb/norm_max"] = norms.max().item()

    feature_mean = embs.mean(dim=0)
    metrics["emb/feature_mean_abs"] = feature_mean.abs().mean().item()

    # Per-dimension variance
    dim_var = embs.var(dim=0, unbiased=False)  # (H,)
    metrics["emb/var_per_dim_mean"] = dim_var.mean().item()
    metrics["emb/var_per_dim_min"] = dim_var.min().item()
    metrics["emb/var_per_dim_max"] = dim_var.max().item()

    # Pairwise cosine similarity (subsample if large)
    if N > 1000:
        idx = torch.randperm(N, device=embs.device)[:1000]
        sample = embs[idx]
    else:
        sample = embs
    sample_norm = F.normalize(sample, dim=1)
    cos_sim = sample_norm @ sample_norm.T
    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    pairwise = cos_sim[mask]
    metrics["emb/cosine_sim_mean"] = pairwise.mean().item()
    metrics["emb/cosine_sim_std"] = pairwise.std(unbiased=False).item()
    metrics["emb/cosine_sim_p95"] = pairwise.quantile(0.95).item()

    # Effective rank (participation ratio of covariance eigenvalues)
    centered = embs - embs.mean(dim=0, keepdim=True)
    # Use SVD for numerical stability
    _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = s.pow(2) / (N - 1)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = eigenvalues.pow(2).sum()
    effective_rank = (
        0.0 if sum_eig_sq <= 1e-24 else (sum_eig.pow(2) / sum_eig_sq).item()
    )
    metrics["emb/effective_rank"] = effective_rank
    metrics["emb/effective_rank_max"] = float(min(H, N - 1))
    metrics["emb/effective_rank_frac"] = effective_rank / min(H, N - 1)
    eig_sum = sum_eig.clamp_min(1e-24)
    eig_sorted = eigenvalues.sort(descending=True).values
    metrics["emb/cov_trace"] = sum_eig.item()
    metrics["emb/eig_top1_frac"] = (eig_sorted[:1].sum() / eig_sum).item()
    metrics["emb/eig_top5_frac"] = (eig_sorted[:5].sum() / eig_sum).item()

    return metrics


def log_step(step, losses, metrics, mask_counts, wandb_run):
    """Log training metrics to wandb.

    Args:
        step: training step
        losses: dict with loss_total, loss_pred, loss_sigreg
        metrics: dict from compute_metrics
        mask_counts: (B,) tensor of mask counts per sample
        wandb_run: wandb run object (or None if disabled)
    """
    log_dict = {
        "step": step,
        "loss/total": losses["total"],
        "loss/prediction": losses["prediction"],
        "loss/sigreg": losses["sigreg"],
    }
    if "sigreg_document" in losses:
        log_dict["loss/sigreg_document"] = losses["sigreg_document"]
    if "sigreg_contextual" in losses:
        log_dict["loss/sigreg_contextual"] = losses["sigreg_contextual"]
    log_dict.update(metrics)

    # Mask count distribution
    if mask_counts is not None and mask_counts.numel() > 0:
        counts = mask_counts.float()
        log_dict["masking/count_mean"] = counts.mean().item()
        log_dict["masking/count_std"] = counts.std().item()
        log_dict["masking/count_min"] = counts.min().item()
        log_dict["masking/count_max"] = counts.max().item()

    if wandb_run is not None:
        wandb_run.log(log_dict, step=step)
    else:
        # Console fallback
        loss_str = (
            f"step={step} | "
            f"loss={losses['total']:.4f} "
            f"pred={losses['prediction']:.4f} "
            f"sig={losses['sigreg']:.4f}"
        )
        if "emb/effective_rank" in metrics:
            loss_str += (
                f" | rank={metrics['emb/effective_rank']:.1f}"
                f"/{metrics.get('emb/effective_rank_max', 0):.0f}"
            )
        if "emb/cosine_sim_mean" in metrics:
            loss_str += f" | cos={metrics['emb/cosine_sim_mean']:.3f}"
        print(loss_str)


def log_val(step, losses, metrics, wandb_run):
    """Log validation metrics to wandb.

    Args:
        step: training step
        losses: dict with total, prediction, sigreg (averaged over val set)
        metrics: dict from compute_metrics
        wandb_run: wandb run object (or None if disabled)
    """
    log_dict = {
        "val/loss_total": losses["total"],
        "val/loss_prediction": losses["prediction"],
        "val/loss_sigreg": losses["sigreg"],
    }
    for k, v in metrics.items():
        log_dict[f"val/{k}"] = v

    if wandb_run is not None:
        wandb_run.log(log_dict, step=step)
    else:
        val_str = (
            f"  [val] step={step} | "
            f"loss={losses['total']:.4f} "
            f"pred={losses['prediction']:.4f} "
            f"sig={losses['sigreg']:.4f}"
        )
        if "emb/effective_rank" in metrics:
            val_str += (
                f" | rank={metrics['emb/effective_rank']:.1f}"
                f"/{metrics.get('emb/effective_rank_max', 0):.0f}"
            )
        if "emb/cosine_sim_mean" in metrics:
            val_str += f" | cos={metrics['emb/cosine_sim_mean']:.3f}"
        print(val_str)
