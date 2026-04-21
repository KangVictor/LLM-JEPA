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
    embs = enc_out[sentence_mask]
    N, H = embs.shape

    if N < 2:
        return {}

    metrics = {}

    # Embedding norms
    norms = embs.norm(dim=1)
    metrics["emb/norm_mean"] = norms.mean().item()
    metrics["emb/norm_std"] = norms.std().item()

    # Per-dimension variance
    dim_var = embs.var(dim=0)  # (H,)
    metrics["emb/var_per_dim_mean"] = dim_var.mean().item()
    metrics["emb/var_per_dim_min"] = dim_var.min().item()

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
    metrics["emb/cosine_sim_std"] = pairwise.std().item()

    # Effective rank (participation ratio of covariance eigenvalues)
    centered = embs - embs.mean(dim=0, keepdim=True)
    # Use SVD for numerical stability
    _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = s.pow(2) / (N - 1)
    eigenvalues = eigenvalues.clamp(min=1e-12)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = eigenvalues.pow(2).sum()
    effective_rank = (sum_eig.pow(2) / sum_eig_sq).item()
    metrics["emb/effective_rank"] = effective_rank

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
            loss_str += f" | rank={metrics['emb/effective_rank']:.1f}"
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
        print(
            f"  [val] step={step} | "
            f"loss={losses['total']:.4f} "
            f"pred={losses['prediction']:.4f} "
            f"sig={losses['sigreg']:.4f}"
        )
