import math
import torch


def sigreg_loss(embeddings, num_projections=64):
    """SIGReg: enforce Gaussianity of random 1D projections.

    Projects embeddings onto random directions and penalizes deviation
    from a standard normal distribution using sorted-quantile MSE.

    Args:
        embeddings: (N, D) tensor of sentence embeddings (real sentences only)
        num_projections: number of random projection directions

    Returns:
        loss: scalar — mean squared deviation from normal quantiles
    """
    N, D = embeddings.shape
    if N < 2:
        return embeddings.new_tensor(0.0)

    device = embeddings.device

    # Random unit projection directions
    directions = torch.randn(D, num_projections, device=device)
    directions = directions / directions.norm(dim=0, keepdim=True)

    # Project: (N, num_projections)
    projections = embeddings @ directions

    # Standardize each projection
    mean = projections.mean(dim=0, keepdim=True)
    std = projections.std(dim=0, keepdim=True).clamp(min=1e-8)
    standardized = (projections - mean) / std

    # Sort along sample dimension
    sorted_vals, _ = standardized.sort(dim=0)  # (N, num_projections)

    # Expected quantiles of N(0,1) for N samples
    # Use the inverse CDF (percent point function) at evenly spaced probabilities
    probs = (torch.arange(N, device=device, dtype=torch.float32) + 0.5) / N
    expected = torch.erfinv(2 * probs - 1) * math.sqrt(2)  # (N,)
    expected = expected.unsqueeze(1)  # (N, 1)

    # MSE between sorted empirical and expected quantiles
    loss = (sorted_vals - expected).pow(2).mean()

    return loss
