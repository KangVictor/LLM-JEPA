import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Uses the Epps-Pulley characteristic function test to enforce that
    random 1D projections of embeddings follow N(0,1). Based on the
    Le-WM implementation (https://github.com/lucas-maes/le-wm).

    This tests both shape AND scale — collapsed embeddings (near-zero
    variance) will produce a high loss because their characteristic
    function won't match the Gaussian one.
    """

    def __init__(self, knots=17, num_projections=1024):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings, mask=None):
        """Compute sample-count-normalized SIGReg loss.

        Args:
            embeddings: (S, B, D) tensor of sentence embeddings, or (N, D)
                for one global set of valid sentence embeddings.
            mask: optional (S, B) bool tensor marking real embeddings.

        Returns:
            loss: scalar, averaged over projections and valid sample groups.
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        S, B, D = embeddings.shape
        if B < 2:
            return embeddings.new_tensor(0.0)

        # Random unit projection directions
        A = torch.randn(D, self.num_projections, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Project: (S, B, num_projections)
        proj = embeddings @ A

        # Epps-Pulley characteristic function test
        # x_t: (S, B, num_projections, knots)
        x_t = proj.unsqueeze(-1) * self.t

        # Compare empirical characteristic function to Gaussian phi(t) = exp(-t^2/2)
        # Real part: E[cos(x*t)] should equal phi(t)
        # Imaginary part: E[sin(x*t)] should equal 0
        if mask is None:
            err = (x_t.cos().mean(1) - self.phi).square() + x_t.sin().mean(1).square()
            valid_steps = torch.ones(S, dtype=torch.bool, device=embeddings.device)
        else:
            mask = mask.to(dtype=torch.bool, device=embeddings.device)
            mask_weights = mask[:, :, None, None].to(dtype=x_t.dtype)
            sample_counts = mask_weights.sum(dim=1).squeeze(-1).squeeze(-1)
            denom = sample_counts.clamp(min=1).view(S, 1, 1)
            cos_mean = (x_t.cos() * mask_weights).sum(dim=1) / denom
            sin_mean = (x_t.sin() * mask_weights).sum(dim=1) / denom
            err = (cos_mean - self.phi).square() + sin_mean.square()
            valid_steps = sample_counts >= 2

        # Weighted integration over t. Do not multiply by sample count: the
        # empirical characteristic function already averages over samples.
        statistic = err @ self.weights
        if not valid_steps.any():
            return embeddings.new_tensor(0.0)
        statistic = statistic[valid_steps]

        # Average over projections and sequence positions.
        return statistic.mean()
