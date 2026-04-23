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

    def forward(self, embeddings):
        """Compute SIGReg loss on a batch of embeddings.

        Args:
            embeddings: (N, D) tensor of sentence embeddings

        Returns:
            loss: scalar — Epps-Pulley statistic averaged over projections
        """
        N, D = embeddings.shape
        if N < 2:
            return embeddings.new_tensor(0.0)

        # Random unit projection directions
        A = torch.randn(D, self.num_projections, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Project: (N, num_projections)
        proj = embeddings @ A

        # Epps-Pulley characteristic function test
        # x_t: (N, num_projections, knots)
        x_t = proj.unsqueeze(-1) * self.t

        # Compare empirical characteristic function to Gaussian phi(t) = exp(-t^2/2)
        # Real part: E[cos(x*t)] should equal phi(t)
        # Imaginary part: E[sin(x*t)] should equal 0
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()

        # Weighted integration over t
        statistic = err @ self.weights

        # Average over projections
        return statistic.mean()
