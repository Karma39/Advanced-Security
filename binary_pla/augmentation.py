"""SpecAugment-style augmentation for DGT matrices (training only)."""

import torch


class SpecAugmentDGT:
    """
    SpecAugment-style augmentation for 150×150 DGT power matrices.

    Parameters
    ----------
    T_max     : maximum number of consecutive time steps to mask
    F_max     : maximum number of consecutive frequency bins to mask
    noise_std : std of additive Gaussian noise (relative to signal scale)
    p         : probability of applying each individual augmentation
    """

    def __init__(self, T_max=30, F_max=30, noise_std=0.02, p=0.5):
        self.T_max     = T_max
        self.F_max     = F_max
        self.noise_std = noise_std
        self.p         = p

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, T, F) DGT matrices. Returns augmented tensor, same shape."""
        X = X.clone()
        B, T, F = X.shape

        # Time masking
        if torch.rand(1).item() < self.p:
            t = torch.randint(0, self.T_max + 1, (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            X[:, t0:t0 + t, :] = 0.0

        # Frequency masking
        if torch.rand(1).item() < self.p:
            f = torch.randint(0, self.F_max + 1, (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            X[:, :, f0:f0 + f] = 0.0

        # Additive Gaussian noise
        if torch.rand(1).item() < self.p:
            X = X + torch.randn_like(X) * self.noise_std

        return X
