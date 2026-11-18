from __future__ import annotations

import torch
from torch.nn import functional as F
from .model import gaussian_blur


def global_brightness_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image mean luminance error, averaged over the batch."""
    dimensions = tuple(range(1, prediction.ndim))
    return (prediction.mean(dimensions) - target.mean(dimensions)).abs().mean()


def local_low_frequency_loss(
    prediction: torch.Tensor, target: torch.Tensor, sigma: float = 4.0,
) -> torch.Tensor:
    """Compare spatial low frequencies after removing each filtered image's mean."""
    error = gaussian_blur(prediction - target, sigma)
    return (error - error.mean(dim=(1, 2, 3), keepdim=True)).abs().mean()


def structural_similarity(prediction: torch.Tensor, target: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """Return one differentiable SSIM value per image using the evaluation convention."""
    mu_x = F.avg_pool2d(prediction, kernel, stride=1, padding=kernel // 2)
    mu_y = F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2)
    sigma_x = F.avg_pool2d(prediction.square(), kernel, stride=1, padding=kernel // 2) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), kernel, stride=1, padding=kernel // 2) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, kernel, stride=1, padding=kernel // 2) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    result = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return result.mean(dim=(1, 2, 3))


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - structural_similarity(prediction, target).mean()
