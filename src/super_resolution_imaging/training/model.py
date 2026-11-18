from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_blur(values: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    radius = int(round(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)[None, None]
    padded = F.pad(values, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_2d)


def frequency_split(values: torch.Tensor, sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    low = gaussian_blur(values, sigma)
    return low, values - low


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, 2, stride=2)
        self.conv = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((value, skip), dim=1))


class BurstUNet2x(nn.Module):
    """Burst U-Net with direct or B0-bicubic residual 2x output."""

    def __init__(
        self, base_channels: int = 24, residual_output: bool = False,
        split_residual_output: bool = False, split_sigma: float = 4.0,
        input_channels: int = 4,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.input_channels = input_channels
        if residual_output and split_residual_output:
            raise ValueError("residual_output and split_residual_output are mutually exclusive")
        self.residual_output = residual_output
        self.split_residual_output = split_residual_output
        self.split_sigma = split_sigma
        c = base_channels
        self.enc1 = ConvBlock(input_channels, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.bottleneck = ConvBlock(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpBlock(8 * c, 4 * c, 4 * c)
        self.up2 = UpBlock(4 * c, 2 * c, 2 * c)
        self.up1 = UpBlock(2 * c, c, c)
        output_convolution = nn.Conv2d(c, 4, 3, padding=1)
        if split_residual_output:
            low_convolution = nn.Conv2d(c, 4, 3, padding=1)
            high_convolution = nn.Conv2d(c, 4, 3, padding=1)
            nn.init.zeros_(low_convolution.weight)
            nn.init.zeros_(low_convolution.bias)
            nn.init.zeros_(high_convolution.weight)
            nn.init.zeros_(high_convolution.bias)
            self.low_head = nn.Sequential(low_convolution, nn.PixelShuffle(2))
            self.high_head = nn.Sequential(high_convolution, nn.PixelShuffle(2))
            self.sr_head = None
        elif residual_output:
            nn.init.zeros_(output_convolution.weight)
            nn.init.zeros_(output_convolution.bias)
            self.sr_head = nn.Sequential(output_convolution, nn.PixelShuffle(2))
        else:
            self.sr_head = nn.Sequential(output_convolution, nn.PixelShuffle(2), nn.Sigmoid())

    def _features(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        center = self.bottleneck(self.pool(e3))
        value = self.up3(center, e3)
        value = self.up2(value, e2)
        return self.up1(value, e1)

    def forward_with_components(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = F.interpolate(value[:, :1], scale_factor=2, mode="bicubic", align_corners=False, antialias=True)
        features = self._features(value)
        if self.split_residual_output:
            low = gaussian_blur(self.low_head(features), self.split_sigma)
            high_raw = self.high_head(features)
            high = high_raw - gaussian_blur(high_raw, self.split_sigma)
            prediction = torch.clamp(base + low + high, 0.0, 1.0)
            return prediction, {"base": base, "low": low, "high": high}
        reconstruction = self.sr_head(features)
        if self.residual_output:
            prediction = torch.clamp(base + reconstruction, 0.0, 1.0)
            return prediction, {"base": base, "residual": reconstruction}
        return reconstruction, {"base": base}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.forward_with_components(value)[0]
