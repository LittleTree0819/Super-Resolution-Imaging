from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .losses import structural_similarity


def bicubic(frame: torch.Tensor) -> torch.Tensor:
    return F.interpolate(frame, scale_factor=2, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def four_frame_mean(frames: torch.Tensor) -> torch.Tensor:
    return bicubic(frames.mean(dim=1, keepdim=True))


def shift_and_add(
    frames: torch.Tensor,
    shifts_lr_pixels: tuple[tuple[float, float], ...],
    origin_offset: float = 0.5,
) -> torch.Tensor:
    """Bilinearly splat LR samples onto the reference 2x grid and normalize weights.

    A positive image shift means scene content moved toward increasing image x/y.
    Such a sample is placed at reference coordinate (x - shift_x, y - shift_y).
    """
    batch, channels, height, width = frames.shape
    if channels != len(shifts_lr_pixels):
        raise ValueError("One image-plane shift is required for every input frame")
    device, dtype = frames.device, frames.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    accumulated = torch.zeros((batch, 1, 2 * height, 2 * width), device=device, dtype=dtype)
    weights = torch.zeros_like(accumulated)
    flat_accumulated = accumulated.view(batch, -1)
    flat_weights = weights.view(batch, -1)
    for channel, (shift_x, shift_y) in enumerate(shifts_lr_pixels):
        target_x = 2.0 * (xx - shift_x) + origin_offset
        target_y = 2.0 * (yy - shift_y) + origin_offset
        x0, y0 = torch.floor(target_x), torch.floor(target_y)
        values = frames[:, channel].reshape(batch, -1)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xi, yi = x0 + dx, y0 + dy
            weight = (1.0 - torch.abs(target_x - xi)) * (1.0 - torch.abs(target_y - yi))
            valid = (xi >= 0) & (xi < 2 * width) & (yi >= 0) & (yi < 2 * height)
            indices = (yi.clamp(0, 2 * height - 1) * (2 * width) + xi.clamp(0, 2 * width - 1)).long().reshape(-1)
            expanded_indices = indices.unsqueeze(0).expand(batch, -1)
            expanded_weight = (weight * valid).reshape(1, -1).expand(batch, -1)
            flat_accumulated.scatter_add_(1, expanded_indices, values * expanded_weight)
            flat_weights.scatter_add_(1, expanded_indices, expanded_weight)
    result = accumulated / weights.clamp_min(1e-8)
    missing = weights <= 1e-8
    if missing.any():
        fallback = bicubic(frames[:, :1])
        result = torch.where(missing, fallback, result)
    return result.clamp(0, 1)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimensions = (1, 2, 3)
    l1 = torch.mean(torch.abs(prediction - target), dim=dimensions)
    mse = torch.mean((prediction - target) ** 2, dim=dimensions)
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
    ssim = structural_similarity(prediction, target)
    return l1, psnr, ssim


def evaluate(
    loader: DataLoader,
    shifts: tuple[tuple[float, float], ...],
    origin_offset: float,
) -> tuple[list[dict[str, float | str]], dict[str, dict[str, float]]]:
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for batch in loader:
            frames, target = batch["input"], batch["target"]
            predictions = {
                "bicubic": bicubic(frames[:, :1]),
                "mean": four_frame_mean(frames),
                "shift_and_add": shift_and_add(frames, shifts, origin_offset),
            }
            for method, prediction in predictions.items():
                l1, psnr, ssim = metrics(prediction, target)
                for index, sample_id in enumerate(batch["sample_id"]):
                    rows.append({"sample_id": sample_id, "method": method, "l1": float(l1[index]),
                                 "psnr_db": float(psnr[index]), "ssim": float(ssim[index])})
    summary: dict[str, dict[str, float]] = {}
    for method in ("bicubic", "mean", "shift_and_add"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([float(row[key]) for row in selected])) for key in ("l1", "psnr_db", "ssim")
        }
    return rows, summary


def save_preview(
    path: Path,
    dataset: ZemaxBurstDataset,
    shifts: tuple[tuple[float, float], ...],
    origin_offset: float,
    count: int = 4,
) -> None:
    labels = ("B0 bicubic", "4-frame mean", "shift-and-add", "target")
    canvas = Image.new("L", (4 * 256, count * 280), 255)
    for row in range(min(count, len(dataset))):
        item = dataset[row]
        frames = item["input"].unsqueeze(0)
        images = (bicubic(frames[:, :1])[0, 0], four_frame_mean(frames)[0, 0],
                  shift_and_add(frames, shifts, origin_offset)[0, 0], item["target"][0])
        for column, (values, label) in enumerate(zip(images, labels)):
            image = Image.fromarray(np.uint8(values.clamp(0, 1).numpy() * 255))
            tile = Image.new("L", (256, 280), 255)
            tile.paste(image, (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{item['sample_id']} {label}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-learned super-resolution baselines.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--x-image-shift", type=float, default=0.508)
    parser.add_argument("--y-image-shift", type=float, default=-0.506)
    parser.add_argument("--origin-offset", type=float, choices=(0.0, 0.5), default=0.5)
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDED_IDS))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = ZemaxBurstDataset(args.dataset, "b", args.split, args.limit, tuple(args.exclude))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    shifts = ((0.0, 0.0), (args.x_image_shift, 0.0), (0.0, args.y_image_shift),
              (args.x_image_shift, args.y_image_shift))
    rows, summary = evaluate(loader, shifts, args.origin_offset)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "method", "l1", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerows(rows)
    report = {"split": args.split, "samples": len(dataset), "shifts_lr_pixels": shifts,
              "origin_offset": args.origin_offset, "metrics": summary}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_preview(output / "preview.png", dataset, shifts, args.origin_offset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
