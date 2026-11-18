from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baselines import bicubic
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import load_model


def gaussian_blur(values: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    radius = int(round(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)[None, None]
    padded = F.pad(values, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_2d)


def gradient_magnitude(values: torch.Tensor) -> torch.Tensor:
    sobel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    sobel_y = sobel_x.transpose(-1, -2)
    dx = F.conv2d(values, sobel_x, padding=1)
    dy = F.conv2d(values, sobel_y, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def signed_image(values: torch.Tensor, scale: float) -> Image.Image:
    normalized = (values.detach().cpu().numpy() / max(scale, 1e-8)).clip(-1, 1)
    rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(127 + 128 * np.maximum(normalized, 0))
    rgb[..., 2] = np.uint8(127 + 128 * np.maximum(-normalized, 0))
    rgb[..., 1] = np.uint8(127 * (1 - np.abs(normalized)))
    return Image.fromarray(rgb)


def gray_image(values: torch.Tensor, scale: float = 1.0) -> Image.Image:
    array = (values.detach().cpu().numpy() / max(scale, 1e-8)).clip(0, 1)
    return Image.fromarray(np.uint8(array * 255)).convert("RGB")


def correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return (left * right).sum(dim=1) / (
        torch.sqrt(left.square().sum(dim=1) * right.square().sum(dim=1)) + 1e-12
    )


def save_examples(path: Path, examples: list[dict[str, Any]], signed_scale: float) -> None:
    names = ("bicubic", "target", "prediction", "residual", "abs_residual", "low", "high")
    width, tile_height = 256, 280
    canvas = Image.new("RGB", (len(names) * width, len(examples) * tile_height), "white")
    for row, example in enumerate(examples):
        for column, name in enumerate(names):
            values = example[name]
            tile = Image.new("RGB", (width, tile_height), "white")
            if name in {"residual", "low", "high"}:
                rendered = signed_image(values, signed_scale)
            elif name == "abs_residual":
                rendered = gray_image(values, signed_scale)
            else:
                rendered = gray_image(values)
            tile.paste(rendered, (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{example['sample_id']} {name}", fill="black")
            canvas.paste(tile, (column * width, row * tile_height))
    canvas.save(path)


def save_aggregate(path: Path, mean_signed: torch.Tensor, mean_absolute: torch.Tensor, signed_scale: float) -> None:
    canvas = Image.new("RGB", (512, 280), "white")
    panels = (
        (signed_image(mean_signed, signed_scale), "dataset mean signed residual"),
        (gray_image(mean_absolute, signed_scale), "dataset mean absolute residual"),
    )
    for column, (rendered, label) in enumerate(panels):
        tile = Image.new("RGB", (256, 280), "white")
        tile.paste(rendered, (0, 24))
        ImageDraw.Draw(tile).text((5, 5), label, fill="black")
        canvas.paste(tile, (column * 256, 0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and quantify a residual-output model's corrections.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--blur-sigma", type=float, default=4.0)
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDED_IDS))
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset = ZemaxBurstDataset(args.dataset, "b", args.split, exclude_ids=tuple(args.exclude))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, checkpoint = load_model(args.checkpoint, "b", device)
    if not bool(checkpoint["configuration"].get("residual_output", False)):
        raise ValueError("The checkpoint is not configured for residual output")

    rows: list[dict[str, float | str]] = []
    examples: list[dict[str, Any]] = []
    signed_sum = torch.zeros(256, 256)
    absolute_sum = torch.zeros(256, 256)
    residual_values: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            inputs, target = batch["input"].to(device), batch["target"].to(device)
            base = bicubic(inputs[:, :1])
            prediction = model(inputs)
            residual = prediction - base
            ideal = target - base
            low = gaussian_blur(residual, args.blur_sigma)
            high = residual - low
            edges = gradient_magnitude(target)
            thresholds = torch.quantile(edges.flatten(1), 0.75, dim=1)[:, None, None, None]
            edge_mask = edges >= thresholds
            nonedge_mask = ~edge_mask

            residual_abs = residual.abs()
            residual_rms = torch.sqrt(residual.square().mean(dim=(1, 2, 3)))
            low_rms = torch.sqrt(low.square().mean(dim=(1, 2, 3)))
            high_rms = torch.sqrt(high.square().mean(dim=(1, 2, 3)))
            edge_mae = (residual_abs * edge_mask).sum(dim=(1, 2, 3)) / edge_mask.sum(dim=(1, 2, 3))
            nonedge_mae = (residual_abs * nonedge_mask).sum(dim=(1, 2, 3)) / nonedge_mask.sum(dim=(1, 2, 3))
            edge_energy_fraction = (residual_abs * edge_mask).sum(dim=(1, 2, 3)) / residual_abs.sum(dim=(1, 2, 3)).clamp_min(1e-12)
            edge_correlation = correlation(residual_abs, edges)
            ideal_correlation = correlation(residual, ideal)

            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append({
                    "sample_id": sample_id,
                    "signed_mean": float(residual[index].mean()),
                    "mean_absolute": float(residual_abs[index].mean()),
                    "rms": float(residual_rms[index]),
                    "low_frequency_rms_ratio": float(low_rms[index] / residual_rms[index].clamp_min(1e-12)),
                    "high_frequency_rms_ratio": float(high_rms[index] / residual_rms[index].clamp_min(1e-12)),
                    "edge_to_nonedge_mae_ratio": float(edge_mae[index] / nonedge_mae[index].clamp_min(1e-12)),
                    "edge_energy_fraction": float(edge_energy_fraction[index]),
                    "abs_residual_edge_correlation": float(edge_correlation[index]),
                    "residual_ideal_correlation": float(ideal_correlation[index]),
                })
                signed_sum += residual[index, 0].cpu()
                absolute_sum += residual_abs[index, 0].cpu()
                residual_values.append(residual[index, 0].cpu().flatten())
                if len(examples) < 6:
                    examples.append({
                        "sample_id": sample_id,
                        "bicubic": base[index, 0].cpu(), "target": target[index, 0].cpu(),
                        "prediction": prediction[index, 0].cpu(), "residual": residual[index, 0].cpu(),
                        "abs_residual": residual_abs[index, 0].cpu(), "low": low[index, 0].cpu(),
                        "high": high[index, 0].cpu(),
                    })

    all_residuals = torch.cat(residual_values)
    signed_scale = float(torch.quantile(all_residuals.abs(), 0.99))
    mean_signed = signed_sum / len(rows)
    mean_absolute = absolute_sum / len(rows)
    field_bias_rms = float(torch.sqrt(mean_signed.square().mean()))
    typical_residual_rms = float(np.mean([float(row["rms"]) for row in rows]))
    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_best_epoch": checkpoint["best_epoch"],
        "definitions": {
            "residual": "clamped model prediction minus B0 bicubic",
            "low_frequency": f"Gaussian blur of residual, sigma={args.blur_sigma} HR pixels",
            "high_frequency": "residual minus low-frequency component",
            "edge_pixels": "top 25% of target Sobel gradient magnitude",
        },
        "mean_metrics": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0] if key != "sample_id"
        },
        "residual_display_scale_99th_abs": signed_scale,
        "dataset_mean_signed_residual_rms": field_bias_rms,
        "dataset_mean_signed_rms_over_typical_residual_rms": field_bias_rms / max(typical_residual_rms, 1e-12),
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_examples(output / "residual_examples.png", examples, signed_scale)
    save_aggregate(output / "aggregate_residual.png", mean_signed, mean_absolute, signed_scale)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
