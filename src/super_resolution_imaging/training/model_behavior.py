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

from .baselines import bicubic, metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import load_model


def image(values: torch.Tensor, scale: float = 1.0) -> Image.Image:
    array = (values.detach().cpu().numpy() / scale).clip(0, 1)
    return Image.fromarray(np.uint8(array * 255))


def gradient_magnitude(values: torch.Tensor) -> torch.Tensor:
    sobel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    sobel_y = sobel_x.transpose(-1, -2)
    dx = F.conv2d(values, sobel_x, padding=1)
    dy = F.conv2d(values, sobel_y, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def laplacian_rms(values: torch.Tensor) -> torch.Tensor:
    kernel = values.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    response = F.conv2d(values, kernel, padding=1)
    return torch.sqrt(response.square().mean(dim=(1, 2, 3)))


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def group_summary(rows: list[dict[str, float | str]], selector: np.ndarray, features: tuple[str, ...]) -> dict[str, Any]:
    chosen = [row for row, keep in zip(rows, selector) if keep]
    return {
        "samples": len(chosen),
        "feature_means": {feature: float(np.mean([float(row[feature]) for row in chosen])) for feature in features},
    }


def save_ranked_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    labels = ("target", "bicubic", "direct", "residual", "4x direct error", "4x residual error")
    canvas = Image.new("L", (6 * 256, len(examples) * 280), 255)
    for row, example in enumerate(examples):
        values = (
            example["target"], example["bicubic"], example["direct"], example["residual"],
            (example["direct"] - example["target"]).abs() * 4,
            (example["residual"] - example["target"]).abs() * 4,
        )
        for column, (value, label) in enumerate(zip(values, labels)):
            tile = Image.new("L", (256, 280), 255)
            tile.paste(image(value), (0, 24))
            title = f"{example['sample_id']} {label} dPSNR={example['delta_psnr']:+.2f}"
            ImageDraw.Draw(tile).text((4, 5), title, fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze which image types benefit from residual output.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDED_IDS))
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset = ZemaxBurstDataset(args.dataset, "b", args.split, exclude_ids=tuple(args.exclude))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    direct, direct_checkpoint = load_model(args.direct, "b", device)
    residual, residual_checkpoint = load_model(args.residual, "b", device)
    if bool(direct_checkpoint["configuration"].get("residual_output", False)):
        raise ValueError("--direct is a residual-output checkpoint")
    if not bool(residual_checkpoint["configuration"].get("residual_output", False)):
        raise ValueError("--residual is not a residual-output checkpoint")

    rows: list[dict[str, float | str]] = []
    example_by_id: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in loader:
            inputs, target = batch["input"].to(device), batch["target"].to(device)
            base = bicubic(inputs[:, :1])
            prediction_direct, prediction_residual = direct(inputs), residual(inputs)
            direct_metrics = metrics(prediction_direct, target)
            residual_metrics = metrics(prediction_residual, target)
            base_metrics = metrics(base, target)
            gradient = gradient_magnitude(target)
            target_std = target.std(dim=(1, 2, 3))
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {
                    "sample_id": sample_id,
                    "target_mean": float(target[index].mean()),
                    "target_std": float(target_std[index]),
                    "gradient_mean": float(gradient[index].mean()),
                    "strong_edge_fraction": float((gradient[index] > 0.10).float().mean()),
                    "laplacian_rms": float(laplacian_rms(target[index:index + 1])[0]),
                    "near_black_fraction": float((target[index] < 0.05).float().mean()),
                    "near_white_fraction": float((target[index] > 0.95).float().mean()),
                    "bicubic_psnr_db": float(base_metrics[1][index]),
                    "bicubic_ssim": float(base_metrics[2][index]),
                    "correction_rms": float(torch.sqrt((prediction_residual[index] - base[index]).square().mean())),
                    "direct_l1": float(direct_metrics[0][index]),
                    "direct_psnr_db": float(direct_metrics[1][index]),
                    "direct_ssim": float(direct_metrics[2][index]),
                    "residual_l1": float(residual_metrics[0][index]),
                    "residual_psnr_db": float(residual_metrics[1][index]),
                    "residual_ssim": float(residual_metrics[2][index]),
                }
                row["delta_l1_improvement"] = float(row["direct_l1"]) - float(row["residual_l1"])
                row["delta_psnr_db"] = float(row["residual_psnr_db"]) - float(row["direct_psnr_db"])
                row["delta_ssim"] = float(row["residual_ssim"]) - float(row["direct_ssim"])
                rows.append(row)
                example_by_id[sample_id] = {
                    "sample_id": sample_id, "target": target[index, 0].cpu(), "bicubic": base[index, 0].cpu(),
                    "direct": prediction_direct[index, 0].cpu(), "residual": prediction_residual[index, 0].cpu(),
                    "delta_psnr": float(row["delta_psnr_db"]),
                }

    features = (
        "target_mean", "target_std", "gradient_mean", "strong_edge_fraction", "laplacian_rms",
        "near_black_fraction", "near_white_fraction", "bicubic_psnr_db", "bicubic_ssim", "correction_rms",
    )
    psnr_delta = np.asarray([float(row["delta_psnr_db"]) for row in rows])
    ssim_delta = np.asarray([float(row["delta_ssim"]) for row in rows])
    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "direct_checkpoint": str(args.direct.resolve()), "residual_checkpoint": str(args.residual.resolve()),
        "groups": {
            "psnr_improved": group_summary(rows, psnr_delta > 0, features),
            "psnr_degraded": group_summary(rows, psnr_delta < 0, features),
            "ssim_improved": group_summary(rows, ssim_delta > 0, features),
            "ssim_degraded": group_summary(rows, ssim_delta < 0, features),
        },
        "spearman_feature_correlations": {
            feature: {
                "with_delta_psnr": correlation(rank(np.asarray([float(row[feature]) for row in rows])), rank(psnr_delta)),
                "with_delta_ssim": correlation(rank(np.asarray([float(row[feature]) for row in rows])), rank(ssim_delta)),
            }
            for feature in features
        },
    }
    by_psnr = sorted(rows, key=lambda row: float(row["delta_psnr_db"]))
    by_ssim = sorted(rows, key=lambda row: float(row["delta_ssim"]))
    summary["largest_psnr_gains"] = [row["sample_id"] for row in reversed(by_psnr[-5:])]
    summary["largest_psnr_losses"] = [row["sample_id"] for row in by_psnr[:5]]
    summary["largest_ssim_gains"] = [row["sample_id"] for row in reversed(by_ssim[-5:])]
    summary["ssim_degraded_samples"] = [row["sample_id"] for row in by_ssim if float(row["delta_ssim"]) < 0]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked = [example_by_id[str(row["sample_id"])] for row in list(reversed(by_psnr[-5:])) + by_psnr[:5]]
    save_ranked_examples(output / "best_and_worst_psnr.png", ranked)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
