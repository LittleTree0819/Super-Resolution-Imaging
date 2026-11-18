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

from .baselines import metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import confidence_interval, load_model


def gradient_mean(values: torch.Tensor) -> torch.Tensor:
    sx = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    sy = sx.transpose(-1, -2)
    dx, dy = F.conv2d(values, sx, padding=1), F.conv2d(values, sy, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1e-12).mean(dim=(1, 2, 3))


def image(values: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.uint8(values.detach().cpu().clamp(0, 1).numpy() * 255))


def save_preview(path: Path, examples: list[dict[str, Any]]) -> None:
    names = ("reference", "ssim", "target", "reference_error", "ssim_error")
    canvas = Image.new("L", (5 * 256, len(examples) * 280), 255)
    for row, example in enumerate(examples):
        for column, name in enumerate(names):
            tile = Image.new("L", (256, 280), 255)
            tile.paste(image(example[name]), (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{example['sample_id']} {name}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def subgroup(rows: list[dict[str, float | str]], indices: np.ndarray) -> dict[str, float | int]:
    selected = [rows[int(index)] for index in indices]
    return {
        "samples": len(selected),
        "mean_delta_l1_improvement": float(np.mean([float(row["delta_l1_improvement"]) for row in selected])),
        "mean_delta_psnr_db": float(np.mean([float(row["delta_psnr_db"]) for row in selected])),
        "mean_delta_ssim": float(np.mean([float(row["delta_ssim"]) for row in selected])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare split-residual training with and without SSIM loss.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ssim", type=Path, required=True)
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
    reference, reference_checkpoint = load_model(args.reference, "b", device)
    ssim_model, ssim_checkpoint = load_model(args.ssim, "b", device)
    if float(reference_checkpoint["configuration"].get("ssim_loss_weight", 0.0)) != 0.0:
        raise ValueError("--reference must have zero SSIM loss weight")
    if float(ssim_checkpoint["configuration"].get("ssim_loss_weight", 0.0)) <= 0.0:
        raise ValueError("--ssim must have a positive SSIM loss weight")

    rows: list[dict[str, float | str]] = []
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, target = batch["input"].to(device), batch["target"].to(device)
            predictions = {"reference": reference(inputs), "ssim": ssim_model(inputs)}
            computed = {name: metrics(value, target) for name, value in predictions.items()}
            gradients = gradient_mean(target)
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {
                    "sample_id": sample_id, "target_mean": float(target[index].mean()),
                    "target_gradient_mean": float(gradients[index]),
                }
                for name, (l1, psnr, ssim) in computed.items():
                    row[f"{name}_l1"] = float(l1[index])
                    row[f"{name}_psnr_db"] = float(psnr[index])
                    row[f"{name}_ssim"] = float(ssim[index])
                row["delta_l1_improvement"] = float(row["reference_l1"]) - float(row["ssim_l1"])
                row["delta_psnr_db"] = float(row["ssim_psnr_db"]) - float(row["reference_psnr_db"])
                row["delta_ssim"] = float(row["ssim_ssim"]) - float(row["reference_ssim"])
                rows.append(row)
                if len(examples) < 4:
                    reference_image, ssim_image, target_image = predictions["reference"][index, 0].cpu(), predictions["ssim"][index, 0].cpu(), target[index, 0].cpu()
                    examples.append({
                        "sample_id": sample_id, "reference": reference_image, "ssim": ssim_image, "target": target_image,
                        "reference_error": (reference_image - target_image).abs() * 4,
                        "ssim_error": (ssim_image - target_image).abs() * 4,
                    })

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "reference_checkpoint": str(args.reference.resolve()), "ssim_checkpoint": str(args.ssim.resolve()),
        "ssim_loss_weight": float(ssim_checkpoint["configuration"]["ssim_loss_weight"]),
        "metrics": {}, "ssim_loss_improvement": {}, "subgroups": {},
    }
    for name in ("reference", "ssim"):
        summary["metrics"][name] = {
            metric: float(np.mean([float(row[f"{name}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }
    for metric, field in (("l1", "delta_l1_improvement"), ("psnr_db", "delta_psnr_db"), ("ssim", "delta_ssim")):
        values = np.asarray([float(row[field]) for row in rows])
        lower, upper = confidence_interval(values)
        summary["ssim_loss_improvement"][metric] = {
            "mean": float(values.mean()), "median": float(np.median(values)), "ci95_mean": [lower, upper],
            "ssim_model_better_samples": int((values > 0).sum()),
            "reference_better_samples": int((values < 0).sum()), "ties": int((values == 0).sum()),
        }
    count = max(1, len(rows) // 4)
    texture_order = np.argsort([float(row["target_gradient_mean"]) for row in rows])
    brightness_order = np.argsort([float(row["target_mean"]) for row in rows])
    summary["subgroups"] = {
        "lowest_texture_quartile": subgroup(rows, texture_order[:count]),
        "highest_texture_quartile": subgroup(rows, texture_order[-count:]),
        "darkest_quartile": subgroup(rows, brightness_order[:count]),
        "brightest_quartile": subgroup(rows, brightness_order[-count:]),
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_preview(output / "preview.png", examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
