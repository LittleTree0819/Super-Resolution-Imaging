from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .baselines import bicubic, metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import confidence_interval, load_model


def image(values: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.uint8(values.detach().cpu().clamp(0, 1).numpy() * 255))


def save_preview(path: Path, examples: list[dict[str, Any]]) -> None:
    names = ("bicubic", "direct", "residual", "target", "direct_error", "residual_error")
    canvas = Image.new("L", (6 * 256, len(examples) * 280), 255)
    for row, example in enumerate(examples):
        for column, name in enumerate(names):
            tile = Image.new("L", (256, 280), 255)
            tile.paste(image(example[name]), (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{example['sample_id']} {name}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct and residual B-model checkpoints.")
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
        raise ValueError("--direct checkpoint is configured for residual output")
    if not bool(residual_checkpoint["configuration"].get("residual_output", False)):
        raise ValueError("--residual checkpoint is not configured for residual output")

    rows: list[dict[str, float | str]] = []
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, target = batch["input"].to(device), batch["target"].to(device)
            predictions = {"bicubic": bicubic(inputs[:, :1]), "direct": direct(inputs), "residual": residual(inputs)}
            computed = {name: metrics(prediction, target) for name, prediction in predictions.items()}
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {"sample_id": sample_id}
                for name, (l1, psnr, ssim) in computed.items():
                    row[f"{name}_l1"] = float(l1[index])
                    row[f"{name}_psnr_db"] = float(psnr[index])
                    row[f"{name}_ssim"] = float(ssim[index])
                for metric in ("l1", "psnr_db", "ssim"):
                    row[f"residual_minus_direct_{metric}"] = float(row[f"residual_{metric}"]) - float(row[f"direct_{metric}"])
                rows.append(row)
                if len(examples) < 4:
                    direct_image, residual_image = predictions["direct"][index, 0].cpu(), predictions["residual"][index, 0].cpu()
                    target_image = target[index, 0].cpu()
                    examples.append({
                        "sample_id": sample_id, "bicubic": predictions["bicubic"][index, 0].cpu(),
                        "direct": direct_image, "residual": residual_image, "target": target_image,
                        "direct_error": (direct_image - target_image).abs() * 4,
                        "residual_error": (residual_image - target_image).abs() * 4,
                    })

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "direct_checkpoint": str(args.direct.resolve()), "residual_checkpoint": str(args.residual.resolve()),
        "direct_best_epoch": direct_checkpoint["best_epoch"], "residual_best_epoch": residual_checkpoint["best_epoch"],
        "metrics": {}, "residual_improvement_over_direct": {},
    }
    for name in ("bicubic", "direct", "residual"):
        summary["metrics"][name] = {
            metric: float(np.mean([float(row[f"{name}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }
    for metric in ("l1", "psnr_db", "ssim"):
        raw = np.asarray([float(row[f"residual_minus_direct_{metric}"]) for row in rows])
        improvement = -raw if metric == "l1" else raw
        lower, upper = confidence_interval(improvement)
        summary["residual_improvement_over_direct"][metric] = {
            "mean": float(improvement.mean()), "median": float(np.median(improvement)),
            "ci95_mean": [lower, upper], "residual_better_samples": int((improvement > 0).sum()),
            "direct_better_samples": int((improvement < 0).sum()), "ties": int((improvement == 0).sum()),
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
