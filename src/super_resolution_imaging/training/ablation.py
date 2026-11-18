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

from .baselines import metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import confidence_interval, load_model


def input_variants(frames: torch.Tensor) -> dict[str, torch.Tensor]:
    repeat_b0 = frames[:, :1].repeat(1, 4, 1, 1)
    no_b1, no_b2, no_b3 = frames.clone(), frames.clone(), frames.clone()
    no_b1[:, 1] = frames[:, 0]
    no_b2[:, 2] = frames[:, 0]
    no_b3[:, 3] = frames[:, 0]
    return {
        "full_b": frames,
        "repeat_b0": repeat_b0,
        "replace_b1": no_b1,
        "replace_b2": no_b2,
        "replace_b3": no_b3,
        "swap_b1_b2": frames[:, [0, 2, 1, 3]],
    }


def tensor_image(values: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.uint8(values.detach().cpu().clamp(0, 1).numpy() * 255))


def save_preview(path: Path, examples: list[dict[str, Any]]) -> None:
    names = ("full_b", "repeat_b0", "replace_b1", "replace_b2", "replace_b3", "target")
    canvas = Image.new("L", (6 * 256, len(examples) * 280), 255)
    for row, example in enumerate(examples):
        for column, name in enumerate(names):
            tile = Image.new("L", (256, 280), 255)
            tile.paste(tensor_image(example[name]), (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{example['sample_id']} {name}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-channel ablation for the trained B model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
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
    model, checkpoint = load_model(args.checkpoint, "b", device)
    names = tuple(input_variants(torch.zeros(1, 4, 1, 1)).keys())
    rows: list[dict[str, float | str]] = []
    examples: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            frames, target = batch["input"].to(device), batch["target"].to(device)
            variants = input_variants(frames)
            predictions = {name: model(value) for name, value in variants.items()}
            computed = {name: metrics(prediction, target) for name, prediction in predictions.items()}
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {"sample_id": sample_id}
                for name, (l1, psnr, ssim) in computed.items():
                    row[f"{name}_l1"] = float(l1[index])
                    row[f"{name}_psnr_db"] = float(psnr[index])
                    row[f"{name}_ssim"] = float(ssim[index])
                rows.append(row)
                if len(examples) < 4:
                    examples.append({"sample_id": sample_id, "target": target[index, 0].cpu(), **{
                        name: prediction[index, 0].cpu() for name, prediction in predictions.items()
                    }})

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_best_epoch": checkpoint["best_epoch"],
        "metrics": {}, "full_b_improvement_over_ablation": {},
    }
    for name in names:
        summary["metrics"][name] = {
            metric: float(np.mean([float(row[f"{name}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }
    for name in names[1:]:
        summary["full_b_improvement_over_ablation"][name] = {}
        for metric in ("l1", "psnr_db", "ssim"):
            full = np.asarray([float(row[f"full_b_{metric}"]) for row in rows])
            ablated = np.asarray([float(row[f"{name}_{metric}"]) for row in rows])
            values = ablated - full if metric == "l1" else full - ablated
            lower, upper = confidence_interval(values)
            summary["full_b_improvement_over_ablation"][name][metric] = {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "ci95_mean": [lower, upper], "full_b_better_samples": int((values > 0).sum()),
                "ablation_better_samples": int((values < 0).sum()), "ties": int((values == 0).sum()),
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
