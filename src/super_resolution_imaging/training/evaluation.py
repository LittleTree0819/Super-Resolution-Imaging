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
from .model import BurstUNet2x


def load_model(path: Path, expected_group: str, device: torch.device) -> tuple[BurstUNet2x, dict[str, Any]]:
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    if configuration["group"] != expected_group:
        raise ValueError(f"{path} was trained for group {configuration['group']!r}, expected {expected_group!r}")
    model = BurstUNet2x(
        int(configuration["base_channels"]),
        residual_output=bool(configuration.get("residual_output", False)),
        split_residual_output=bool(configuration.get("split_residual_output", False)),
        split_sigma=float(configuration.get("split_sigma", 4.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def confidence_interval(values: np.ndarray, seed: int = 5400, repetitions: int = 10_000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return float(lower), float(upper)


def tensor_image(values: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.uint8(values.detach().cpu().clamp(0, 1).numpy() * 255))


def save_preview(path: Path, examples: list[dict[str, Any]]) -> None:
    labels = ("B0 bicubic", "A prediction", "B prediction", "target", "4x A error", "4x B error")
    canvas = Image.new("L", (6 * 256, len(examples) * 280), 255)
    for row, example in enumerate(examples):
        target = example["target"]
        images = (
            example["bicubic"], example["prediction_a"], example["prediction_b"], target,
            (example["prediction_a"] - target).abs() * 4,
            (example["prediction_b"] - target).abs() * 4,
        )
        for column, (values, label) in enumerate(zip(images, labels)):
            tile = Image.new("L", (256, 280), 255)
            tile.paste(tensor_image(values), (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{example['sample_id']} {label}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired evaluation for an A-input model and a B-input model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
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
    data_a = ZemaxBurstDataset(args.dataset, "a", args.split, exclude_ids=tuple(args.exclude))
    data_b = ZemaxBurstDataset(args.dataset, "b", args.split, exclude_ids=tuple(args.exclude))
    ids_a = [record["sample_id"] for record in data_a.records]
    ids_b = [record["sample_id"] for record in data_b.records]
    if ids_a != ids_b:
        raise ValueError("A and B test sample order differs")
    loader_a = DataLoader(data_a, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loader_b = DataLoader(data_b, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_a, checkpoint_a = load_model(args.checkpoint_a, "a", device)
    model_b, checkpoint_b = load_model(args.checkpoint_b, "b", device)

    rows: list[dict[str, float | str]] = []
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_a, batch_b in zip(loader_a, loader_b):
            if list(batch_a["sample_id"]) != list(batch_b["sample_id"]):
                raise ValueError("A and B batch sample order differs")
            inputs_a, inputs_b = batch_a["input"].to(device), batch_b["input"].to(device)
            target = batch_a["target"].to(device)
            prediction_a, prediction_b = model_a(inputs_a), model_b(inputs_b)
            prediction_bicubic = bicubic(inputs_a[:, :1])
            computed = {
                "bicubic": metrics(prediction_bicubic, target),
                "a": metrics(prediction_a, target),
                "b": metrics(prediction_b, target),
            }
            for index, sample_id in enumerate(batch_a["sample_id"]):
                row: dict[str, float | str] = {"sample_id": sample_id}
                for name, (l1, psnr, ssim) in computed.items():
                    row[f"{name}_l1"] = float(l1[index])
                    row[f"{name}_psnr_db"] = float(psnr[index])
                    row[f"{name}_ssim"] = float(ssim[index])
                row["b_minus_a_l1"] = float(row["b_l1"]) - float(row["a_l1"])
                row["b_minus_a_psnr_db"] = float(row["b_psnr_db"]) - float(row["a_psnr_db"])
                row["b_minus_a_ssim"] = float(row["b_ssim"]) - float(row["a_ssim"])
                rows.append(row)
                if len(examples) < 4:
                    examples.append({
                        "sample_id": sample_id, "bicubic": prediction_bicubic[index, 0].cpu(),
                        "prediction_a": prediction_a[index, 0].cpu(), "prediction_b": prediction_b[index, 0].cpu(),
                        "target": target[index, 0].cpu(),
                    })

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "checkpoint_a": str(args.checkpoint_a.resolve()), "checkpoint_b": str(args.checkpoint_b.resolve()),
        "checkpoint_a_best_epoch": checkpoint_a["best_epoch"], "checkpoint_b_best_epoch": checkpoint_b["best_epoch"],
        "metrics": {}, "paired_b_minus_a": {},
    }
    for name in ("bicubic", "a", "b"):
        summary["metrics"][name] = {
            metric: float(np.mean([float(row[f"{name}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }
    for metric in ("l1", "psnr_db", "ssim"):
        values = np.asarray([float(row[f"b_minus_a_{metric}"]) for row in rows])
        lower, upper = confidence_interval(values)
        improvement = values < 0 if metric == "l1" else values > 0
        summary["paired_b_minus_a"][metric] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "ci95_mean": [lower, upper], "b_better_samples": int(improvement.sum()),
            "a_better_samples": int((~improvement & (values != 0)).sum()), "ties": int((values == 0).sum()),
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0].keys())
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_preview(output / "preview.png", examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
