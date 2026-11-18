"""Quantify calibration stability and extra-frame gain in the nine-frame pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from super_resolution_imaging.training.baselines import bicubic, metrics, shift_and_add


FOUR_FRAME_INDICES = (0, 1, 7, 8)


def load_gray(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)), "p95": float(np.percentile(array, 95)),
        "min": float(array.min()), "max": float(array.max()),
    }


def save_preview(path: Path, examples: list[tuple[str, list[torch.Tensor]]]) -> None:
    labels = ("B0 bicubic", "4-frame S&A", "9-frame S&A", "target")
    canvas = Image.new("L", (4 * 256, len(examples) * 280), 255)
    for row, (sample_id, images) in enumerate(examples):
        for column, (tensor, label) in enumerate(zip(images, labels)):
            tile = Image.new("L", (256, 280), 255)
            image = Image.fromarray(np.uint8(tensor.clamp(0, 1).numpy() * 255))
            tile.paste(image, (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{sample_id} {label}", fill=0)
            canvas.paste(tile, (column * 256, row * 280))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    shifts = tuple(tuple(calibration["frames"][str(i)]["median_measured_shift_xy"]) for i in range(9))
    shifts_four = tuple(shifts[i] for i in FOUR_FRAME_INDICES)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    rows: list[dict[str, float | str]] = []
    examples: list[tuple[str, list[torch.Tensor]]] = []
    for record_index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        sample = root / "samples" / sample_id
        frames = torch.stack([load_gray(sample / "b" / f"frame_{i:02d}.png") for i in range(9)])[None]
        target = load_gray(sample / "target.png")[None, None]
        predictions = {
            "bicubic": bicubic(frames[:, :1]),
            "shift_add_4": shift_and_add(frames[:, FOUR_FRAME_INDICES], shifts_four),
            "shift_add_9": shift_and_add(frames, shifts),
        }
        row: dict[str, float | str] = {"sample_id": sample_id}
        for name, prediction in predictions.items():
            l1, psnr, ssim = metrics(prediction, target)
            row[f"{name}_l1"] = float(l1[0])
            row[f"{name}_psnr_db"] = float(psnr[0])
            row[f"{name}_ssim"] = float(ssim[0])
        row["nine_minus_four_psnr_db"] = float(row["shift_add_9_psnr_db"]) - float(row["shift_add_4_psnr_db"])
        row["nine_minus_four_ssim"] = float(row["shift_add_9_ssim"]) - float(row["shift_add_4_ssim"])
        means = frames.mean(dim=(0, 2, 3)).numpy()
        row["frame_mean_range"] = float(means.max() - means.min())
        row["max_abs_mean_delta_from_center"] = float(np.max(np.abs(means - means[0])))
        rows.append(row)
        if record_index in {0, max(0, len(records) // 3), max(0, 2 * len(records) // 3), len(records) - 1}:
            examples.append((sample_id, [predictions[name][0, 0] for name in predictions] + [target[0, 0]]))

    methods = ("bicubic", "shift_add_4", "shift_add_9")
    report = {
        "samples": len(rows),
        "four_frame_indices": FOUR_FRAME_INDICES,
        "calibrated_shifts_lr_pixels": shifts,
        "metrics": {
            method: {
                metric: float(np.mean([float(row[f"{method}_{metric}"]) for row in rows]))
                for metric in ("l1", "psnr_db", "ssim")
            } for method in methods
        },
        "nine_minus_four": {
            "psnr_db": summarize([float(row["nine_minus_four_psnr_db"]) for row in rows]),
            "ssim": summarize([float(row["nine_minus_four_ssim"]) for row in rows]),
            "psnr_improved_samples": sum(float(row["nine_minus_four_psnr_db"]) > 0 for row in rows),
            "ssim_improved_samples": sum(float(row["nine_minus_four_ssim"]) > 0 for row in rows),
        },
        "brightness": {
            "frame_mean_range": summarize([float(row["frame_mean_range"]) for row in rows]),
            "max_abs_mean_delta_from_center": summarize([float(row["max_abs_mean_delta_from_center"]) for row in rows]),
        },
    }
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_preview(output / "preview.png", examples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
