from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import confidence_interval, load_model


FRAME_LABELS = {1: "B1_x", 2: "B2_y", 3: "B3_xy"}


def combination_name(indices: tuple[int, ...]) -> str:
    return "B0" if not indices else "B0+" + "+".join(FRAME_LABELS[index] for index in indices)


def input_combinations(frames: torch.Tensor) -> dict[tuple[int, ...], torch.Tensor]:
    result: dict[tuple[int, ...], torch.Tensor] = {}
    for size in range(4):
        for indices in itertools.combinations((1, 2, 3), size):
            value = frames[:, :1].repeat(1, 4, 1, 1)
            for index in indices:
                value[:, index] = frames[:, index]
            result[indices] = value
    return result


def shapley_values(values: dict[tuple[int, ...], float]) -> dict[int, float]:
    players = (1, 2, 3)
    result: dict[int, float] = {}
    for player in players:
        contribution = 0.0
        others = tuple(value for value in players if value != player)
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                with_player = tuple(sorted((*subset, player)))
                weight = math.factorial(size) * math.factorial(len(players) - size - 1) / math.factorial(len(players))
                contribution += weight * (values[with_player] - values[subset])
        result[player] = contribution
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every B1/B2/B3 input-frame combination.")
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
    keys = tuple(input_combinations(torch.zeros(1, 4, 1, 1)).keys())
    rows: list[dict[str, float | str]] = []

    with torch.inference_mode():
        for batch in loader:
            frames, target = batch["input"].to(device), batch["target"].to(device)
            variants = input_combinations(frames)
            computed = {key: metrics(model(value), target) for key, value in variants.items()}
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {"sample_id": sample_id}
                for key, (l1, psnr, ssim) in computed.items():
                    name = combination_name(key)
                    row[f"{name}_l1"] = float(l1[index])
                    row[f"{name}_psnr_db"] = float(psnr[index])
                    row[f"{name}_ssim"] = float(ssim[index])
                rows.append(row)

    mean_metrics: dict[tuple[int, ...], dict[str, float]] = {}
    for key in keys:
        name = combination_name(key)
        mean_metrics[key] = {
            metric: float(np.mean([float(row[f"{name}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_best_epoch": checkpoint["best_epoch"],
        "method": "Missing B1/B2/B3 channels are replaced by B0; model weights remain fixed.",
        "frame_meaning": {"B0": "center", "B1": "+x", "B2": "+y", "B3": "+x,+y"},
        "combination_metrics": {combination_name(key): mean_metrics[key] for key in keys},
        "shapley_contribution": {}, "full_over_b0": {}, "full_over_each_ablation": {},
    }
    for metric in ("l1", "psnr_db", "ssim"):
        utility = {key: (-mean_metrics[key][metric] if metric == "l1" else mean_metrics[key][metric]) for key in keys}
        shapley = shapley_values(utility)
        summary["shapley_contribution"][metric] = {FRAME_LABELS[key]: value for key, value in shapley.items()}
        sample_improvements = []
        for row in rows:
            b0 = float(row[f"B0_{metric}"])
            full = float(row[f"{combination_name((1, 2, 3))}_{metric}"])
            sample_improvements.append(b0 - full if metric == "l1" else full - b0)
        values = np.asarray(sample_improvements)
        lower, upper = confidence_interval(values)
        summary["full_over_b0"][metric] = {
            "mean": float(values.mean()), "ci95_mean": [lower, upper],
            "full_better_samples": int((values > 0).sum()), "b0_better_samples": int((values < 0).sum()),
        }

    full_key = (1, 2, 3)
    for removed in (1, 2, 3):
        reduced = tuple(value for value in full_key if value != removed)
        label = FRAME_LABELS[removed]
        summary["full_over_each_ablation"][label] = {}
        for metric in ("l1", "psnr_db", "ssim"):
            values = np.asarray([
                float(row[f"{combination_name(reduced)}_{metric}"]) - float(row[f"{combination_name(full_key)}_{metric}"])
                if metric == "l1" else
                float(row[f"{combination_name(full_key)}_{metric}"]) - float(row[f"{combination_name(reduced)}_{metric}"])
                for row in rows
            ])
            lower, upper = confidence_interval(values)
            summary["full_over_each_ablation"][label][metric] = {
                "mean": float(values.mean()), "ci95_mean": [lower, upper],
                "full_better_samples": int((values > 0).sum()), "without_frame_better_samples": int((values < 0).sum()),
            }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
