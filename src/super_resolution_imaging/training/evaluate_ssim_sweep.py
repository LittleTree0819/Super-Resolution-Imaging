from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import metrics
from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .evaluation import confidence_interval, load_model


def parse_model(value: str) -> tuple[float, Path]:
    try:
        weight, path = value.split("=", 1)
        return float(weight), Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model must be WEIGHT=CHECKPOINT") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an SSIM-loss weight sweep on one fixed split.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True, help="WEIGHT=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDED_IDS))
    args = parser.parse_args()

    models_by_weight = dict(args.model)
    if len(models_by_weight) != len(args.model) or 0.0 not in models_by_weight:
        raise ValueError("SSIM weights must be unique and include 0")
    weights = tuple(sorted(models_by_weight))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset = ZemaxBurstDataset(args.dataset, "b", args.split, exclude_ids=tuple(args.exclude))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loaded: dict[float, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for weight in weights:
        model, checkpoint = load_model(models_by_weight[weight], "b", device)
        actual = float(checkpoint["configuration"].get("ssim_loss_weight", 0.0))
        if not np.isclose(actual, weight):
            raise ValueError(f"Checkpoint weight {actual} does not match requested weight {weight}")
        loaded[weight] = model, checkpoint

    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, target = batch["input"].to(device), batch["target"].to(device)
            computed = {weight: metrics(loaded[weight][0](inputs), target) for weight in weights}
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {"sample_id": sample_id}
                for weight, (l1, psnr, ssim) in computed.items():
                    tag = f"w{weight:g}"
                    row[f"{tag}_l1"] = float(l1[index])
                    row[f"{tag}_psnr_db"] = float(psnr[index])
                    row[f"{tag}_ssim"] = float(ssim[index])
                rows.append(row)

    aggregate: dict[float, dict[str, float]] = {}
    for weight in weights:
        tag = f"w{weight:g}"
        aggregate[weight] = {
            metric: float(np.mean([float(row[f"{tag}_{metric}"]) for row in rows]))
            for metric in ("l1", "psnr_db", "ssim")
        }
    pareto: list[float] = []
    for weight in weights:
        candidate = aggregate[weight]
        dominated = any(
            other != weight
            and aggregate[other]["l1"] <= candidate["l1"]
            and aggregate[other]["psnr_db"] >= candidate["psnr_db"]
            and aggregate[other]["ssim"] >= candidate["ssim"]
            and (
                aggregate[other]["l1"] < candidate["l1"]
                or aggregate[other]["psnr_db"] > candidate["psnr_db"]
                or aggregate[other]["ssim"] > candidate["ssim"]
            )
            for other in weights
        )
        if not dominated:
            pareto.append(weight)

    summary: dict[str, Any] = {
        "split": args.split, "samples": len(rows), "device": str(device),
        "weights": list(weights), "metrics": {f"{weight:g}": aggregate[weight] for weight in weights},
        "improvement_over_weight_0": {}, "pairwise_improvements": {}, "pareto_weights": pareto,
        "best_by_metric": {
            "l1": min(weights, key=lambda value: aggregate[value]["l1"]),
            "psnr_db": max(weights, key=lambda value: aggregate[value]["psnr_db"]),
            "ssim": max(weights, key=lambda value: aggregate[value]["ssim"]),
        },
        "checkpoints": {f"{weight:g}": str(models_by_weight[weight].resolve()) for weight in weights},
    }
    for weight in weights[1:]:
        summary["improvement_over_weight_0"][f"{weight:g}"] = {}
        for metric in ("l1", "psnr_db", "ssim"):
            base = np.asarray([float(row[f"w0_{metric}"]) for row in rows])
            candidate = np.asarray([float(row[f"w{weight:g}_{metric}"]) for row in rows])
            values = base - candidate if metric == "l1" else candidate - base
            lower, upper = confidence_interval(values)
            summary["improvement_over_weight_0"][f"{weight:g}"][metric] = {
                "mean": float(values.mean()), "median": float(np.median(values)), "ci95_mean": [lower, upper],
                "weighted_model_better_samples": int((values > 0).sum()),
                "weight_0_better_samples": int((values < 0).sum()), "ties": int((values == 0).sum()),
            }

    # Report every higher-weight model against every lower-weight model. Positive
    # values always mean that the candidate (higher weight) performed better.
    for reference_index, reference_weight in enumerate(weights[:-1]):
        for candidate_weight in weights[reference_index + 1:]:
            comparison = f"{candidate_weight:g}_over_{reference_weight:g}"
            summary["pairwise_improvements"][comparison] = {}
            for metric in ("l1", "psnr_db", "ssim"):
                reference = np.asarray([
                    float(row[f"w{reference_weight:g}_{metric}"]) for row in rows
                ])
                candidate = np.asarray([
                    float(row[f"w{candidate_weight:g}_{metric}"]) for row in rows
                ])
                values = reference - candidate if metric == "l1" else candidate - reference
                lower, upper = confidence_interval(values)
                summary["pairwise_improvements"][comparison][metric] = {
                    "mean": float(values.mean()), "median": float(np.median(values)),
                    "ci95_mean": [lower, upper],
                    "candidate_better_samples": int((values > 0).sum()),
                    "reference_better_samples": int((values < 0).sum()),
                    "ties": int((values == 0).sum()),
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
