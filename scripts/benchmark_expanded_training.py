"""Measure CPU epoch time for the expanded two-stage training protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from super_resolution_imaging.training.cli import run_epoch, seed_everything
from super_resolution_imaging.training.dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from super_resolution_imaging.training.model import BurstUNet2x


def benchmark_group(root: Path, group: str, samples: int, batch_size: int) -> dict[str, float | int | str]:
    train = ZemaxBurstDataset(root, group, "train", samples, DEFAULT_EXCLUDED_IDS)
    validation = ZemaxBurstDataset(root, group, "validation", exclude_ids=DEFAULT_EXCLUDED_IDS)
    generator = torch.Generator().manual_seed(5400)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    model = BurstUNet2x(24, split_residual_output=True, split_sigma=4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if group == "a" else 1e-4, weight_decay=1e-5)
    loss = nn.L1Loss()

    # Warm up kernels and allocation outside the measured interval.
    item = train[0]
    model(item["input"].unsqueeze(0))

    started = time.perf_counter()
    train_l1, train_psnr = run_epoch(model, train_loader, loss, torch.device("cpu"), optimizer, 0.2, 0.2)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    validation_l1, validation_psnr = run_epoch(model, validation_loader, loss, torch.device("cpu"))
    validation_seconds = time.perf_counter() - started
    measured_steps = math.ceil(len(train) / batch_size)
    formal_steps = math.ceil(2586 / batch_size)
    estimated_train_seconds = train_seconds * formal_steps / measured_steps
    estimated_epoch_seconds = estimated_train_seconds + validation_seconds
    return {
        "group": group, "measured_train_samples": len(train), "validation_samples": len(validation),
        "batch_size": batch_size, "measured_train_steps": measured_steps, "formal_train_steps": formal_steps,
        "measured_train_seconds": train_seconds, "measured_validation_seconds": validation_seconds,
        "estimated_formal_train_seconds": estimated_train_seconds,
        "estimated_formal_epoch_seconds": estimated_epoch_seconds,
        "train_l1": train_l1, "train_psnr_db": train_psnr,
        "validation_l1": validation_l1, "validation_psnr_db": validation_psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    seed_everything(5400)
    root = args.dataset.resolve()
    groups = [benchmark_group(root, group, args.train_samples, args.batch_size) for group in ("a", "b")]
    per_epoch = {str(row["group"]): float(row["estimated_formal_epoch_seconds"]) for row in groups}
    estimates = {
        "old_early_stop_pattern_seconds": per_epoch["a"] * 72 + per_epoch["b"] * 39,
        "nominal_100_plus_50_seconds": per_epoch["a"] * 100 + per_epoch["b"] * 50,
        "maximum_200_plus_100_seconds": per_epoch["a"] * 200 + per_epoch["b"] * 100,
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "device": "cpu",
        "formal_training_samples": 2586, "formal_validation_samples": 72,
        "groups": groups, "estimates": estimates,
        "notes": [
            "Old early-stop pattern uses 72 A epochs and 39 B epochs from the prior lambda=0.2 run.",
            "Wall-clock estimates exclude interruptions and include measured full validation time.",
            "Allow additional margin for thermal throttling and background applications.",
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
