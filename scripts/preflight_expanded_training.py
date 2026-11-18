"""Preflight checks for the expanded DIV2K two-stage training run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn

from super_resolution_imaging.training.cli import select_device
from super_resolution_imaging.training.dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from super_resolution_imaging.training.losses import ssim_loss
from super_resolution_imaging.training.model import BurstUNet2x, frequency_split


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--component-loss-weight", type=float, default=0.2)
    parser.add_argument("--ssim-loss-weight", type=float, default=0.2)
    parser.add_argument("--split-sigma", type=float, default=4.0)
    args = parser.parse_args()

    root = args.dataset.resolve()
    complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    policy = json.loads((root / "quality_exclusions.json").read_text(encoding="utf-8"))
    if complete["samples"] != 2744:
        raise ValueError(f"Expected 2744 complete samples, got {complete['samples']}")
    if tuple(policy["excluded_sample_ids"]) != DEFAULT_EXCLUDED_IDS:
        raise ValueError("quality_exclusions.json does not match DEFAULT_EXCLUDED_IDS")

    datasets = {
        f"{group}_{split}": ZemaxBurstDataset(root, group, split, exclude_ids=DEFAULT_EXCLUDED_IDS)
        for group in ("a", "b") for split in ("train", "validation", "test")
    }
    for key in ("a_train", "b_train"):
        if len(datasets[key]) != 2586:
            raise ValueError(f"{key} expected 2586 samples, got {len(datasets[key])}")
    for group in ("a", "b"):
        if len(datasets[f"{group}_validation"]) != 72 or len(datasets[f"{group}_test"]) != 80:
            raise ValueError(f"Frozen evaluation split count changed for group {group}")

    device = select_device(args.device)
    item = datasets["b_train"][0]
    inputs = item["input"].unsqueeze(0).to(device)
    target = item["target"].unsqueeze(0).to(device)
    if tuple(inputs.shape) != (1, 4, 128, 128) or tuple(target.shape) != (1, 1, 256, 256):
        raise ValueError(f"Unexpected tensor shapes: input={tuple(inputs.shape)}, target={tuple(target.shape)}")
    model = BurstUNet2x(
        args.base_channels, split_residual_output=True, split_sigma=args.split_sigma,
    ).to(device)
    prediction, components = model.forward_with_components(inputs)
    ideal_low, ideal_high = frequency_split(target - components["base"], args.split_sigma)
    l1 = nn.functional.l1_loss(prediction, target)
    component = nn.functional.l1_loss(components["low"], ideal_low) + nn.functional.l1_loss(components["high"], ideal_high)
    loss = l1 + args.ssim_loss_weight * ssim_loss(prediction, target) + args.component_loss_weight * component
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Preflight loss is not finite")

    report = {
        "status": "passed", "training_started": False, "dataset": str(root),
        "dataset_samples": complete["samples"], "train_samples": 2586,
        "validation_samples": 72, "test_samples": 80,
        "excluded_ids": list(DEFAULT_EXCLUDED_IDS),
        "input_shape": list(inputs.shape), "target_shape": list(target.shape),
        "patch_size": None, "device": str(device), "cuda_available": torch.cuda.is_available(),
        "model": "BurstUNet2x split residual", "base_channels": args.base_channels,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "component_loss_weight": args.component_loss_weight,
        "ssim_loss_weight": args.ssim_loss_weight, "split_sigma": args.split_sigma,
        "smoke_loss": float(loss.detach().cpu()),
        "hashes": {
            "manifest": digest(root / "manifest.jsonl"),
            "complete": digest(root / "COMPLETE.json"),
            "quality_policy": digest(root / "quality_exclusions.json"),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
