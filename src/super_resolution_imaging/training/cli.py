from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from .dataset import DEFAULT_EXCLUDED_IDS, ZemaxBurstDataset
from .losses import ssim_loss, global_brightness_loss, local_low_frequency_loss
from .model import BurstUNet2x, frequency_split


HISTORY_FIELDS = ("epoch", "learning_rate", "train_l1", "train_psnr_db", "validation_l1", "validation_psnr_db")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    name = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return float((-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    component_loss_weight: float = 0.0,
    ssim_loss_weight: float = 0.0,
    global_brightness_weight: float = 0.0,
    local_low_frequency_weight: float = 0.0,
    low_frequency_sigma: float = 4.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_l1 = total_psnr = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if training and getattr(model, "split_residual_output", False):
                prediction, components = model.forward_with_components(inputs)
                ideal_low, ideal_high = frequency_split(target - components["base"], model.split_sigma)
                reconstruction_loss = loss_function(prediction, target)
                reconstruction_objective = reconstruction_loss + ssim_loss_weight * ssim_loss(prediction, target)
                component_loss = loss_function(components["low"], ideal_low) + loss_function(components["high"], ideal_high)
                loss = reconstruction_objective + component_loss_weight * component_loss
            else:
                prediction = model(inputs)
                reconstruction_loss = loss_function(prediction, target)
                loss = reconstruction_loss + ssim_loss_weight * ssim_loss(prediction, target) if training else reconstruction_loss
            if training:
                if global_brightness_weight:
                    loss = loss + global_brightness_weight * global_brightness_loss(prediction, target)
                if local_low_frequency_weight:
                    loss = loss + local_low_frequency_weight * local_low_frequency_loss(
                        prediction, target, low_frequency_sigma,
                    )
                loss.backward()
                optimizer.step()
            size = inputs.shape[0]
            total_l1 += reconstruction_loss.item() * size
            total_psnr += _batch_psnr(prediction.detach(), target)
            count += size
    return total_l1 / count, total_psnr / count


def save_preview(path: Path, inputs: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> None:
    labels = ["input frame 0", "prediction", "target", "4x absolute error"]
    input_image = Image.fromarray(np.uint8(inputs[0].numpy() * 255)).resize((256, 256), Image.Resampling.BICUBIC)
    pred = np.clip(prediction[0].numpy(), 0, 1)
    truth = np.clip(target[0].numpy(), 0, 1)
    images = [input_image, Image.fromarray(np.uint8(pred * 255)), Image.fromarray(np.uint8(truth * 255)),
              Image.fromarray(np.uint8(np.clip(np.abs(pred - truth) * 4, 0, 1) * 255))]
    canvas = Image.new("L", (1024, 280), 255)
    for index, (image, label) in enumerate(zip(images, labels)):
        tile = Image.new("L", (256, 280), 255)
        tile.paste(image, (0, 24))
        ImageDraw.Draw(tile).text((5, 5), label, fill=0)
        canvas.paste(tile, (index * 256, 0))
    canvas.save(path)


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train and validate the Zemax burst super-resolution model.")
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--group", choices=("a", "b"), default="b")
    result.add_argument("--samples", "--train-samples", dest="train_samples", type=int)
    result.add_argument("--validation-samples", type=int)
    result.add_argument("--patch-size", type=int, help="Random LR training crop size; validation remains full resolution")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--base-channels", type=int, default=24)
    output_mode = result.add_mutually_exclusive_group()
    output_mode.add_argument("--residual-output", action="store_true", help="Predict a residual on top of bicubic B0")
    output_mode.add_argument(
        "--split-residual-output", action="store_true",
        help="Predict separately constrained low- and high-frequency residuals on top of bicubic B0",
    )
    result.add_argument("--split-sigma", type=float, default=4.0, help="HR-pixel Gaussian sigma for residual split")
    result.add_argument("--component-loss-weight", type=float, default=0.2)
    result.add_argument("--ssim-loss-weight", type=float, default=0.0, help="Weight for 1-SSIM in the reconstruction loss")
    result.add_argument("--global-brightness-weight", type=float, default=0.0,
                        help="Weight for per-image mean brightness error (disabled by default)")
    result.add_argument("--local-low-frequency-weight", type=float, default=0.0,
                        help="Weight for mean-centered low-frequency error (disabled by default; uses split-sigma)")
    result.add_argument("--scheduler-patience", type=int, default=5)
    result.add_argument("--scheduler-factor", type=float, default=0.5)
    result.add_argument("--early-stopping-patience", type=int, default=20)
    result.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    result.add_argument("--seed", type=int, default=5400)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDED_IDS))
    checkpoint_mode = result.add_mutually_exclusive_group()
    checkpoint_mode.add_argument("--resume", type=Path, help="Restore the complete state and continue an interrupted run")
    checkpoint_mode.add_argument(
        "--initialize-from", type=Path,
        help="Load only model weights from another run and start a new optimizer/history (for A to B fine-tuning)",
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    for name in ("global_brightness_weight", "local_low_frequency_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.local_low_frequency_weight and (not math.isfinite(args.split_sigma) or args.split_sigma <= 0):
        raise ValueError("split-sigma must be finite and positive for local low-frequency loss")
    seed_everything(args.seed)
    device = select_device(args.device)
    excluded = tuple(args.exclude)
    train_data = ZemaxBurstDataset(
        args.dataset, args.group, "train", args.train_samples, excluded, patch_size=args.patch_size
    )
    validation_data = ZemaxBurstDataset(args.dataset, args.group, "validation", args.validation_samples, excluded)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BurstUNet2x(
        args.base_channels, residual_output=args.residual_output,
        split_residual_output=args.split_residual_output, split_sigma=args.split_sigma,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience,
        min_lr=args.minimum_learning_rate,
    )
    loss_function = nn.L1Loss()
    output = args.output.resolve()
    start_epoch, best_epoch, best_psnr, epochs_without_improvement = 1, 0, -math.inf, 0
    history: list[dict[str, float | int]] = []

    configuration = {
        "dataset": str(args.dataset.resolve()), "group": args.group, "base_channels": args.base_channels,
        "dataset_manifest_sha256": sha256_file(args.dataset.resolve() / "manifest.jsonl"),
        "dataset_complete_sha256": sha256_file(args.dataset.resolve() / "COMPLETE.json"),
        "quality_policy_sha256": (
            sha256_file(args.dataset.resolve() / "quality_exclusions.json")
            if (args.dataset.resolve() / "quality_exclusions.json").is_file() else None
        ),
        "train_samples": len(train_data), "validation_samples": len(validation_data), "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed,
        "excluded_ids": list(excluded), "device": str(device), "patch_size": args.patch_size,
        "residual_output": args.residual_output,
        "split_residual_output": args.split_residual_output, "split_sigma": args.split_sigma,
        "component_loss_weight": args.component_loss_weight,
        "ssim_loss_weight": args.ssim_loss_weight,
        "global_brightness_weight": args.global_brightness_weight,
        "local_low_frequency_weight": args.local_low_frequency_weight,
        "initialized_from": None,
    }
    train_sample_ids = [record["sample_id"] for record in train_data.records]
    validation_sample_ids = [record["sample_id"] for record in validation_data.records]

    if args.resume:
        output.mkdir(parents=True, exist_ok=True)
        saved = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        saved_configuration = saved["configuration"]
        if saved_configuration["group"] != args.group or saved_configuration["base_channels"] != args.base_channels:
            raise ValueError("Resume checkpoint group/base_channels do not match the requested model")
        if bool(saved_configuration.get("residual_output", False)) != args.residual_output:
            raise ValueError("Resume checkpoint residual_output does not match the requested model")
        if bool(saved_configuration.get("split_residual_output", False)) != args.split_residual_output:
            raise ValueError("Resume checkpoint split_residual_output does not match the requested model")
        if saved_configuration.get("patch_size") != args.patch_size:
            raise ValueError("Resume checkpoint patch_size does not match the requested training crop")
        if float(saved_configuration.get("ssim_loss_weight", 0.0)) != args.ssim_loss_weight:
            raise ValueError("Resume checkpoint ssim_loss_weight does not match the requested loss")
        for name in ("global_brightness_weight", "local_low_frequency_weight"):
            if float(saved_configuration.get(name, 0.0)) != getattr(args, name):
                raise ValueError(f"Resume checkpoint {name} does not match the requested loss")
        if args.local_low_frequency_weight and float(saved_configuration.get("split_sigma", 4.0)) != args.split_sigma:
            raise ValueError("Resume checkpoint split_sigma does not match local low-frequency loss")
        if saved["train_sample_ids"] != train_sample_ids or saved["validation_sample_ids"] != validation_sample_ids:
            raise ValueError("Resume checkpoint uses different training or validation samples")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        generator.set_state(saved["data_loader_generator_state"])
        torch.set_rng_state(saved["torch_random_state"])
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_psnr = float(saved["best_validation_psnr_db"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
        configuration = dict(saved_configuration)
        configuration["device"] = str(device)
        configuration.setdefault("global_brightness_weight", 0.0)
        configuration.setdefault("local_low_frequency_weight", 0.0)
    else:
        output.mkdir(parents=True, exist_ok=False)
        if args.initialize_from:
            initialization_path = args.initialize_from.resolve()
            saved = torch.load(initialization_path, map_location=device, weights_only=False)
            source_configuration = saved.get("configuration", {})
            source_base_channels = source_configuration.get("base_channels")
            if source_base_channels is not None and source_base_channels != args.base_channels:
                raise ValueError("Initialization checkpoint base_channels do not match the requested model")
            if bool(source_configuration.get("residual_output", False)) != args.residual_output:
                raise ValueError("Initialization checkpoint residual_output does not match the requested model")
            if bool(source_configuration.get("split_residual_output", False)) != args.split_residual_output:
                raise ValueError("Initialization checkpoint split_residual_output does not match the requested model")
            model.load_state_dict(saved["model_state"])
            configuration["initialized_from"] = {
                "path": str(initialization_path),
                "sha256": sha256_file(initialization_path),
                "source_group": source_configuration.get("group"),
                "source_epoch": saved.get("epoch"),
                "source_best_epoch": saved.get("best_epoch"),
            }

    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint is already at epoch {start_epoch - 1}; --epochs must be larger")
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_l1, train_psnr = run_epoch(
            model, train_loader, loss_function, device, optimizer,
            args.component_loss_weight, args.ssim_loss_weight,
            args.global_brightness_weight, args.local_low_frequency_weight, args.split_sigma,
        )
        validation_l1, validation_psnr = run_epoch(model, validation_loader, loss_function, device)
        scheduler.step(validation_l1)
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch, "learning_rate": learning_rate,
            "train_l1": train_l1, "train_psnr_db": train_psnr,
            "validation_l1": validation_l1, "validation_psnr_db": validation_psnr,
        }
        history.append(row)
        improved = validation_psnr > best_psnr
        if improved:
            best_psnr, best_epoch, epochs_without_improvement = validation_psnr, epoch, 0
        else:
            epochs_without_improvement += 1
        state = {
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "configuration": configuration,
            "train_sample_ids": train_sample_ids, "validation_sample_ids": validation_sample_ids,
            "data_loader_generator_state": generator.get_state(), "torch_random_state": torch.get_rng_state(),
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "history": history, "best_epoch": best_epoch, "best_validation_psnr_db": best_psnr,
            "epochs_without_improvement": epochs_without_improvement,
        }
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        write_history(output / "history.csv", history)
        print(
            f"epoch={epoch:04d} lr={learning_rate:.2e} train_l1={train_l1:.6f} "
            f"train_psnr={train_psnr:.3f} val_l1={validation_l1:.6f} val_psnr={validation_psnr:.3f}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    example = validation_data[0]
    model.eval()
    with torch.inference_mode():
        prediction = model(example["input"].unsqueeze(0).to(device)).cpu()[0]
    save_preview(output / "best_validation_preview.png", example["input"], prediction, example["target"])
    summary = {
        **configuration, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "completed_epoch": history[-1]["epoch"], "best_epoch": best_epoch,
        "best_validation_psnr_db": best_psnr,
        "best_validation_l1": next(row["validation_l1"] for row in history if row["epoch"] == best_epoch),
        "final_learning_rate": history[-1]["learning_rate"], "stopped_early": stopped_early,
        "checkpoint_reload": "passed",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
