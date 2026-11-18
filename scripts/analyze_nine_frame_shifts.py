"""Fit signed subpixel translations and photometric changes in a nine-frame dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


TARGET_SHIFTS = ((0, 0), (.5, 0), (.5, .5), (0, .5), (-.5, .5), (-.5, 0), (-.5, -.5), (0, -.5), (.5, -.5))


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def fit_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit translation plus gain/offset using a deterministic coarse-to-fine search.

    ``grid_sample`` uses bicubic interpolation.  The three search levels resolve
    the shift to 0.0005 LR pixels without requiring SciPy in the acquisition
    environment.
    """
    ref = torch.from_numpy(reference).to(dtype=torch.float32)[None, None]
    target = torch.from_numpy(moving[8:-8, 8:-8]).to(dtype=torch.float32)
    height, width = reference.shape
    ys = torch.linspace(-1, 1, height)
    xs = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)

    def evaluate(candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        best_mse = torch.tensor(float("inf"))
        best_xy = candidates[0]
        best_photo = torch.zeros(2)
        for start in range(0, len(candidates), 256):
            xy = candidates[start:start + 256]
            grids = base_grid[None].repeat(len(xy), 1, 1, 1)
            # To move content by +dx/+dy, output pixels sample input at x-dx/y-dy.
            grids[..., 0] -= 2 * xy[:, 0, None, None] / (width - 1)
            grids[..., 1] -= 2 * xy[:, 1, None, None] / (height - 1)
            shifted = F.grid_sample(
                ref.expand(len(xy), -1, -1, -1), grids, mode="bicubic",
                padding_mode="reflection", align_corners=True,
            )[:, 0, 8:-8, 8:-8]
            pred_mean = shifted.mean(dim=(1, 2))
            target_mean = target.mean()
            centered = shifted - pred_mean[:, None, None]
            target_centered = target - target_mean
            gain = (centered * target_centered).mean(dim=(1, 2)) / centered.square().mean(dim=(1, 2)).clamp_min(1e-12)
            offset = target_mean - gain * pred_mean
            mse = (gain[:, None, None] * shifted + offset[:, None, None] - target).square().mean(dim=(1, 2))
            index = int(torch.argmin(mse))
            if mse[index] < best_mse:
                best_mse = mse[index]
                best_xy = xy[index]
                best_photo = torch.stack((gain[index], offset[index]))
        return best_xy, best_photo, best_mse

    center = torch.zeros(2)
    photo = torch.tensor((1., 0.))
    mse = torch.tensor(0.)
    for radius, step in ((1.0, .05), (.05, .005), (.005, .0005)):
        axis = torch.arange(-radius, radius + step / 2, step)
        offsets = torch.cartesian_prod(axis, axis)
        center, photo, mse = evaluate(center[None] + offsets)
    return float(center[0]), float(center[1]), float(photo[0]), float(photo[1]), float(mse)


def percentile(values: list[float]) -> dict[str, float]:
    p = np.percentile(values, [0, 5, 50, 95, 100])
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, p)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    rows = []
    for record in records:
        sample = root / "samples" / record["sample_id"] / "b"
        frames = [load(sample / f"frame_{index:02d}.png") for index in range(9)]
        for index, frame in enumerate(frames):
            dx, dy, gain, offset, mse = (0., 0., 1., 0., 0.) if index == 0 else fit_shift(frames[0], frame)
            target_dx, target_dy = TARGET_SHIFTS[index]
            rows.append({
                "sample_id": record["sample_id"], "frame": index,
                "target_shift_xy": [target_dx, target_dy], "measured_shift_xy": [dx, dy],
                "shift_error_xy": [dx - target_dx, dy - target_dy],
                "shift_error_norm": float(np.hypot(dx - target_dx, dy - target_dy)),
                "mean_delta": float(frame.mean() / 255 - frames[0].mean() / 255),
                "gain_after_registration": gain, "offset_8bit": offset, "registered_mse_8bit": mse,
            })
    by_frame = {}
    for index in range(9):
        selected = [row for row in rows if row["frame"] == index]
        by_frame[str(index)] = {
            "target_shift_xy": list(TARGET_SHIFTS[index]),
            "median_measured_shift_xy": [float(np.median([row["measured_shift_xy"][axis] for row in selected])) for axis in (0, 1)],
            "shift_error_norm": percentile([row["shift_error_norm"] for row in selected]),
            "absolute_mean_delta": percentile([abs(row["mean_delta"]) for row in selected]),
            "gain_after_registration": percentile([row["gain_after_registration"] for row in selected]),
        }
    report = {"samples": len(records), "target_image_shifts_lr_px": TARGET_SHIFTS, "frames": by_frame, "rows": rows}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"samples": len(records), "frames": by_frame}, indent=2))


if __name__ == "__main__":
    main()
