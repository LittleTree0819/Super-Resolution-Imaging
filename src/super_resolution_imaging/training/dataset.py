from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Group = Literal["a", "b"]

# Dataset-specific quality exclusions. Raw samples remain untouched for auditability.
DEFAULT_EXCLUDED_IDS = (
    "0552_00",  # original low-information sample
    "0036_01",  # expansion: near-blank target (std 0.00852)
    "0449_00", "0449_01", "0449_02", "0449_03",  # source-level photometric outlier
)


def _load_grayscale(path: Path, expected_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "L" or image.size != expected_size:
            raise ValueError(f"Expected L image of size {expected_size} at {path}, got {image.mode} {image.size}")
        values = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).unsqueeze(0)


class ZemaxBurstDataset(Dataset):
    """Load one four-frame Zemax burst and its ideal 2x image."""

    def __init__(
        self,
        root: str | Path,
        group: Group,
        split: str = "train",
        limit: int | None = None,
        exclude_ids: tuple[str, ...] = (),
        patch_size: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.group = group.lower()  # type: ignore[assignment]
        if patch_size is not None and not 1 <= patch_size <= 128:
            raise ValueError("patch_size must be between 1 and 128")
        self.patch_size = patch_size
        if self.group not in ("a", "b"):
            raise ValueError("group must be 'a' or 'b'")
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        excluded = set(exclude_ids)
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.records = [r for r in records if r["split"] == split and r["sample_id"] not in excluded]
        self.records.sort(key=lambda record: record["sample_id"])
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"No samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        sample = self.root / "samples" / record["sample_id"]
        frames = [
            _load_grayscale(sample / self.group / f"frame_{frame:02d}.png", (128, 128))
            for frame in range(4)
        ]
        target = _load_grayscale(sample / "target.png", (256, 256))
        inputs = torch.cat(frames, dim=0)
        result: dict[str, torch.Tensor | str] = {
            "input": inputs,
            "target": target,
            "sample_id": record["sample_id"],
        }
        if self.patch_size is not None:
            maximum = 128 - self.patch_size
            top = int(torch.randint(maximum + 1, (1,)).item())
            left = int(torch.randint(maximum + 1, (1,)).item())
            hr_top, hr_left, hr_size = 2 * top, 2 * left, 2 * self.patch_size
            result["input"] = inputs[:, top:top + self.patch_size, left:left + self.patch_size]
            result["target"] = target[:, hr_top:hr_top + hr_size, hr_left:hr_left + hr_size]
            result["crop_xy"] = torch.tensor((left, top), dtype=torch.int64)
        return result
