"""Numerical and visual sanity checks for the generated Zemax dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.ravel() - left.mean()
    right = right.ravel() - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def percentiles(values: list[float]) -> dict[str, float]:
    points = np.percentile(values, [0, 1, 5, 50, 95, 99, 100])
    return {key: round(float(value), 7) for key, value in zip(
        ("min", "p01", "p05", "median", "p95", "p99", "max"), points
    )}


def labelled(image: Image.Image, label: str, size: int = 256) -> Image.Image:
    image = ImageOps.fit(image.convert("L"), (size, size), method=Image.Resampling.LANCZOS)
    tile = Image.new("L", (size, size + 24), 255)
    tile.paste(image, (0, 24))
    ImageDraw.Draw(tile).text((5, 5), label, fill=0)
    return tile


def montage(records: list[dict], root: Path, destination: Path) -> None:
    columns, tile, header = 7, 256, 24
    canvas = Image.new("L", (columns * tile, len(records) * (tile + header)), 255)
    for row, record in enumerate(records):
        sample = root / "samples" / record["sample_id"]
        paths = [sample / "source.png", sample / "target.png"]
        paths += [sample / "b" / f"frame_{index:02d}.png" for index in range(4)]
        b0, b3 = array(paths[2]), array(paths[5])
        difference = Image.fromarray(np.uint8(np.clip(np.abs(b3 - b0) * 10.0, 0, 1) * 255))
        images = [Image.open(path) for path in paths] + [difference]
        labels = [f"{record['sample_id']} source", "target", "B0 (0,0)", "B1 (+x)",
                  "B2 (+y)", "B3 (+x,+y)", "10x |B3-B0|"]
        for column, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(labelled(image, label), (column * tile, row * (tile + header)))
            image.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    values: dict[str, list[float]] = defaultdict(list)
    anomalies: list[dict] = []
    split_counts = Counter(record["split"] for record in records)
    for record in records:
        sample = root / "samples" / record["sample_id"]
        target = array(sample / "target.png")
        a = [array(sample / "a" / f"frame_{index:02d}.png") for index in range(4)]
        b = [array(sample / "b" / f"frame_{index:02d}.png") for index in range(4)]
        target_lr = np.asarray(Image.fromarray(np.uint8(target * 255)).resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        values["target_mean"].append(float(target.mean()))
        values["target_std"].append(float(target.std()))
        values["target_black_fraction"].append(float(np.mean(target == 0)))
        values["target_white_fraction"].append(float(np.mean(target == 1)))
        values["target_to_b0_correlation"].append(correlation(target_lr, b[0]))
        values["a_max_difference"].append(max(float(np.max(np.abs(frame - a[0]))) for frame in a[1:]))
        for index in range(1, 4):
            values[f"b{index}_to_b0_mae"].append(float(np.mean(np.abs(b[index] - b[0]))))
            values[f"b{index}_to_b0_correlation"].append(correlation(b[index], b[0]))
        if target.std() < 0.01 or b[0].std() < 0.01:
            anomalies.append({"sample_id": record["sample_id"], "reason": "near_blank"})
        if correlation(target_lr, b[0]) < 0.5:
            anomalies.append({"sample_id": record["sample_id"], "reason": "weak_target_input_alignment"})

    chosen: list[dict] = []
    for split in ("train", "validation", "test"):
        subset = [record for record in records if record["split"] == split]
        indices = np.linspace(0, len(subset) - 1, 4, dtype=int)
        chosen.extend(subset[index] for index in indices)
    output.mkdir(parents=True, exist_ok=True)
    montage(chosen, root, output / "montage.png")
    report = {
        "dataset": str(root),
        "sample_count": len(records),
        "split_counts": dict(split_counts),
        "metrics": {name: percentiles(series) for name, series in values.items()},
        "flagged_count": len(anomalies),
        "flagged_samples": anomalies,
        "montage_samples": [record["sample_id"] for record in chosen],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
