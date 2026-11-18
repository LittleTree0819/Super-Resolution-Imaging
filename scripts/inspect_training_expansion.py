"""Integrity and visual checks for newly added DIV2K training patches."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_gray(path: Path, expected_size: tuple[int, int], errors: list[str]) -> np.ndarray:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return np.zeros(expected_size[::-1], dtype=np.float32)
    try:
        with Image.open(path) as image:
            if image.size != expected_size:
                errors.append(f"wrong size {image.size}: {path}")
            if image.mode != "L":
                errors.append(f"wrong mode {image.mode}: {path}")
            return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    except Exception as exc:
        errors.append(f"unreadable {path}: {exc}")
        return np.zeros(expected_size[::-1], dtype=np.float32)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x, y = left.ravel() - left.mean(), right.ravel() - right.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def summary(values: list[float]) -> dict[str, float]:
    points = np.percentile(values, [0, 1, 5, 50, 95, 99, 100])
    return dict(zip(("min", "p01", "p05", "median", "p95", "p99", "max"), map(float, points)))


def crop_iou(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    area = (left[2] - left[0]) * (left[3] - left[1])
    return intersection / (2 * area - intersection)


def labelled(image: Image.Image, label: str, size: int = 256) -> Image.Image:
    tile = Image.new("L", (size, size + 24), 255)
    tile.paste(ImageOps.fit(image.convert("L"), (size, size), method=Image.Resampling.LANCZOS), (0, 24))
    ImageDraw.Draw(tile).text((5, 5), label, fill=0)
    return tile


def save_montage(root: Path, rows: list[dict[str, object]], output: Path) -> None:
    canvas = Image.new("L", (7 * 256, len(rows) * 280), 255)
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        sample = root / "samples" / sample_id
        paths = [sample / "source.png", sample / "target.png"] + [
            sample / "b" / f"frame_{index:02d}.png" for index in range(4)
        ]
        b0 = np.asarray(Image.open(paths[2]).convert("L"), dtype=np.float32)
        b3 = np.asarray(Image.open(paths[5]).convert("L"), dtype=np.float32)
        difference = Image.fromarray(np.uint8(np.clip(np.abs(b3 - b0) * 10, 0, 255)))
        images = [Image.open(path) for path in paths] + [difference]
        labels = [f"{sample_id} source", "target", "B0", "B1 +x", "B2 +y", "B3 +xy", "10x |B3-B0|"]
        for column, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(labelled(image, label), (column * 256, row_index * 280))
            image.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    added = [row for row in records if row["split"] == "train" and int(str(row["sample_id"]).rsplit("_", 1)[1]) in (1, 2, 3)]
    errors: list[str] = []
    flags: list[dict[str, str]] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    per_sample: list[dict[str, object]] = []
    crop_boxes: dict[str, list[tuple[int, ...]]] = defaultdict(list)

    for record in added:
        sample_id = str(record["sample_id"])
        sample = root / "samples" / sample_id
        metadata_path = sample / "metadata.json"
        if not metadata_path.is_file():
            errors.append(f"missing: {metadata_path}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != record:
                errors.append(f"metadata differs from manifest: {sample_id}")
        source = read_gray(sample / "source.png", (768, 768), errors)
        target_path = sample / "target.png"
        target = read_gray(target_path, (256, 256), errors)
        a_paths = [sample / "a" / f"frame_{index:02d}.png" for index in range(4)]
        b_paths = [sample / "b" / f"frame_{index:02d}.png" for index in range(4)]
        a = [read_gray(path, (128, 128), errors) for path in a_paths]
        b = [read_gray(path, (128, 128), errors) for path in b_paths]
        if target_path.is_file() and file_sha256(target_path) != record["target_sha256"]:
            errors.append(f"target hash mismatch: {sample_id}")
        for group, paths in (("a", a_paths), ("b", b_paths)):
            for index, path in enumerate(paths):
                if path.is_file() and file_sha256(path) != record["output_sha256"][group][index]:
                    errors.append(f"{group}{index} hash mismatch: {sample_id}")

        target_lr = np.asarray(Image.fromarray(np.uint8(target * 255)).resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        values = {
            "source_std": float(source.std()), "target_mean": float(target.mean()),
            "target_std": float(target.std()), "target_black_fraction": float(np.mean(target == 0)),
            "target_white_fraction": float(np.mean(target == 1)),
            "target_b0_correlation": correlation(target_lr, b[0]),
            "a_max_difference": max(float(np.max(np.abs(frame - a[0]))) for frame in a[1:]),
            "b1_b0_mae": float(np.mean(np.abs(b[1] - b[0]))),
            "b2_b0_mae": float(np.mean(np.abs(b[2] - b[0]))),
            "b3_b0_mae": float(np.mean(np.abs(b[3] - b[0]))),
        }
        for name, value in values.items():
            metrics[name].append(value)
        if values["target_std"] < 0.01 or float(b[0].std()) < 0.01:
            flags.append({"sample_id": sample_id, "reason": "near_blank"})
        if values["target_b0_correlation"] < 0.5:
            flags.append({"sample_id": sample_id, "reason": "weak_target_input_alignment"})
        if values["a_max_difference"] != 0:
            flags.append({"sample_id": sample_id, "reason": "A_frames_not_identical"})
        source_key = str(record["source_sha256"])
        crop_boxes[source_key].append(tuple(record["crop_box"]))
        per_sample.append({"sample_id": sample_id, **values})

    duplicate_crop_sources = [key for key, boxes in crop_boxes.items() if len(boxes) != len(set(boxes))]
    if duplicate_crop_sources:
        errors.extend(f"duplicate new crop box for source: {key}" for key in duplicate_crop_sources)
    ordered = sorted(per_sample, key=lambda row: float(row["target_std"]))
    indices = sorted(set(np.linspace(0, len(ordered) - 1, 10, dtype=int)))
    selected_ids = {str(ordered[index]["sample_id"]) for index in indices}
    selected = [record for record in added if str(record["sample_id"]) in selected_ids]
    all_train_boxes: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            all_train_boxes[str(record["source_sha256"])].append(tuple(record["crop_box"]))
    crop_ious = [
        crop_iou(boxes[left], boxes[right])
        for boxes in all_train_boxes.values()
        for left in range(len(boxes))
        for right in range(left + 1, len(boxes))
    ]

    output.mkdir(parents=True, exist_ok=False)
    save_montage(root, selected, output / "new_samples_montage.png")
    flagged_ids = {item["sample_id"] for item in flags}
    flagged_sources = {str(record["source_sha256"]) for record in records if record["sample_id"] in flagged_ids}
    flagged_context = [record for record in records if str(record["source_sha256"]) in flagged_sources]
    if flagged_context:
        save_montage(root, flagged_context, output / "flagged_context_montage.png")
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    report = {
        "dataset": str(root), "scope": "training patches 01-03", "samples_checked": len(added),
        "expected_samples": 1944, "integrity_error_count": len(errors), "integrity_errors": errors[:100],
        "png_files_checked": len(added) * 10, "metadata_files_checked": len(added),
        "sha256_values_checked": len(added) * 9,
        "flagged_count": len(flags), "flagged_samples": flags,
        "duplicate_new_crop_source_count": len(duplicate_crop_sources),
        "same_source_crop_iou": summary(crop_ious),
        "metrics": {name: summary(values) for name, values in metrics.items()},
        "montage_samples": [str(record["sample_id"]) for record in selected],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if len(added) != 1944 or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
