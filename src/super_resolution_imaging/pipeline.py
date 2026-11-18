from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from PIL import Image

from .config import Config
from .data.sources import content_sha256, discover_images, prepare_source, split_for_hash
from .data.writer import DatasetWriter, sha256_file, write_json
from .zos.image_simulation import run_image_simulation
from .zos.model import file_sha256, prepare_shift_model, set_lens_shift


def _verify_size(path: Path, pixels: int) -> None:
    with Image.open(path) as image:
        if image.size != (pixels, pixels):
            raise RuntimeError(f"Unexpected image size for {path}: {image.size}, expected {(pixels, pixels)}")
        if image.mode != "L":
            raise RuntimeError(f"Unexpected image mode for {path}: {image.mode}, expected single-channel L")


def _normalize_grayscale(path: Path) -> None:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.save(path, format="PNG")


def generate(
    session: object,
    config: Config,
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    patches: int,
    *,
    start: int = 0,
    resume: bool = False,
    only_split: str | None = None,
) -> int:
    sources = discover_images(input_dir.resolve())
    if start < 0:
        raise ValueError("start must be non-negative")
    sources = sources[start:]
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        raise ValueError(f"No supported images found in {input_dir}")
    if patches < 1:
        raise ValueError("patches must be positive")
    if only_split not in (None, "train", "validation", "test"):
        raise ValueError("only_split must be train, validation, test, or None")
    if only_split is not None:
        sources = [
            source for source in sources
            if split_for_hash(
                content_sha256(source), config.dataset.split_train, config.dataset.split_validation,
            ) == only_split
        ]
        if not sources:
            raise ValueError(f"No source images belong to split={only_split!r}")
    if not config.zemax.images_dir.is_dir():
        raise FileNotFoundError(f"Zemax images_dir does not exist: {config.zemax.images_dir}")

    writer = DatasetWriter(output_dir, resume=resume)
    prepared_model = writer.root / "prepared_model.zmx"
    config_snapshot = json_safe(asdict(config))
    dataset_path = writer.root / "dataset.json"
    if writer.is_new:
        prepare_shift_model(session, config, prepared_model)
        write_json(dataset_path, {"config": config_snapshot, "prepared_model_sha256": file_sha256(prepared_model)})
    else:
        if not dataset_path.is_file() or not prepared_model.is_file():
            raise ValueError("Resume directory is missing dataset.json or prepared_model.zmx")
        existing = json.loads(dataset_path.read_text(encoding="utf-8"))
        if existing.get("config") != config_snapshot:
            raise ValueError("Resume configuration does not match dataset.json")
        if existing.get("prepared_model_sha256") != file_sha256(prepared_model):
            raise ValueError("Prepared model hash changed since the run started")
        session.system.LoadFile(str(prepared_model), False)
    completed = writer.completed_ids()
    count = len(completed)
    (writer.root / "COMPLETE.json").unlink(missing_ok=True)
    for source in sources:
        for patch_index in range(patches):
            sample_id = f"{source.stem}_{patch_index:02d}"
            if sample_id in completed:
                print(f"skip completed sample {sample_id}", flush=True)
                continue
            final_sample = writer.root / "samples" / sample_id
            sample = writer.root / "samples" / f".{sample_id}.partial"
            if final_sample.exists():
                raise ValueError(f"Completed sample directory is missing from manifest: {final_sample}")
            if sample.exists():
                shutil.rmtree(sample)
            sample.mkdir(parents=True)
            source_meta = prepare_source(source, sample / "source.png", config.dataset.crop_size, config.dataset.seed, patch_index)
            split = split_for_hash(str(source_meta["source_sha256"]), config.dataset.split_train, config.dataset.split_validation)
            stage_name = f"sr_{source_meta['source_sha256']}_{patch_index:02d}.png"
            staged = config.zemax.images_dir / stage_name
            shutil.copy2(sample / "source.png", staged)
            try:
                set_lens_shift(session, config, 0.0, 0.0)
                target = sample / "target.png"
                run_image_simulation(session, stage_name, target, config.simulation, ideal=True)
                _normalize_grayscale(target)
                _verify_size(target, config.simulation.hr_pixels)
                outputs: dict[str, list[str]] = {"a": [], "b": []}
                set_lens_shift(session, config, 0.0, 0.0)
                a0 = sample / "a" / "frame_00.png"
                run_image_simulation(session, stage_name, a0, config.simulation)
                _normalize_grayscale(a0)
                _verify_size(a0, config.simulation.lr_pixels)
                for frame_index in range(config.dataset.frames):
                    frame = sample / "a" / f"frame_{frame_index:02d}.png"
                    if frame_index:
                        frame.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(a0, frame)
                    outputs["a"].append(sha256_file(frame))

                for frame_index, (dx, dy) in enumerate(config.dataset.positions_mm):
                    frame = sample / "b" / f"frame_{frame_index:02d}.png"
                    if dx == 0.0 and dy == 0.0:
                        frame.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(a0, frame)
                    else:
                        set_lens_shift(session, config, dx, dy)
                        run_image_simulation(session, stage_name, frame, config.simulation)
                        _normalize_grayscale(frame)
                    _verify_size(frame, config.simulation.lr_pixels)
                    outputs["b"].append(sha256_file(frame))
            finally:
                staged.unlink(missing_ok=True)
            metadata = {
                "sample_id": sample_id, "split": split, **source_meta,
                "input_shape": [config.dataset.frames, config.simulation.lr_pixels, config.simulation.lr_pixels],
                "target_shape": [1, config.simulation.hr_pixels, config.simulation.hr_pixels],
                "image_mode": "L",
                "positions_a_mm": [[0.0, 0.0]] * config.dataset.frames,
                "positions_b_mm": [list(xy) for xy in config.dataset.positions_mm],
                "target_sha256": sha256_file(target), "output_sha256": outputs,
            }
            write_json(sample / "metadata.json", metadata)
            sample.rename(final_sample)
            writer.append(metadata)
            count += 1
            print(f"completed {count}: {sample_id}", flush=True)
    writer.complete(count)
    return count


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
