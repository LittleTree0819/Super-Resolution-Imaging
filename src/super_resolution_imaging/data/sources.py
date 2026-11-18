from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import random

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def content_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_hash(digest: str, train: int, validation: int) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def crop_box(width: int, height: int, size: int, seed: int, key: str, index: int) -> tuple[int, int, int, int]:
    if width < size or height < size:
        raise ValueError(
            f"Preprocessing invariant failed: prepared image {width}x{height} is smaller than crop size {size}"
        )
    rng = random.Random(f"{seed}:{key}:{index}")
    left = rng.randint(0, width - size)
    top = rng.randint(0, height - size)
    return left, top, left + size, top + size


def prepare_source(source: Path, destination: Path, size: int, seed: int, index: int) -> dict[str, object]:
    digest = content_sha256(source)
    with Image.open(source) as image:
        gray = image.convert("L")
        original_size = gray.size
        scale = max(1.0, size / min(gray.size))
        if scale > 1.0:
            resized_size = (math.ceil(gray.width * scale), math.ceil(gray.height * scale))
            gray = gray.resize(resized_size, Image.Resampling.LANCZOS)
        box = crop_box(gray.width, gray.height, size, seed, digest, index)
        crop = gray.crop(box)
        destination.parent.mkdir(parents=True, exist_ok=True)
        crop.save(destination, format="PNG")
    return {
        "source": str(source.resolve()),
        "source_sha256": digest,
        "original_size": list(original_size),
        "prepared_size": [gray.width, gray.height],
        "pre_crop_scale": scale,
        "crop_box": list(box),
    }
