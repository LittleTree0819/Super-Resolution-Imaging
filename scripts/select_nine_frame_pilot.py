"""Select 32 texture-stratified existing crops for the nine-frame pilot."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


EXCLUDED_SOURCES = {"0036", "0449", "0552"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = []
    for record in records:
        sample_id = str(record["sample_id"])
        if record["split"] != "train" or sample_id.split("_", 1)[0] in EXCLUDED_SOURCES:
            continue
        target_path = root / "samples" / sample_id / "target.png"
        with Image.open(target_path) as image:
            target_std = float(np.asarray(image, dtype=np.float32).std() / 255.0)
        candidates.append((target_std, sample_id, record))
    candidates.sort()
    indices = np.linspace(0, len(candidates) - 1, args.count, dtype=int)
    chosen = [candidates[index] for index in indices]
    if len({sample_id.split("_", 1)[0] for _, sample_id, _ in chosen}) != len(chosen):
        raise RuntimeError("Pilot selection contains repeated DIV2K source images")
    output.mkdir(parents=True)
    manifest = []
    for rank, (target_std, sample_id, record) in enumerate(chosen):
        source = root / "samples" / sample_id / "source.png"
        destination = output / f"{sample_id}.png"
        shutil.copy2(source, destination)
        manifest.append({
            "rank": rank, "pilot_source": destination.name, "original_sample_id": sample_id,
            "original_source": record["source"], "crop_box": record["crop_box"],
            "target_std": target_std, "source_png_sha256": sha256(destination.read_bytes()).hexdigest(),
        })
    (output / "selection.json").write_text(json.dumps({"count": len(manifest), "samples": manifest}, indent=2) + "\n")
    print(json.dumps({"count": len(manifest), "min_target_std": chosen[0][0], "max_target_std": chosen[-1][0]}, indent=2))


if __name__ == "__main__":
    main()
