"""Resumable parallel downloader for the official DIV2K training archive."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import time
import urllib.request


URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
TOTAL = 3_530_603_713
PARTS = 12
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "DIV2K_train_HR.zip"
PART_DIR = ROOT / "data" / ".DIV2K_train_HR.parts"


def bounds(index: int) -> tuple[int, int]:
    size = (TOTAL + PARTS - 1) // PARTS
    start = index * size
    return start, min(TOTAL - 1, start + size - 1)


def download_part(index: int) -> None:
    start, end = bounds(index)
    path = PART_DIR / f"part-{index:02d}"
    present = path.stat().st_size if path.exists() else 0
    expected = end - start + 1
    if present > expected:
        raise RuntimeError(f"oversized partial file: {path}")
    if present == expected:
        return
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start + present}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"server did not honor Range for part {index}: HTTP {response.status}")
        with path.open("ab") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)


def downloaded() -> int:
    return sum(p.stat().st_size for p in PART_DIR.glob("part-*") if p.is_file())


def main() -> None:
    PART_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=PARTS) as pool:
        futures = [pool.submit(download_part, index) for index in range(PARTS)]
        while not all(f.done() for f in futures):
            done = downloaded()
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"downloaded={done / 1024**3:.3f}/{TOTAL / 1024**3:.3f} GiB speed={done / elapsed / 1024**2:.2f} MiB/s", flush=True)
            time.sleep(10)
        for future in futures:
            future.result()

    temporary = OUTPUT.with_suffix(".zip.complete.tmp")
    with temporary.open("wb") as target:
        for index in range(PARTS):
            start, end = bounds(index)
            part = PART_DIR / f"part-{index:02d}"
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"incomplete part: {part}")
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
    if temporary.stat().st_size != TOTAL:
        raise RuntimeError("assembled archive has the wrong size")
    os.replace(temporary, OUTPUT)
    print(f"complete={OUTPUT} bytes={OUTPUT.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
