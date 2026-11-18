from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class DatasetWriter:
    def __init__(self, root: Path, *, resume: bool = False) -> None:
        self.root = root.resolve()
        existed = self.root.exists()
        if existed and not resume:
            raise FileExistsError(f"Output directory already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=resume)
        self.manifest = self.root / "manifest.jsonl"
        self.is_new = not existed

    def completed_ids(self) -> set[str]:
        if not self.manifest.exists():
            return set()
        completed: set[str] = set()
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid manifest record at line {line_number}") from exc
            if sample_id in completed:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            if not (self.root / "samples" / sample_id / "metadata.json").is_file():
                raise ValueError(f"Manifest references incomplete sample: {sample_id}")
            completed.add(sample_id)
        return completed

    def append(self, record: dict[str, object]) -> None:
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def complete(self, count: int) -> None:
        write_json(self.root / "COMPLETE.json", {"samples": count, "completed_at": datetime.now(timezone.utc).isoformat()})
