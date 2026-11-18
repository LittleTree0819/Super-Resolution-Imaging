from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def frozen_digest(rows: list[dict[str, object]]) -> str:
    frozen = [row for row in rows if row["split"] != "train"]
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in frozen)
    return sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot or verify the frozen validation/test split.")
    parser.add_argument("mode", choices=("snapshot", "check"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=2592)
    args = parser.parse_args()

    manifest = args.dataset / "manifest.jsonl"
    rows = load_manifest(manifest)
    counts = Counter(str(row["split"]) for row in rows)
    state = {
        "frozen_digest": frozen_digest(rows),
        "validation_ids": sorted(str(row["sample_id"]) for row in rows if row["split"] == "validation"),
        "test_ids": sorted(str(row["sample_id"]) for row in rows if row["split"] == "test"),
        "initial_counts": dict(counts),
        "expected_train": args.expected_train,
    }
    if args.mode == "snapshot":
        if args.snapshot.exists():
            raise FileExistsError(args.snapshot)
        args.snapshot.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state["initial_counts"], indent=2))
        return

    baseline = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if state["frozen_digest"] != baseline["frozen_digest"]:
        raise RuntimeError("Frozen validation/test records changed")
    if state["validation_ids"] != baseline["validation_ids"] or state["test_ids"] != baseline["test_ids"]:
        raise RuntimeError("Frozen validation/test membership changed")
    expected = {"train": args.expected_train, "validation": 72, "test": 80}
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected final split counts: {dict(counts)}; expected {expected}")
    print(json.dumps({"status": "ok", "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
