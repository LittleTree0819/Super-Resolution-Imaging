from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .pipeline import generate
from .zos.connection import ZOSConnectionError, connect, installation_report
from .zos.model import prepare_shift_model, validate_source


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sr-zemax")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "inspect-model"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    prepare = commands.add_parser("prepare-model")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    create = commands.add_parser("generate")
    create.add_argument("--config", required=True, type=Path)
    create.add_argument("--input", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--limit", type=int)
    create.add_argument("--start", type=int, default=0)
    create.add_argument("--patches", type=int, default=1)
    create.add_argument(
        "--only-split", choices=("train", "validation", "test"),
        help="Generate patches only for source images assigned to this split",
    )
    create.add_argument("--resume", action="store_true")
    return root


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command == "doctor":
            report = installation_report(config.zemax.install_dir)
            report["source_model"] = validate_source(config)
            report["images_dir_exists"] = config.zemax.images_dir.is_dir()
            with connect(config.zemax.install_dir) as session:
                report["zos_api"] = "connected"
                report["license_status"] = str(session.app.LicenseStatus)
            _print(report)
            return 0
        if args.command == "inspect-model":
            report = validate_source(config)
            with connect(config.zemax.install_dir) as session:
                session.system.LoadFile(str(config.zemax.source_model.resolve()), False)
                report["surface_count"] = int(session.system.LDE.NumberOfSurfaces)
                report["system_mode"] = str(session.system.Mode)
            _print(report)
            return 0
        if args.command == "prepare-model":
            output = args.output.resolve()
            if output.exists():
                raise FileExistsError(output)
            output.mkdir(parents=True)
            with connect(config.zemax.install_dir) as session:
                report = prepare_shift_model(session, config, output / "prepared_model.zmx")
            _print(report)
            return 0
        if args.command == "generate":
            with connect(config.zemax.install_dir) as session:
                count = generate(
                    session, config, args.input, args.output, args.limit, args.patches,
                    start=args.start, resume=args.resume, only_split=args.only_split,
                )
            _print({"samples": count, "output": str(args.output.resolve())})
            return 0
    except (ZOSConnectionError, FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
