from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..config import Config


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(config: Config) -> dict[str, object]:
    source = config.zemax.source_model.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = file_sha256(source)
    if actual != config.zemax.source_sha256:
        raise ValueError(f"Source model hash mismatch: expected {config.zemax.source_sha256}, got {actual}")
    return {"path": str(source), "sha256": actual, "size_bytes": source.stat().st_size}


def _surface_type(settings: object, name: str) -> None:
    surface_types = settings.GetType().Assembly.GetType("ZOSAPI.Editors.LDE.SurfaceType")
    import System
    enum_value = System.Enum.Parse(surface_types, name)
    settings.Type = enum_value


def prepare_shift_model(session: object, config: Config, destination: Path) -> dict[str, object]:
    validate_source(config)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)

    system = session.system
    system.LoadFile(str(config.zemax.source_model.resolve()), False)
    lde = system.LDE

    # A manually prepared source model is preferred. Copy it without inserting
    # another pair of Coordinate Breaks, then let all later code locate the pair
    # by their stable comments.
    try:
        before, after = find_shift_surfaces(system, config)
    except RuntimeError:
        before = after = None
    if before is not None and after is not None:
        if "Coordinate" not in str(before.TypeName) or "Coordinate" not in str(after.TypeName):
            raise ValueError("Named shift surfaces must both be Coordinate Break surfaces")
        gap = float(after.Thickness)
        if abs(float(before.Thickness)) > 1e-12:
            raise ValueError("The first Coordinate Break thickness must be 0 mm")
        if abs(gap - config.model.image_gap_mm) > 1e-9:
            raise ValueError(
                f"The second Coordinate Break thickness must be {config.model.image_gap_mm} mm, got {gap}"
            )
        system.SaveAs(str(destination))
        return {
            "destination": str(destination),
            "mode": "reused_existing_coordinate_breaks",
            "image_gap_mm": gap,
            "surface_count": int(lde.NumberOfSurfaces),
        }

    before = lde.InsertNewSurfaceAt(config.model.front_surface)
    type_settings = before.GetSurfaceTypeSettings(session.ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak)
    before.ChangeType(type_settings)
    before.Comment = config.model.before_comment
    before.Thickness = 0.0

    rear_after_insert = config.model.rear_surface + 1
    rear = lde.GetSurfaceAt(rear_after_insert)
    original_gap = float(rear.Thickness)
    after_index = rear_after_insert + 1
    after = lde.InsertNewSurfaceAt(after_index)
    after.ChangeType(after.GetSurfaceTypeSettings(session.ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak))
    after.Comment = config.model.after_comment
    rear.Thickness = 0.0
    after.Thickness = original_gap
    system.SaveAs(str(destination))
    return {
        "destination": str(destination),
        "mode": "inserted_coordinate_breaks",
        "original_image_gap_mm": original_gap,
        "surface_count": int(lde.NumberOfSurfaces),
    }


def find_shift_surfaces(system: object, config: Config) -> tuple[object, object]:
    found: dict[str, tuple[int, object]] = {}
    for index in range(int(system.LDE.NumberOfSurfaces)):
        surface = system.LDE.GetSurfaceAt(index)
        if str(surface.Comment) in (config.model.before_comment, config.model.after_comment):
            comment = str(surface.Comment)
            if comment in found:
                raise RuntimeError(f"Duplicate Coordinate Break comment: {comment}")
            found[comment] = (index, surface)
    try:
        before_index, before = found[config.model.before_comment]
        after_index, after = found[config.model.after_comment]
    except KeyError as exc:
        raise RuntimeError("Prepared model does not contain both named Coordinate Break surfaces") from exc
    if before_index >= after_index:
        raise RuntimeError("The begin Coordinate Break must precede the end Coordinate Break")
    return before, after


def set_lens_shift(session: object, config: Config, dx_mm: float, dy_mm: float) -> None:
    before, after = find_shift_surfaces(session.system, config)
    columns = session.ZOSAPI.Editors.LDE.SurfaceColumn
    before.GetSurfaceCell(columns.Par1).DoubleValue = float(dx_mm)
    before.GetSurfaceCell(columns.Par2).DoubleValue = float(dy_mm)
    after.GetSurfaceCell(columns.Par1).DoubleValue = -float(dx_mm)
    after.GetSurfaceCell(columns.Par2).DoubleValue = -float(dy_mm)
