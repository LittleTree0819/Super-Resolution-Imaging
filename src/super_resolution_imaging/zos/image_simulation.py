from __future__ import annotations

from pathlib import Path
import time

from ..config import SimulationConfig


def _enum_member(enum: object, name: str) -> object:
    for candidate in (name, name + "_"):
        if hasattr(enum, candidate):
            return getattr(enum, candidate)
    raise AttributeError(f"{enum} has no enum member {name}")


def _sample_size(ZOSAPI: object, value: int) -> object:
    return _enum_member(ZOSAPI.Analysis.SampleSizes, f"S_{value}x{value}")


def run_image_simulation(
    session: object,
    input_name: str,
    output_path: Path,
    cfg: SimulationConfig,
    *,
    ideal: bool = False,
    attempts: int = 3,
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        analysis = session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.ImageSimulation)
        settings = analysis.GetSettings()
        settings.InputFile = input_name
        settings.FieldHeight = cfg.field_height_mm
        settings.Field.SetFieldNumber(cfg.field_number)
        settings.PupilSampling = _sample_size(session.ZOSAPI, cfg.pupil_sampling)
        settings.ImageSampling = _sample_size(session.ZOSAPI, cfg.image_sampling)
        settings.PSFXPoints = cfg.psf_x_points
        settings.PSFYPoints = cfg.psf_y_points
        settings.Reference = _enum_member(session.ZOSAPI.Analysis.Settings.ReferenceGia, cfg.reference)
        aberration_enum = session.ZOSAPI.Analysis.Settings.ExtendedScene.ISAberrationTypes
        settings.Aberrations = _enum_member(aberration_enum, "None" if ideal else cfg.aberrations)
        settings.XPixels = cfg.hr_pixels if ideal else cfg.lr_pixels
        settings.YPixels = cfg.hr_pixels if ideal else cfg.lr_pixels
        settings.PixelSize = cfg.hr_pixel_size_mm if ideal else cfg.lr_pixel_size_mm
        settings.OutputFile = str(output_path)
        try:
            analysis.ApplyAndWaitForCompletion()
        finally:
            analysis.Close()
        if output_path.is_file():
            return
        if attempt < attempts:
            print(
                f"warning: Zemax produced no output on attempt {attempt}/{attempts}; retrying {output_path.name}",
                flush=True,
            )
            time.sleep(attempt)
    raise RuntimeError(f"Zemax did not create the requested output after {attempts} attempts: {output_path}")
