from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # OpticStudio 2026 R1 bundles Python 3.10.
    import tomli as tomllib


@dataclass(frozen=True)
class ZemaxConfig:
    install_dir: Path
    source_model: Path
    source_sha256: str
    images_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    front_surface: int
    rear_surface: int
    image_gap_mm: float
    before_comment: str
    after_comment: str


@dataclass(frozen=True)
class SimulationConfig:
    field_number: int
    field_height_mm: float
    lr_pixels: int
    lr_pixel_size_mm: float
    hr_pixels: int
    hr_pixel_size_mm: float
    pupil_sampling: int
    image_sampling: int
    psf_x_points: int
    psf_y_points: int
    reference: str
    aberrations: str


@dataclass(frozen=True)
class DatasetConfig:
    frames: int
    crop_size: int
    positions_mm: tuple[tuple[float, float], ...]
    split_train: int
    split_validation: int
    split_test: int
    seed: int


@dataclass(frozen=True)
class Config:
    zemax: ZemaxConfig
    model: ModelConfig
    simulation: SimulationConfig
    dataset: DatasetConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    z = raw["zemax"]
    m = raw["model"]
    s = raw["simulation"]
    d = raw["dataset"]
    config = Config(
        zemax=ZemaxConfig(Path(z["install_dir"]), Path(z["source_model"]), z["source_sha256"].lower(), Path(z["images_dir"])),
        model=ModelConfig(**m),
        simulation=SimulationConfig(**s),
        dataset=DatasetConfig(
            frames=d["frames"], crop_size=d["crop_size"],
            positions_mm=tuple(tuple(map(float, xy)) for xy in d["positions_mm"]),
            split_train=d["split_train"], split_validation=d["split_validation"],
            split_test=d["split_test"], seed=d["seed"],
        ),
    )
    _validate(config)
    return config


def _validate(c: Config) -> None:
    if c.dataset.frames != len(c.dataset.positions_mm):
        raise ValueError("dataset.frames must equal the number of positions_mm")
    if sum((c.dataset.split_train, c.dataset.split_validation, c.dataset.split_test)) != 100:
        raise ValueError("dataset split percentages must total 100")
    if c.simulation.hr_pixels != 2 * c.simulation.lr_pixels:
        raise ValueError("HR output must have twice the LR pixel count")
    if abs(c.simulation.lr_pixel_size_mm / 2 - c.simulation.hr_pixel_size_mm) > 1e-12:
        raise ValueError("HR pixel size must be half the LR pixel size")
    if c.simulation.reference != "Vertex":
        raise ValueError("Reference must be Vertex so lens-induced image motion is retained")
    if c.model.front_surface >= c.model.rear_surface:
        raise ValueError("front_surface must precede rear_surface")
