# Single-Lens Burst Super-Resolution

Super-Resolution-Imaging reconstructs a higher-resolution grayscale image from four simulated observations of the same scene. Ansys Zemax OpticStudio models how small lateral shifts of a single lens change the image sampled by a fixed detector. A PyTorch U-Net combines the shifted observations and predicts corrections to a bicubic reference image.

The reconstruction experiments use synthetic Zemax data throughout training, validation, and testing. A separate optical bench captured physical axial-defocus images. Those captures were outside the reconstruction experiments.

## Method

- **Group A** repeats one centered, aberrated 128 × 128 observation across four channels.
- **Group B** contains four ordered 128 × 128 observations at different lens positions.
- **Target** is a centered 256 × 256 Zemax render with aberrations disabled. Both grids cover the same simulated detector area.
- **Stage A** learns center-position reconstruction. **Stage B** initializes from the Stage A weights and fine-tunes on shifted observations.
- The split-residual U-Net adds separately supervised low- and high-frequency corrections to a 2× bicubic interpolation of the first observation.

The final layout targets image-plane offsets of ±0.25 low-resolution pixel per axis. The [dataset guide](docs/dataset_design.md) describes the sampling layouts, split policy, and output format.

## Selected results

These are mean full-image scores on the fixed **80-scene simulated test split**. The final checkpoint was selected on validation at epoch 34. Earlier development analyses also used this test split.

| Model | L1 ↓ | PSNR ↑ | SSIM ↑ |
| --- | ---: | ---: | ---: |
| Center A | 0.052483 | 23.5607 dB | 0.84362 |
| Original-layout B, epoch 19 | 0.051774 | 23.7393 dB | 0.86472 |
| Final-layout B, epoch 34 | 0.045961 | **24.9425 dB** | **0.91540** |

Final B gained **1.2032 dB** over original-layout B, with improvements on 69 of 80 scenes. This comparison includes differences in sampling layout, training budget, and training exclusions. In a separate four-frame pilot with matched update counts, changing the layout improved mean full-image validation PSNR by **0.2389 dB** across three seeds. Physical transverse reconstruction remains to be validated.

## Setup

The repository uses [uv](https://docs.astral.sh/uv/) and pins Python 3.12 for the default environment. Install the optional training dependencies with:

```powershell
uv sync --locked --extra train
uv run --no-sync python -m unittest discover -s tests
```

Training and evaluation can use CPU or CUDA with a compatible PyTorch installation. Rendering requires Windows, a licensed OpticStudio installation with ZOS-API access, and a compatible Python/.NET environment. Install the base dependencies with `uv sync --locked` if you only need the Zemax tools.

## Generate a dataset

Supply a sequential single-lens ZMX model and source images. DIV2K training images were used in the reported experiments. If needed, `uv run python scripts/download_div2k.py` downloads the archive into `data/`. Extract it and point `--input` to the directory containing source images.

From the repository root, copy the original-layout example to an ignored local file:

```powershell
Copy-Item configs/zemax_b.example.toml configs/zemax_b.local.toml
Get-FileHash -Algorithm SHA256 data/lens.zmx
```

Set `install_dir`, `source_model`, `source_sha256`, and `images_dir` in the copy. Review the surface indices and optical settings for your lens using the [Zemax model guide](docs/zemax_model.md). Paths in the example assume commands run from the repository root.

```powershell
uv run sr-zemax doctor --config configs/zemax_b.local.toml
uv run sr-zemax inspect-model --config configs/zemax_b.local.toml
uv run sr-zemax prepare-model --config configs/zemax_b.local.toml --output runs/model_review
uv run sr-zemax generate --config configs/zemax_b.local.toml --input data/DIV2K_train_HR --output runs/pilot --limit 2
```

Inspect the prepared copy in OpticStudio before generating the pilot. Check channel order, image orientation, lens-induced motion, and alignment with the target. The two-scene pilot checks rendering. For training, generate a dataset with both training and validation samples. Use a fresh output directory for a new run or `--resume` to continue a compatible incomplete run.

The [quarter-pixel example](configs/zemax_quarter_four.example.toml) provides the final layout's lens offsets. Its generic generation path uses the supplied input and crop settings. Exact reproduction of the reported release requires the original source crops, split, quality exclusions, local lens model, and frozen run records.

## Train and evaluate

The following commands illustrate Stage A and Stage B on a generated dataset at `runs/dataset`. Change the dataset path to your own generated output.

```powershell
uv run sr-train --dataset runs/dataset --output runs/train_a --group a --split-residual-output --ssim-loss-weight 0.2 --epochs 120 --learning-rate 3e-4
uv run sr-train --dataset runs/dataset --output runs/train_b --group b --split-residual-output --ssim-loss-weight 0.2 --epochs 100 --learning-rate 1e-4 --initialize-from runs/train_a/best.pt
uv run sr-evaluate --dataset runs/dataset --checkpoint-a runs/train_a/best.pt --checkpoint-b runs/train_b/best.pt --output runs/evaluation --split test
```

Use `--initialize-from` to carry model weights into a new stage. Use `--resume` with a stage's `last.pt` to continue the same run. Training saves `best.pt`, `last.pt`, and per-epoch metrics. The evaluator writes aggregate and per-scene L1, PSNR, and SSIM, plus paired PSNR statistics.

`sr-baselines` provides bicubic, frame averaging, and shift-and-add comparisons. Its default shift estimates are for the original layout. `sr-ablation` measures the effect of replacing or permuting individual input frames. Use `--help` on either command for options.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/super_resolution_imaging/` | OpticStudio integration, dataset generation, training, and evaluation |
| `configs/` | Public configuration examples and portable diagnostic protocols |
| `scripts/` | Data preparation and research utilities. Historical experiment scripts may require local configurations or saved runs. |
| `tests/` | Unit tests for the data pipeline, training components, and metrics |
| `docs/` | Dataset design and Zemax setup guides |

Source images, proprietary lens files, generated datasets, trained weights, and academic submission materials are kept outside the published repository.
