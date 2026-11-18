# Dataset Design

Each sample contains a grayscale scene crop, one ideal high-resolution target, and two groups of low-resolution observations. Group A repeats a centered observation across all channels. Group B stores the observations in the order defined by `dataset.positions_mm`. Both groups share the same scene and target.

## Optical simulation

1. Prepare a copy of the sequential lens model with one Coordinate Break before the lens and another after it.
2. Apply `(dx, dy)` at the first break and `(-dx, -dy)` at the second. This keeps the detector reference fixed while shifting the lens.
3. Use Image Simulation with the `Vertex` reference, which retains lens-induced image motion.
4. Render Group B at each configured decenter and Group A at zero decenter. The standard examples use 128 × 128 pixels at 2.4 µm pitch and diffraction simulation.
5. Render the centered target with aberrations disabled at 256 × 256 pixels and 1.2 µm pitch over the same detector area.

The generator computes a single centered low-resolution render and reuses it for Group A. It also reuses that render for any zero-offset Group B observation. The generated images represent optical simulation and have no separately calibrated camera-noise model. Image Simulation intensity scaling should be checked before making radiometric claims.

## Sampling layouts

The original four-frame example uses a centered observation, an X-only shift, a Y-only shift, and a diagonal shift. Mechanical steps of 1.089 µm in X and 1.031 µm in Y gave an estimated combined raster shift of `(+0.508, -0.506)` low-resolution pixels in the calibration pilot.

The final example targets the ordered raster offsets `(+0.25, +0.25)`, `(-0.25, +0.25)`, `(-0.25, -0.25)`, and `(+0.25, -0.25)` in low-resolution pixels. Raster Y increases downward. The configured lens offsets are signed mechanical decenters in millimetres.

## Source images and outputs

Convert each source image to grayscale and take a deterministic 768 × 768 crop. If the shorter side is below 768 pixels, first upscale proportionally with Lanczos resampling. Assign source images to splits by content hash, so crops from the same source stay in one split.

A generated dataset contains `dataset.json`, `manifest.jsonl`, a `COMPLETE.json` marker on successful completion, and sample folders with this structure:

```text
samples/<sample_id>/
  source.png
  target.png
  a/frame_00.png ... frame_03.png
  b/frame_00.png ... frame_03.png
  metadata.json
```

Sample records include the source hash, crop box, configured shifts, and output hashes. The source crop supplies the scene shown to Zemax. The target is rendered separately and retains the orientation of the simulated image.

## Pilot checks

- Confirm that the Coordinate Breaks preserve the lens prescription and axial spacing. Compare the zero-decenter result with the original model.
- Use a dot grid or sharp edges to measure image motion along both axes. Check sign, channel order, image orientation, and target registration.
- Inspect field edges, vignetting, clipping, and simulation messages. Record any diffraction-to-geometric fallback before scaling up.
- For physical comparison, characterize exposure stability and sensor response separately using dark frames, flat fields, and repeated captures.
