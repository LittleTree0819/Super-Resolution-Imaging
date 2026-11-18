# Preparing a Sequential Zemax Model

Supply a local sequential single-lens model and enter its path and SHA-256 digest in the configuration. The generator checks the digest before use and writes changes to a working copy.

For an unmodified model with object, front lens surface, rear lens surface, and image surface, insert the first Coordinate Break immediately before the front lens surface. Insert the second immediately after the rear lens surface and before the image. The resulting order is:

```text
Object | Coordinate Break | Front lens surface | Rear lens surface | Coordinate Break | Image
```

The rear lens surface's original image gap transfers to the second break. Set the first break's X and Y decenter parameters to the requested lens shift and the second break's parameters to the opposite values. At zero shift, the modified model should produce the same image as the original optical prescription.

If the input model already contains both breaks, use the configured comments `SR_DATASET_LENS_SHIFT_BEGIN` and `SR_DATASET_LENS_SHIFT_END`. The generator identifies and reuses a correctly marked pair. Review the indices in the example for an unmodified model before adapting it to another design.

Run `sr-zemax prepare-model`, open its `prepared_model.zmx` in the OpticStudio Lens Data Editor, and verify the surface order, glass, radii, lens thickness, aperture, total lens-to-image spacing, and zero-decenter image. Test a small lateral decenter to confirm that the lens moves while the image reference stays fixed.
