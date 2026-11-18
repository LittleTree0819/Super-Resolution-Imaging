from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "training extra is not installed")
class TrainingTests(unittest.TestCase):
    def test_brightness_losses_separate_constant_and_spatial_errors(self) -> None:
        from super_resolution_imaging.training.losses import global_brightness_loss, local_low_frequency_loss
        target = torch.full((2, 1, 32, 32), 0.4)
        offsets = torch.tensor([0.1, -0.1]).view(2, 1, 1, 1)
        prediction = target + offsets
        self.assertAlmostEqual(float(global_brightness_loss(prediction, target)), 0.1, places=6)
        self.assertLess(float(local_low_frequency_loss(prediction, target)), 1e-6)
        ramp = torch.linspace(-0.1, 0.1, 32).view(1, 1, 1, 32)
        prediction = target + ramp
        self.assertLess(float(global_brightness_loss(prediction, target)), 1e-6)
        self.assertGreater(float(local_low_frequency_loss(prediction, target)), 0.01)

    def test_brightness_losses_identical_and_finite_gradients(self) -> None:
        from super_resolution_imaging.training.losses import global_brightness_loss, local_low_frequency_loss
        target = torch.rand(2, 1, 32, 32)
        for loss_fn in (global_brightness_loss, local_low_frequency_loss):
            prediction = target.clone().requires_grad_(True)
            loss = loss_fn(prediction, target)
            self.assertEqual(float(loss.detach()), 0.0)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        prediction = (target * 0.8 + 0.03).requires_grad_(True)
        loss = global_brightness_loss(prediction, target) + local_low_frequency_loss(prediction, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_brightness_cli_defaults_are_disabled(self) -> None:
        from super_resolution_imaging.training.cli import build_parser
        args = build_parser().parse_args(['--dataset', '.', '--output', 'unused'])
        self.assertEqual(args.global_brightness_weight, 0.0)
        self.assertEqual(args.local_low_frequency_weight, 0.0)

    def test_dataset_and_model_shapes(self) -> None:
        from super_resolution_imaging.training.dataset import ZemaxBurstDataset
        from super_resolution_imaging.training.model import BurstUNet2x

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "samples" / "test_00"
            for group in ("a", "b"):
                (sample / group).mkdir(parents=True)
                for frame in range(4):
                    Image.fromarray(np.full((128, 128), frame, np.uint8)).save(sample / group / f"frame_{frame:02d}.png")
            Image.new("L", (256, 256), 127).save(sample / "target.png")
            (root / "manifest.jsonl").write_text(json.dumps({"sample_id": "test_00", "split": "train"}) + "\n")
            item = ZemaxBurstDataset(root, "b")[0]
            self.assertEqual(tuple(item["input"].shape), (4, 128, 128))
            self.assertEqual(tuple(item["target"].shape), (1, 256, 256))
            output = BurstUNet2x(base_channels=4)(item["input"].unsqueeze(0))
            self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
            self.assertTrue(bool(torch.all((0 <= output) & (output <= 1))))

    def test_zero_initialized_residual_starts_from_bicubic_b0(self) -> None:
        from torch.nn import functional as functional
        from super_resolution_imaging.training.model import BurstUNet2x

        inputs = torch.rand(2, 4, 64, 64)
        model = BurstUNet2x(base_channels=4, residual_output=True)
        expected = functional.interpolate(inputs[:, :1], scale_factor=2, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        self.assertTrue(torch.allclose(model(inputs), expected, atol=1e-7))

    def test_zero_initialized_split_residual_starts_from_bicubic_b0(self) -> None:
        from torch.nn import functional as functional
        from super_resolution_imaging.training.model import BurstUNet2x

        inputs = torch.rand(2, 4, 64, 64)
        model = BurstUNet2x(base_channels=4, split_residual_output=True)
        expected = functional.interpolate(inputs[:, :1], scale_factor=2, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        prediction, components = model.forward_with_components(inputs)
        self.assertTrue(torch.allclose(prediction, expected, atol=1e-7))
        self.assertTrue(torch.count_nonzero(components["low"]) == 0)
        self.assertTrue(torch.count_nonzero(components["high"]) == 0)

    def test_auto_device_selects_an_available_device(self) -> None:
        from super_resolution_imaging.training.cli import select_device

        device = select_device("auto")
        self.assertEqual(device.type, "cuda" if torch.cuda.is_available() else "cpu")

    def test_ssim_loss_is_zero_for_identical_images_and_has_finite_gradient(self) -> None:
        from super_resolution_imaging.training.losses import ssim_loss

        target = torch.rand(2, 1, 32, 32)
        identical = target.clone().requires_grad_(True)
        loss = ssim_loss(identical, target)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(identical.grad).all()))

    def test_validation_split_is_loaded_independently(self) -> None:
        from super_resolution_imaging.training.dataset import ZemaxBurstDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for sample_id, split in (("train_00", "train"), ("validation_00", "validation")):
                sample = root / "samples" / sample_id
                for group in ("a", "b"):
                    (sample / group).mkdir(parents=True)
                    for frame in range(4):
                        Image.new("L", (128, 128), frame).save(sample / group / f"frame_{frame:02d}.png")
                Image.new("L", (256, 256), 127).save(sample / "target.png")
                records.append(json.dumps({"sample_id": sample_id, "split": split}))
            (root / "manifest.jsonl").write_text("\n".join(records) + "\n")
            validation = ZemaxBurstDataset(root, "b", split="validation")
            self.assertEqual(validation.records[0]["sample_id"], "validation_00")

    def test_random_patch_keeps_lr_hr_coordinates_aligned(self) -> None:
        from super_resolution_imaging.training.dataset import ZemaxBurstDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "samples" / "patch_00"
            lr = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128) % 256
            for group in ("a", "b"):
                (sample / group).mkdir(parents=True)
                for frame in range(4):
                    Image.fromarray(lr.astype(np.uint8)).save(sample / group / f"frame_{frame:02d}.png")
            hr = np.repeat(np.repeat(lr.astype(np.uint8), 2, axis=0), 2, axis=1)
            Image.fromarray(hr).save(sample / "target.png")
            (root / "manifest.jsonl").write_text(json.dumps({"sample_id": "patch_00", "split": "train"}) + "\n")
            torch.manual_seed(5400)
            item = ZemaxBurstDataset(root, "b", patch_size=64)[0]
            left, top = item["crop_xy"].tolist()
            self.assertEqual(tuple(item["input"].shape), (4, 64, 64))
            self.assertEqual(tuple(item["target"].shape), (1, 128, 128))
            expected = torch.from_numpy(hr[2 * top:2 * (top + 64), 2 * left:2 * (left + 64)].copy()).float() / 255
            self.assertTrue(torch.equal(item["target"][0], expected))


if __name__ == "__main__":
    unittest.main()
