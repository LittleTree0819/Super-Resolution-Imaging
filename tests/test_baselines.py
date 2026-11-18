from __future__ import annotations

import unittest

import torch

from super_resolution_imaging.training.baselines import bicubic, four_frame_mean, metrics, shift_and_add


class BaselineTests(unittest.TestCase):
    def test_output_shapes_and_finite_values(self) -> None:
        frames = torch.rand(2, 4, 128, 128)
        shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, -0.5), (0.5, -0.5))
        outputs = (bicubic(frames[:, :1]), four_frame_mean(frames), shift_and_add(frames, shifts))
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 1, 256, 256))
            self.assertTrue(bool(torch.isfinite(output).all()))

    def test_identical_frames_have_identical_bicubic_and_mean(self) -> None:
        frame = torch.rand(1, 1, 128, 128)
        frames = frame.repeat(1, 4, 1, 1)
        self.assertTrue(torch.allclose(bicubic(frame), four_frame_mean(frames), atol=1e-7))

    def test_perfect_metrics(self) -> None:
        image = torch.rand(2, 1, 256, 256)
        l1, psnr, ssim = metrics(image, image)
        self.assertTrue(bool(torch.allclose(l1, torch.zeros_like(l1))))
        self.assertTrue(bool(torch.all(psnr >= 119.0)))
        self.assertTrue(bool(torch.allclose(ssim, torch.ones_like(ssim), atol=1e-5)))


if __name__ == "__main__":
    unittest.main()
