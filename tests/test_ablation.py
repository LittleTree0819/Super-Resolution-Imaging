from __future__ import annotations

import unittest

import torch

from super_resolution_imaging.training.ablation import input_variants


class AblationTests(unittest.TestCase):
    def test_channel_replacements_and_swap(self) -> None:
        frames = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        variants = input_variants(frames)
        self.assertEqual(variants["repeat_b0"].flatten().tolist(), [0, 0, 0, 0])
        self.assertEqual(variants["replace_b1"].flatten().tolist(), [0, 0, 2, 3])
        self.assertEqual(variants["replace_b2"].flatten().tolist(), [0, 1, 0, 3])
        self.assertEqual(variants["replace_b3"].flatten().tolist(), [0, 1, 2, 0])
        self.assertEqual(variants["swap_b1_b2"].flatten().tolist(), [0, 2, 1, 3])

    def test_all_frame_combinations_preserve_channel_positions(self) -> None:
        from super_resolution_imaging.training.frame_combinations import input_combinations

        frames = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        variants = input_combinations(frames)
        self.assertEqual(len(variants), 8)
        self.assertEqual(variants[()].flatten().tolist(), [0, 0, 0, 0])
        self.assertEqual(variants[(1, 3)].flatten().tolist(), [0, 1, 0, 3])
        self.assertTrue(torch.equal(variants[(1, 2, 3)], frames))


if __name__ == "__main__":
    unittest.main()
