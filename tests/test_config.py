from pathlib import Path
import tempfile
import unittest

from super_resolution_imaging.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_consistent(self) -> None:
        config = load_config(ROOT / "configs" / "zemax_b.example.toml")
        self.assertEqual(config.dataset.frames, 4)
        self.assertEqual(len(config.dataset.positions_mm), 4)
        self.assertEqual(config.simulation.lr_pixels, 128)
        self.assertEqual(config.simulation.hr_pixels, 256)
        self.assertEqual(config.simulation.hr_pixels, 2 * config.simulation.lr_pixels)
        self.assertEqual(config.simulation.reference, "Vertex")
        self.assertEqual(config.dataset.positions_mm[1], (0.001089, 0.0))
        self.assertEqual(config.dataset.positions_mm[2], (0.0, 0.001031))

    def test_frame_count_mismatch_is_rejected(self) -> None:
        text = (ROOT / "configs" / "zemax_b.example.toml").read_text(encoding="utf-8").replace("frames = 4", "frames = 3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frames"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
