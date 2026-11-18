from pathlib import Path
import tempfile
import unittest

from PIL import Image

from super_resolution_imaging.pipeline import _normalize_grayscale, _verify_size


class PipelineImageTests(unittest.TestCase):
    def test_rgb_output_is_normalized_to_single_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zemax.png"
            Image.new("RGB", (128, 128), (10, 20, 30)).save(path)
            _normalize_grayscale(path)
            _verify_size(path, 128)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "L")

    def test_wrong_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.png"
            Image.new("L", (127, 128)).save(path)
            with self.assertRaisesRegex(RuntimeError, "Unexpected image size"):
                _verify_size(path, 128)


if __name__ == "__main__":
    unittest.main()
