from pathlib import Path
import tempfile
import unittest

from PIL import Image

from super_resolution_imaging.data.sources import crop_box, discover_images, prepare_source, split_for_hash
from super_resolution_imaging.data.writer import DatasetWriter


class SourceTests(unittest.TestCase):
    def test_crop_is_deterministic_and_in_bounds(self) -> None:
        first = crop_box(1200, 900, 768, 5400, "abc", 0)
        self.assertEqual(first, crop_box(1200, 900, 768, 5400, "abc", 0))
        self.assertGreaterEqual(first[0], 0)
        self.assertLessEqual(first[2], 1200)

    def test_prepare_source_writes_grayscale_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (800, 800), (10, 20, 30)).save(root / "input.png")
            metadata = prepare_source(root / "input.png", root / "out.png", 768, 1, 0)
            with Image.open(root / "out.png") as result:
                self.assertEqual(result.mode, "L")
                self.assertEqual(result.size, (768, 768))
            self.assertEqual(len(metadata["source_sha256"]), 64)
            self.assertEqual(metadata["pre_crop_scale"], 1.0)

    def test_short_image_is_scaled_before_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2040, 732), (10, 20, 30)).save(root / "wide.png")
            metadata = prepare_source(root / "wide.png", root / "out.png", 768, 1, 0)
            with Image.open(root / "out.png") as result:
                self.assertEqual(result.size, (768, 768))
            self.assertEqual(metadata["original_size"], [2040, 732])
            self.assertEqual(metadata["prepared_size"][1], 768)
            self.assertGreater(metadata["pre_crop_scale"], 1.0)

    def test_split_is_stable(self) -> None:
        self.assertEqual(split_for_hash("00000000", 80, 10), "train")
        self.assertEqual(split_for_hash("00000059", 80, 10), "validation")
        self.assertEqual(split_for_hash("00000063", 80, 10), "test")

    def test_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset"
            DatasetWriter(path)
            with self.assertRaises(FileExistsError):
                DatasetWriter(path)

    def test_writer_resumes_only_manifested_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset"
            writer = DatasetWriter(path)
            sample = path / "samples" / "0001_00"
            sample.mkdir(parents=True)
            (sample / "metadata.json").write_text("{}", encoding="utf-8")
            writer.append({"sample_id": "0001_00"})
            resumed = DatasetWriter(path, resume=True)
            self.assertEqual(resumed.completed_ids(), {"0001_00"})


if __name__ == "__main__":
    unittest.main()
