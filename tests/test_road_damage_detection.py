import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from road_damage_detection import (
    DAMAGE_CLASSES,
    _split_counts,
    _yolo_label,
    build_train_options,
    parse_args,
    prepare_coco_dataset,
    validate_dataset,
)


class ConversionTests(unittest.TestCase):
    def test_coco_box_is_clipped_and_normalized(self) -> None:
        label = _yolo_label([-10, 10, 60, 40], 0, 100, 100)
        self.assertEqual(label, "0 0.25000000 0.30000000 0.50000000 0.40000000")
        self.assertIsNone(_yolo_label([10, 10, 0, 40], 0, 100, 100))

    def test_largest_remainder_split_preserves_every_image(self) -> None:
        counts = _split_counts(11, (0.7, 0.15, 0.15))
        self.assertEqual(counts, (8, 2, 1))
        self.assertEqual(sum(counts), 11)

    def test_prepare_is_deterministic_and_produces_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "source"
            images.mkdir()
            coco_images = []
            annotations = []
            for index in range(8):
                name = f"road-{index}.png"
                Image.new("RGB", (100, 80), (index * 10, 30, 50)).save(images / name)
                coco_images.append(
                    {"id": index, "file_name": name, "width": 100, "height": 80}
                )
                annotations.append(
                    {
                        "id": index,
                        "image_id": index,
                        "category_id": (index % 4) + 1,
                        "bbox": [10, 20, 30, 25],
                        "iscrowd": 0,
                    }
                )
            annotations.append(
                {
                    "id": 99,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [0, 0, -1, 5],
                    "iscrowd": 0,
                }
            )
            document = {
                "images": coco_images,
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "D00"},
                    {"id": 2, "name": "D10"},
                    {"id": 3, "name": "D20"},
                    {"id": 4, "name": "D40"},
                ],
            }
            annotations_path = root / "annotations.json"
            annotations_path.write_text(json.dumps(document), encoding="utf-8")

            first = root / "prepared-first"
            second = root / "prepared-second"
            first_manifest = prepare_coco_dataset(
                annotations_path, images, first, seed=23
            )
            second_manifest = prepare_coco_dataset(
                annotations_path, images, second, seed=23
            )

            self.assertEqual(first_manifest["splits"], second_manifest["splits"])
            self.assertEqual(
                [
                    first_manifest["splits"][name]["images"]
                    for name in ("train", "val", "test")
                ],
                [6, 1, 1],
            )
            self.assertEqual(first_manifest["preparation_counts"]["invalid_boxes"], 1)
            self.assertEqual(
                first_manifest["preprocessing"],
                {"autocontrast": False, "median_filter_size": 0},
            )
            audit = validate_dataset(first)
            self.assertTrue(audit["valid"])
            self.assertEqual(
                sum(split["boxes"] for split in audit["splits"].values()), 8
            )
            yaml_text = (first / "dataset.yaml").read_text(encoding="utf-8")
            self.assertIn(first.resolve().as_posix(), yaml_text)

    def test_prepare_refuses_to_replace_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "source"
            images.mkdir()
            Image.new("RGB", (16, 16), (20, 30, 40)).save(images / "road.png")
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "road.png"}],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "D00"}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_coco_dataset(annotations, images, output)


class DatasetAuditTests(unittest.TestCase):
    def _create_split(
        self, root: Path, split: str, color: tuple[int, int, int]
    ) -> Path:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_path = image_dir / f"{split}.png"
        Image.new("RGB", (32, 32), color).save(image_path)
        (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        return image_path

    def test_audit_detects_cross_split_duplicate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = self._create_split(root, "train", (20, 30, 40))
            validation = self._create_split(root, "val", (60, 70, 80))
            self._create_split(root, "test", (90, 100, 110))
            validation.write_bytes(train.read_bytes())
            (root / "dataset.yaml").write_text(
                "names:\n"
                + "\n".join(f"  {i}: {name}" for i, name in enumerate(DAMAGE_CLASSES)),
                encoding="utf-8",
            )

            audit = validate_dataset(root)
            self.assertFalse(audit["valid"])
            self.assertTrue(
                any("Cross-split data leakage" in error for error in audit["errors"])
            )

    def test_audit_rejects_box_outside_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_split(root, "train", (20, 30, 40))
            self._create_split(root, "val", (60, 70, 80))
            self._create_split(root, "test", (90, 100, 110))
            (root / "labels" / "train" / "train.txt").write_text(
                "0 0.9 0.5 0.4 0.4\n", encoding="utf-8"
            )
            (root / "dataset.yaml").write_text(
                "names:\n"
                + "\n".join(f"  {i}: {name}" for i, name in enumerate(DAMAGE_CLASSES)),
                encoding="utf-8",
            )

            audit = validate_dataset(root)
            self.assertFalse(audit["valid"])
            self.assertTrue(
                any("outside the image" in error for error in audit["errors"])
            )


class TrainingConfigurationTests(unittest.TestCase):
    def test_report_profile_and_cli_overrides_are_resolved(self) -> None:
        args = parse_args(
            [
                "train",
                "--dataset",
                "dataset.yaml",
                "--profile",
                "adamw-40",
                "--epochs",
                "2",
                "--model",
                "yolov8n.pt",
            ]
        )
        options = build_train_options(args)
        self.assertEqual(options["model"], "yolov8n.pt")
        self.assertEqual(options["epochs"], 2)
        self.assertEqual(options["optimizer"], "AdamW")
        self.assertEqual(options["mosaic"], 1.0)
        self.assertEqual(options["mixup"], 0.15)

    def test_invalid_split_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _split_counts(10, (0.8, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
