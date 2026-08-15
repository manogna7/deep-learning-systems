from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DAMAGE_CLASSES = ("D00", "D10", "D20", "D40")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_DATASET = Path("data/road-damage/dataset.yaml")
DEFAULT_ARTIFACTS = Path("artifacts/road-damage")


@dataclass(frozen=True)
class ExperimentProfile:
    model: str
    epochs: int
    image_size: int
    batch_size: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    mosaic: float
    mixup: float
    hsv_h: float
    hsv_s: float
    hsv_v: float


EXPERIMENT_PROFILES = {
    "sgd-30": ExperimentProfile(
        model="yolov8m.pt",
        epochs=30,
        image_size=640,
        batch_size=16,
        optimizer="SGD",
        learning_rate=0.005,
        momentum=0.9,
        weight_decay=0.0004,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    ),
    "adamw-40": ExperimentProfile(
        model="yolov8m.pt",
        epochs=40,
        image_size=640,
        batch_size=16,
        optimizer="AdamW",
        learning_rate=0.0015,
        momentum=0.85,
        weight_decay=0.0004,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    ),
    "sgd-50": ExperimentProfile(
        model="yolov8m.pt",
        epochs=50,
        image_size=640,
        batch_size=32,
        optimizer="SGD",
        learning_rate=0.0007,
        momentum=0.85,
        weight_decay=0.00004,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    ),
}


@dataclass(frozen=True)
class PreparedImage:
    source: Path
    source_name: str
    labels: tuple[str, ...]


class DatasetValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_category_name(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _class_index(category_name: object) -> int | None:
    aliases = {
        "d00": 0,
        "longitudinalcrack": 0,
        "longitudinalcracks": 0,
        "d10": 1,
        "transversecrack": 1,
        "transversecracks": 1,
        "lateralcrack": 1,
        "d20": 2,
        "alligatorcrack": 2,
        "alligatorcracks": 2,
        "d40": 3,
        "pothole": 3,
        "potholes": 3,
    }
    return aliases.get(_normalized_category_name(category_name))


def _read_coco_annotations(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"Could not read COCO annotations: {error}"
        ) from error

    required = {"images", "annotations", "categories"}
    if not isinstance(document, dict) or not required.issubset(document):
        raise DatasetValidationError(
            "COCO annotations must contain images, annotations, and categories arrays."
        )
    for key in required:
        if not isinstance(document[key], list):
            raise DatasetValidationError(f"COCO field '{key}' must be an array.")
    return document


def _safe_source_path(images_dir: Path, file_name: object) -> Path:
    root = images_dir.resolve()
    candidate = (root / str(file_name)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise DatasetValidationError(
            f"Image path escapes the source directory: {file_name}"
        ) from error
    return candidate


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for dataset preparation. Install requirements.txt."
        ) from error

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetValidationError(f"Unreadable image '{path}': {error}") from error
    if width <= 0 or height <= 0:
        raise DatasetValidationError(f"Image has invalid dimensions: {path}")
    return width, height


def _yolo_label(
    bbox: object,
    class_index: int,
    image_width: int,
    image_height: int,
) -> str | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    x_min = min(max(x, 0.0), float(image_width))
    y_min = min(max(y, 0.0), float(image_height))
    x_max = min(max(x + width, 0.0), float(image_width))
    y_max = min(max(y + height, 0.0), float(image_height))
    if x_max <= x_min or y_max <= y_min:
        return None

    center_x = ((x_min + x_max) / 2.0) / image_width
    center_y = ((y_min + y_max) / 2.0) / image_height
    normalized_width = (x_max - x_min) / image_width
    normalized_height = (y_max - y_min) / image_height
    return (
        f"{class_index} {center_x:.8f} {center_y:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}"
    )


def _split_counts(total: int, ratios: Sequence[float]) -> tuple[int, ...]:
    if total < 0:
        raise ValueError("The image count cannot be negative.")
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError(
            "Three non-negative train, validation, and test ratios are required."
        )
    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(
        range(len(raw)),
        key=lambda item: (raw[item] - counts[item], -item),
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1

    if total >= 3:
        for target in (1, 2):
            if ratios[target] > 0 and counts[target] == 0:
                donor = max(range(3), key=lambda item: counts[item])
                counts[donor] -= 1
                counts[target] += 1
    return tuple(counts)


def _prepare_images(
    document: dict[str, Any], images_dir: Path
) -> tuple[list[PreparedImage], Counter[str]]:
    category_map: dict[object, int] = {}
    for category in document["categories"]:
        if not isinstance(category, dict) or "id" not in category:
            continue
        mapped = _class_index(category.get("name"))
        if mapped is not None:
            category_map[category["id"]] = mapped
    if not category_map:
        raise DatasetValidationError(
            "No D00, D10, D20, or D40 categories were found in the COCO annotations."
        )

    annotations_by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for annotation in document["annotations"]:
        if not isinstance(annotation, dict) or "image_id" not in annotation:
            counters["malformed_annotations"] += 1
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    prepared: list[PreparedImage] = []
    seen_ids: set[object] = set()
    for image_record in sorted(
        document["images"], key=lambda item: str(item.get("file_name", ""))
    ):
        if not isinstance(image_record, dict):
            counters["malformed_image_records"] += 1
            continue
        image_id = image_record.get("id")
        file_name = image_record.get("file_name")
        if image_id is None or not file_name or image_id in seen_ids:
            counters["malformed_image_records"] += 1
            continue
        seen_ids.add(image_id)

        source = _safe_source_path(images_dir, file_name)
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            counters["missing_or_unsupported_images"] += 1
            continue
        try:
            width, height = _image_size(source)
        except DatasetValidationError:
            counters["corrupt_images"] += 1
            continue

        labels: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            class_index = category_map.get(annotation.get("category_id"))
            if class_index is None or annotation.get("iscrowd", 0):
                counters["ignored_annotations"] += 1
                continue
            label = _yolo_label(annotation.get("bbox"), class_index, width, height)
            if label is None:
                counters["invalid_boxes"] += 1
                continue
            labels.append(label)

        prepared.append(
            PreparedImage(
                source=source,
                source_name=str(file_name).replace("\\", "/"),
                labels=tuple(labels),
            )
        )
    counters["accepted_images"] = len(prepared)
    return prepared, counters


def _write_dataset_yaml(root: Path) -> None:
    lines = [
        f"path: {json.dumps(root.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(DAMAGE_CLASSES))
    (root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preprocessed_image(
    source: Path,
    destination: Path,
    enhance_contrast: bool,
    median_filter_size: int,
) -> None:
    if not enhance_contrast and median_filter_size == 0:
        shutil.copy2(source, destination)
        return
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image preprocessing. Install requirements.txt."
        ) from error

    with Image.open(source) as image:
        processed = image.convert("RGB")
        if enhance_contrast:
            processed = ImageOps.autocontrast(processed)
        if median_filter_size:
            processed = processed.filter(ImageFilter.MedianFilter(median_filter_size))
        processed.save(destination)


def _write_prepared_dataset(
    root: Path,
    splits: dict[str, list[PreparedImage]],
    annotations: Path,
    images_dir: Path,
    seed: int,
    ratios: Sequence[float],
    counters: Counter[str],
    enhance_contrast: bool,
    median_filter_size: int,
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split_name, images in splits.items():
        image_output = root / "images" / split_name
        label_output = root / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        stems: set[str] = set()
        class_counts: Counter[str] = Counter()
        for prepared in images:
            if prepared.source.stem.casefold() in stems:
                raise DatasetValidationError(
                    f"Duplicate image stem '{prepared.source.stem}' in the {split_name} split."
                )
            stems.add(prepared.source.stem.casefold())
            destination = image_output / prepared.source.name
            _write_preprocessed_image(
                prepared.source,
                destination,
                enhance_contrast=enhance_contrast,
                median_filter_size=median_filter_size,
            )
            (label_output / f"{prepared.source.stem}.txt").write_text(
                "\n".join(prepared.labels) + ("\n" if prepared.labels else ""),
                encoding="utf-8",
            )
            for label in prepared.labels:
                class_counts[DAMAGE_CLASSES[int(label.split()[0])]] += 1

        split_summary[split_name] = {
            "images": len(images),
            "boxes": sum(class_counts.values()),
            "classes": dict(sorted(class_counts.items())),
            "sources": sorted(image.source_name for image in images),
        }

    _write_dataset_yaml(root)
    manifest = {
        "schema_version": 1,
        "classes": list(DAMAGE_CLASSES),
        "source": {
            "annotations": str(annotations.resolve()),
            "annotations_sha256": _sha256(annotations),
            "images": str(images_dir.resolve()),
        },
        "split": {
            "seed": seed,
            "ratios": {name: ratio for name, ratio in zip(splits, ratios)},
        },
        "preprocessing": {
            "autocontrast": enhance_contrast,
            "median_filter_size": median_filter_size,
        },
        "splits": split_summary,
        "preparation_counts": dict(sorted(counters.items())),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_coco_dataset(
    annotations: Path,
    images_dir: Path,
    output_dir: Path,
    seed: int = 17,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    enhance_contrast: bool = False,
    median_filter_size: int = 0,
) -> dict[str, Any]:
    annotations = annotations.resolve()
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    if not annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotations}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new directory."
        )
    if output_dir == images_dir or images_dir in output_dir.parents:
        raise DatasetValidationError(
            "The output directory must not be the source image directory or inside it."
        )
    if median_filter_size != 0 and (
        median_filter_size < 3 or median_filter_size % 2 == 0
    ):
        raise ValueError(
            "The median filter size must be zero or an odd integer of at least 3."
        )

    ratios = (train_ratio, val_ratio, test_ratio)
    document = _read_coco_annotations(annotations)
    prepared, counters = _prepare_images(document, images_dir)
    if not prepared:
        raise DatasetValidationError(
            "No valid source images were available for conversion."
        )

    random.Random(seed).shuffle(prepared)
    train_count, val_count, _ = _split_counts(len(prepared), ratios)
    splits = {
        "train": prepared[:train_count],
        "val": prepared[train_count : train_count + val_count],
        "test": prepared[train_count + val_count :],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent))
    )
    try:
        manifest = _write_prepared_dataset(
            staging,
            splits,
            annotations,
            images_dir,
            seed,
            ratios,
            counters,
            enhance_contrast,
            median_filter_size,
        )
        validation = validate_dataset(staging, check_hashes=True)
        if validation["errors"]:
            raise DatasetValidationError("; ".join(validation["errors"]))
        staging.replace(output_dir)
        _write_dataset_yaml(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _iter_images(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_label(path: Path) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return counts, [f"Could not read label '{path}': {error}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number} must contain five values.")
            continue
        try:
            class_index = int(parts[0])
            center_x, center_y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_number} contains a non-numeric value.")
            continue
        values = (center_x, center_y, width, height)
        if class_index not in range(len(DAMAGE_CLASSES)):
            errors.append(f"{path}:{line_number} has class index {class_index}.")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number} contains a non-finite coordinate.")
            continue
        if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
            errors.append(f"{path}:{line_number} has a center outside the image.")
            continue
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            errors.append(f"{path}:{line_number} has an invalid box size.")
            continue
        tolerance = 1e-6
        if (
            center_x - width / 2 < -tolerance
            or center_x + width / 2 > 1 + tolerance
            or center_y - height / 2 < -tolerance
            or center_y + height / 2 > 1 + tolerance
        ):
            errors.append(f"{path}:{line_number} has a box outside the image.")
            continue
        counts[DAMAGE_CLASSES[class_index]] += 1
    return counts, errors


def validate_dataset(dataset_root: Path, check_hashes: bool = True) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_summary: dict[str, Any] = {}
    hashes: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.is_file():
        errors.append(f"Missing dataset configuration: {yaml_path}")
    else:
        yaml_text = yaml_path.read_text(encoding="utf-8")
        for class_name in DAMAGE_CLASSES:
            if class_name not in yaml_text:
                errors.append(f"dataset.yaml does not declare class {class_name}.")

    for split_name in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(
                f"Missing images/labels directories for the {split_name} split."
            )
            continue

        images = list(_iter_images(image_dir))
        if not images:
            warnings.append(f"The {split_name} split contains no images.")
        class_counts: Counter[str] = Counter()
        label_paths: set[Path] = set()
        for image_path in images:
            try:
                _image_size(image_path)
            except DatasetValidationError as error:
                errors.append(str(error))
            label_path = label_dir / f"{image_path.stem}.txt"
            label_paths.add(label_path.resolve())
            if not label_path.is_file():
                errors.append(f"Missing label for image: {image_path}")
            else:
                label_counts, label_errors = _validate_label(label_path)
                class_counts.update(label_counts)
                errors.extend(label_errors)
            if check_hashes:
                hashes[_sha256(image_path)].append((split_name, image_path))

        orphaned_labels = [
            path
            for path in sorted(label_dir.glob("*.txt"))
            if path.resolve() not in label_paths
        ]
        errors.extend(
            f"Label has no matching image: {path}" for path in orphaned_labels
        )
        split_summary[split_name] = {
            "images": len(images),
            "boxes": sum(class_counts.values()),
            "classes": dict(sorted(class_counts.items())),
        }

    duplicate_groups: list[dict[str, Any]] = []
    if check_hashes:
        for digest, entries in sorted(hashes.items()):
            if len(entries) < 2:
                continue
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "files": [str(path) for _, path in entries],
                    "splits": sorted({split_name for split_name, _ in entries}),
                }
            )
            split_names = {split_name for split_name, _ in entries}
            if len(split_names) > 1:
                errors.append(
                    "Cross-split data leakage: "
                    + ", ".join(str(path) for _, path in entries)
                )
            else:
                warnings.append(
                    "Duplicate image within a split: "
                    + ", ".join(str(path) for _, path in entries)
                )

    return {
        "dataset_root": str(dataset_root),
        "classes": list(DAMAGE_CLASSES),
        "splits": split_summary,
        "duplicate_groups": duplicate_groups,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _dataset_root(dataset_yaml: Path) -> Path:
    dataset_yaml = dataset_yaml.resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")
    return dataset_yaml.parent


def _require_ultralytics() -> Any:
    local_config = (
        Path(__file__).resolve().parent / DEFAULT_ARTIFACTS / "config"
    ).resolve()
    if "YOLO_CONFIG_DIR" not in os.environ:
        local_config.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(local_config)
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_config = local_config / "matplotlib"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for this command. Install requirements.txt."
        ) from error
    return YOLO


def _profile_value(args: argparse.Namespace, name: str, profile_name: str) -> Any:
    override = getattr(args, name)
    if override is not None:
        return override
    return getattr(EXPERIMENT_PROFILES[profile_name], name)


def build_train_options(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = args.profile
    options = {
        "data": str(args.dataset.resolve()),
        "model": _profile_value(args, "model", profile_name),
        "epochs": _profile_value(args, "epochs", profile_name),
        "imgsz": _profile_value(args, "image_size", profile_name),
        "batch": _profile_value(args, "batch_size", profile_name),
        "optimizer": _profile_value(args, "optimizer", profile_name),
        "lr0": _profile_value(args, "learning_rate", profile_name),
        "momentum": _profile_value(args, "momentum", profile_name),
        "weight_decay": _profile_value(args, "weight_decay", profile_name),
        "mosaic": _profile_value(args, "mosaic", profile_name),
        "mixup": _profile_value(args, "mixup", profile_name),
        "hsv_h": _profile_value(args, "hsv_h", profile_name),
        "hsv_s": _profile_value(args, "hsv_s", profile_name),
        "hsv_v": _profile_value(args, "hsv_v", profile_name),
        "amp": not args.no_amp,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "cos_lr": args.cosine_schedule,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name or profile_name,
        "exist_ok": args.exist_ok,
        "pretrained": not args.no_pretrained,
        "plots": True,
    }
    if args.device is not None:
        options["device"] = args.device
    if args.freeze is not None:
        options["freeze"] = args.freeze
    if args.fraction is not None:
        options["fraction"] = args.fraction
    return options


def run_training(args: argparse.Namespace) -> None:
    dataset_root = _dataset_root(args.dataset)
    if not args.skip_data_validation:
        audit = validate_dataset(dataset_root, check_hashes=True)
        if audit["errors"]:
            raise DatasetValidationError("; ".join(audit["errors"]))

    options = build_train_options(args)
    model_reference = args.resume.resolve() if args.resume else options.pop("model")
    model = _require_ultralytics()(str(model_reference))
    if args.resume:
        options["resume"] = True
    model.train(**options)


def run_evaluation(args: argparse.Namespace) -> None:
    dataset_root = _dataset_root(args.dataset)
    if not args.skip_data_validation:
        audit = validate_dataset(dataset_root, check_hashes=True)
        if audit["errors"]:
            raise DatasetValidationError("; ".join(audit["errors"]))

    model = _require_ultralytics()(str(args.weights.resolve()))
    options: dict[str, Any] = {
        "data": str(args.dataset.resolve()),
        "split": args.split,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "conf": args.confidence,
        "iou": args.iou,
        "plots": True,
        "save_json": args.save_json,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        options["device"] = args.device
    model.val(**options)


def _prediction_record(result: Any, sequence_index: int) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None:
        class_ids = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        coordinates = boxes.xyxy.detach().cpu().tolist()
        for class_id, confidence, xyxy in zip(class_ids, confidences, coordinates):
            integer_class_id = int(class_id)
            detections.append(
                {
                    "class_id": integer_class_id,
                    "class_name": str(result.names[integer_class_id]),
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in xyxy],
                }
            )
    return {
        "sequence_index": sequence_index,
        "source": str(result.path),
        "original_shape": list(result.orig_shape),
        "speed_ms": {key: float(value) for key, value in result.speed.items()},
        "detections": detections,
    }


def run_prediction(args: argparse.Namespace) -> None:
    model = _require_ultralytics()(str(args.weights.resolve()))
    options: dict[str, Any] = {
        "source": str(args.source),
        "imgsz": args.image_size,
        "conf": args.confidence,
        "iou": args.iou,
        "stream": True,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        options["device"] = args.device

    output_stream = None
    output_path: Path | None = None
    try:
        for index, result in enumerate(model.predict(**options)):
            if output_stream is None:
                output_dir = Path(model.predictor.save_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "detections.jsonl"
                output_stream = output_path.open("w", encoding="utf-8")
            output_stream.write(
                json.dumps(_prediction_record(result, index), allow_nan=False) + "\n"
            )
    finally:
        if output_stream is not None:
            output_stream.close()
    if output_path is None:
        raise RuntimeError("Inference produced no results for the supplied source.")
    print(f"Structured detections: {output_path}")


def run_export(args: argparse.Namespace) -> None:
    model = _require_ultralytics()(str(args.weights.resolve()))
    options: dict[str, Any] = {
        "format": args.format,
        "imgsz": args.image_size,
        "dynamic": args.dynamic,
    }
    if args.precision == "fp16":
        options["quantize"] = 16
    elif args.precision == "int8":
        dataset_root = _dataset_root(args.dataset)
        audit = validate_dataset(dataset_root, check_hashes=True)
        if audit["errors"]:
            raise DatasetValidationError("; ".join(audit["errors"]))
        options["quantize"] = 8
        options["data"] = str(args.dataset.resolve())
        options["fraction"] = args.calibration_fraction
    if args.device is not None:
        options["device"] = args.device
    exported = model.export(**options)
    print(f"Exported model: {exported}")


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ultralytics dataset YAML.",
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", help="Ultralytics device selector, for example cpu, 0, or 0,1."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, train, evaluate, and deploy a YOLO road-damage detector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles_parser = subparsers.add_parser(
        "profiles", help="Print the reproducible training profiles."
    )
    profiles_parser.set_defaults(handler=lambda _: print_profiles())

    prepare_parser = subparsers.add_parser(
        "prepare", help="Convert COCO annotations into a YOLO dataset."
    )
    prepare_parser.add_argument("--annotations", type=Path, required=True)
    prepare_parser.add_argument("--images-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=17)
    prepare_parser.add_argument("--train-ratio", type=float, default=0.7)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.15)
    prepare_parser.add_argument("--test-ratio", type=float, default=0.15)
    prepare_parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply deterministic RGB autocontrast to converted images.",
    )
    prepare_parser.add_argument(
        "--median-filter-size",
        type=int,
        default=0,
        help="Apply an odd-sized median denoising filter; zero disables it.",
    )
    prepare_parser.set_defaults(handler=run_prepare)

    validate_parser = subparsers.add_parser(
        "validate-data", help="Audit labels, image integrity, and split leakage."
    )
    validate_parser.add_argument("--dataset-root", type=Path, required=True)
    validate_parser.add_argument(
        "--skip-hashes", action="store_true", help="Skip exact-duplicate detection."
    )
    validate_parser.set_defaults(handler=run_validation)

    train_parser = subparsers.add_parser("train", help="Fine-tune a YOLO detector.")
    _add_dataset_argument(train_parser)
    _add_device_argument(train_parser)
    train_parser.add_argument(
        "--profile", choices=sorted(EXPERIMENT_PROFILES), default="sgd-30"
    )
    train_parser.add_argument("--model")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--image-size", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--optimizer", choices=("SGD", "AdamW"))
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--momentum", type=float)
    train_parser.add_argument("--weight-decay", type=float)
    train_parser.add_argument("--mosaic", type=float)
    train_parser.add_argument("--mixup", type=float)
    train_parser.add_argument("--hsv-h", type=float)
    train_parser.add_argument("--hsv-s", type=float)
    train_parser.add_argument("--hsv-v", type=float)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--close-mosaic", type=int, default=10)
    train_parser.add_argument("--fraction", type=float)
    train_parser.add_argument("--freeze", type=int)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--project", type=Path, default=DEFAULT_ARTIFACTS / "runs"
    )
    train_parser.add_argument("--name")
    train_parser.add_argument("--cosine-schedule", action="store_true")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--non-deterministic", action="store_true")
    train_parser.add_argument("--skip-data-validation", action="store_true")
    train_parser.add_argument("--exist-ok", action="store_true")
    train_parser.set_defaults(handler=run_training)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint and generate diagnostic plots."
    )
    _add_dataset_argument(evaluate_parser)
    _add_device_argument(evaluate_parser)
    evaluate_parser.add_argument("--weights", type=Path, required=True)
    evaluate_parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test"
    )
    evaluate_parser.add_argument("--image-size", type=int, default=640)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--confidence", type=float, default=0.001)
    evaluate_parser.add_argument("--iou", type=float, default=0.7)
    evaluate_parser.add_argument(
        "--project", type=Path, default=DEFAULT_ARTIFACTS / "evaluation"
    )
    evaluate_parser.add_argument("--name", default="test")
    evaluate_parser.add_argument("--save-json", action="store_true")
    evaluate_parser.add_argument("--skip-data-validation", action="store_true")
    evaluate_parser.add_argument("--exist-ok", action="store_true")
    evaluate_parser.set_defaults(handler=run_evaluation)

    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on images, directories, videos, or streams."
    )
    _add_device_argument(predict_parser)
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--image-size", type=int, default=640)
    predict_parser.add_argument("--confidence", type=float, default=0.25)
    predict_parser.add_argument("--iou", type=float, default=0.7)
    predict_parser.add_argument(
        "--project", type=Path, default=DEFAULT_ARTIFACTS / "predictions"
    )
    predict_parser.add_argument("--name", default="inference")
    predict_parser.add_argument("--exist-ok", action="store_true")
    predict_parser.set_defaults(handler=run_prediction)

    export_parser = subparsers.add_parser(
        "export", help="Export a checkpoint for an inference runtime."
    )
    _add_dataset_argument(export_parser)
    _add_device_argument(export_parser)
    export_parser.add_argument("--weights", type=Path, required=True)
    export_parser.add_argument(
        "--format",
        choices=("onnx", "torchscript", "openvino", "engine", "coreml", "tflite"),
        default="onnx",
    )
    export_parser.add_argument(
        "--precision", choices=("fp32", "fp16", "int8"), default="fp32"
    )
    export_parser.add_argument("--image-size", type=int, default=640)
    export_parser.add_argument("--calibration-fraction", type=float, default=1.0)
    export_parser.add_argument("--dynamic", action="store_true")
    export_parser.set_defaults(handler=run_export)
    return parser.parse_args(argv)


def print_profiles() -> None:
    payload = {
        name: asdict(profile) for name, profile in sorted(EXPERIMENT_PROFILES.items())
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_prepare(args: argparse.Namespace) -> None:
    manifest = prepare_coco_dataset(
        annotations=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        enhance_contrast=args.enhance_contrast,
        median_filter_size=args.median_filter_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def run_validation(args: argparse.Namespace) -> None:
    audit = validate_dataset(args.dataset_root, check_hashes=not args.skip_hashes)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["errors"]:
        raise DatasetValidationError("Dataset validation failed.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        args.handler(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
