"""Read and validate the original V1 artifacts without changing them."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

COUNTS = {"train": 77547, "val": 19382}
TEST_COUNT = 10791
CLASS_COUNT = 50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    part: str


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(train_root, classes, split_bytes):
    """Validate class indices and exact split coverage against the training tree."""
    if not isinstance(classes, list) or len(classes) != CLASS_COUNT or not all(
        isinstance(name, str) and name for name in classes
    ):
        raise ValueError("classes.json must contain the 50 ordered class names")
    if classes != sorted(p.name for p in train_root.iterdir() if p.is_dir()):
        raise ValueError("classes.json does not match ImageFolder's sorted class folders")
    images = {
        p.relative_to(train_root)
        for p in train_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {name: i for i, name in enumerate(classes)}
    rows = []
    seen = set()
    with io.StringIO(split_bytes.decode("utf-8-sig"), newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["path", "label", "part"]:
            raise ValueError("Expected split.csv columns path,label,part")
        for row in reader:
            relative = Path(row["path"])
            label = int(row["label"])
            if relative not in images or relative in seen:
                raise ValueError(f"Missing, invalid, or repeated split path: {relative}")
            if labels.get(relative.parts[0]) != label or row["part"] not in COUNTS:
                raise ValueError(f"Invalid label or partition for {relative}")
            seen.add(relative)
            rows.append(Sample(train_root / relative, label, row["part"]))
    if seen != images or Counter(row.part for row in rows) != COUNTS:
        raise ValueError("Split does not cover the original dataset and partition counts")
    for part in COUNTS:
        if len({row.label for row in rows if row.part == part}) != CLASS_COUNT:
            raise ValueError(f"Missing class in {part}")

    return rows


def load_inputs(data, v1, template):
    train_root = data / "train/train/train"
    test_root = data / "test_images/test_images"
    classes = read_json(v1 / "classes.json")
    config = read_json(v1 / "run_config.json")
    history = read_json(v1 / "history.json")
    if not isinstance(config, dict) or not isinstance(history, list) or not history:
        raise ValueError("Expected V1 configuration and a nonempty V1 epoch history")
    expected = {
        "architecture": "resnet18", "optimizer": "AdamW", "lr": 0.0001,
        "weight_decay": 0.0001,
        "train_transform": "RandomResizedCrop224+RandomHorizontalFlip",
        "eval_transform": "Resize256+CenterCrop224",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"V1 {name} differs from the attached source: {config.get(name)!r}")
    for name in ("seed", "epochs", "batch_size"):
        if type(config.get(name)) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid V1 config value for {name}")
    if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError("V1 history does not cover the configured epochs in order")
    if any(not 0 <= row["val_accuracy"] <= 1 or not row["seconds"] >= 0 for row in history):
        raise ValueError("Invalid V1 accuracy or epoch duration")

    split_bytes = (v1 / "split.csv").read_bytes()
    rows = load_split(train_root, classes, split_bytes)

    with template.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "character"]:
            raise ValueError("Expected template columns id,character")
        ids = [row["id"] for row in reader]
    if len(ids) != TEST_COUNT or len(set(ids)) != TEST_COUNT or any(
        not item or Path(item).name != item or item in {".", ".."} for item in ids
    ):
        raise ValueError("Template must contain 10,791 distinct, nonempty filename IDs")
    test_paths = {p.stem: p for p in test_root.glob("*.jpg") if p.is_file()}
    if test_paths.keys() != set(ids):
        raise ValueError(
            f"Test directory differs from template: {len(set(ids) - test_paths.keys())} missing, "
            f"{len(test_paths.keys() - set(ids))} extra"
        )
    return classes, config, history, rows, [(test_paths[item], item) for item in ids], split_bytes


def duplicate_diagnostic(rows):
    """Group exact decoded RGB pixels, including dimensions in the signature."""
    from PIL import Image

    groups = defaultdict(list)
    for row in rows:
        with Image.open(row.path) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
            digest.update(rgb.tobytes())
        groups[digest.hexdigest()].append(row)

    overlaps = []
    conflicted_groups = 0
    train_paths, val_paths = set(), set()
    pairs = 0
    for digest, group in sorted(groups.items()):
        if len(group) > 1 and len({row.label for row in group}) > 1:
            conflicted_groups += 1
        train = [str(row.path) for row in group if row.part == "train"]
        val = [str(row.path) for row in group if row.part == "val"]
        if train and val:
            train_paths.update(train)
            val_paths.update(val)
            pairs += len(train) * len(val)
            overlaps.append({"sha256_rgb": digest, "train": train, "val": val,
                             "labels": sorted({row.label for row in group})})
    return {
        "cross_partition_groups": len(overlaps),
        "cross_partition_pairs": pairs,
        "cross_partition_train_images": len(train_paths),
        "cross_partition_val_images": len(val_paths),
        "duplicate_groups_with_multiple_labels": conflicted_groups,
        "groups": overlaps,
    }, train_paths, val_paths
