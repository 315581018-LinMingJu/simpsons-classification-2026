"""固定配對驗證，強化訓練只使用競賽 Train 圖片。"""

import csv
import io
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shutil import disk_usage

import numpy as np
from PIL import Image

from compare_v1_v2 import load_comparison_split
from ensemble.data import fresh_validation
from v2.data_checks import TEST_COUNT, read_json
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.geometry import seeded
from v5.challenge import scene
from v6.composite import Targets, composite
from v7.impulse import add_impulses, clean_impulses
from v8.models import RULES_URL, SOURCES, VISION_LICENSE, VISION_SOURCE, head_status

CONDITIONS = ("clean", "context", "impulse", "clipped", "distractor", "negative")
# 讓罕見反色的影響低於主要場景族群；這些是預先固定的壓力測試權重。
HARD_WEIGHTS = {"context": 0.30, "impulse": 0.25, "clipped": 0.20,
                "distractor": 0.20, "negative": 0.05}
FIELDS = ("image", "label", "fold", "distractor")


@dataclass(frozen=True)
class Case:
    item: str
    label: int
    fold: str
    distractor: str = ""


def environment():
    packages = {}
    for name in ("torch", "torchvision", "Pillow", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages,
            "free_gib": round(disk_usage(Path.cwd()).free / 2**30, 2)}


def selected_signature(root, cases):
    paths = {c.item for c in cases} | {c.distractor for c in cases if c.distractor}
    return image_signature(root / path for path in sorted(paths))


def train_cases(rows, root, limit, seed):
    buckets = defaultdict(list)
    for row in rows:
        if row.part == "train":
            buckets[row.label].append(row.path.relative_to(root).as_posix())
    cases = []
    for label, paths in sorted(buckets.items()):
        paths = sorted(paths)
        if limit:
            paths = seeded(seed, f"v8:train:{label}").sample(paths, min(limit, len(paths)))
        for item in paths:
            cases.append(Case(item, label, "train"))
    return cases


def read_cases(path, root, classes, folds):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        if next(reader) != list(FIELDS):
            raise ValueError(f"Invalid benchmark columns: {path}")
        cases = [Case(item, int(label), fold, other) for item, label, fold, other in reader]
    seen = set()
    for case in cases:
        item = Path(case.item)
        if (len(item.parts) != 2 or ".." in item.parts or item.is_absolute() or case.item in seen
                or not 0 <= case.label < len(classes) or item.parts[0] != classes[case.label]
                or case.fold not in folds or not (root / item).is_file()):
            raise ValueError(f"Invalid benchmark image: {case}")
        seen.add(case.item)
        if case.fold == "train":
            if case.distractor:
                raise ValueError("Unexpected training distractor in the fixed manifest")
            continue
        other = Path(case.distractor)
        if (len(other.parts) != 2 or other.is_absolute() or other.parts[0] not in classes
                or other.parts[0] == item.parts[0] or not (root / other).is_file()):
            raise ValueError(f"Invalid validation distractor: {case}")
    for fold in folds:
        if {case.label for case in cases if case.fold == fold} != set(range(len(classes))):
            raise ValueError(f"Missing classes in benchmark: {fold}")
    return cases


def prepare(args):
    directory = args.output / "benchmark"
    root = args.data / "train/train/train"
    classes, rows, split_hash = load_comparison_split(root, args.v1_run, args.v2_run)
    old = read_json(args.v5_run / "settings.json")
    audit = read_json(args.audit)
    if (old["classes"] != classes or old["split_sha256"] != split_hash
            or old["train_image_signature"] != image_signature(row.path for row in rows)
            or audit["provenance"]["split_sha256"] != split_hash
            or audit["labels"]["allowed_characters"] != classes
            or audit["labels"]["test_ground_truth_available"] is not False
            or audit["train"]["decoded_images"] != len(rows) or audit["test"]["decoded_images"] != TEST_COUNT):
        raise ValueError("V5 duplicate metadata or distribution audit differs from the original input")
    paths = (args.v5_run / "validation_subset.json", args.v5_run / "validation_samples.csv", args.audit)
    inputs = {
        "seed": args.seed, "train_per_class": args.train_per_class, "validation_per_class": args.per_class,
        "classes": classes, "split_sha256": split_hash, "train_image_signature": old["train_image_signature"],
        "source_sha256": [file_sha256(path) for path in paths], "hard_weights": HARD_WEIGHTS,
        "targets": asdict(Targets.from_summary(audit)),
    }
    report_path = directory / "prepare.json"
    if report_path.exists():
        saved, _, _ = benchmark(args)
        if saved["inputs"] != inputs:
            raise ValueError("Prepared inputs changed. Use a new V8 --output directory.")
        print(f"沿用 {report_path}", flush=True)
        return
    samples, labels, folds, distractors, exclusions = fresh_validation(
        rows, root, args.v5_run, classes, args.per_class, args.seed,
    )
    training = train_cases(rows, root, args.train_per_class, args.seed)
    validation = [
        Case(item, int(label), "confirm" if fold == "confirmation" else "screen",
             distractors[item].path.relative_to(root).as_posix())
        for (_, item), label, fold in zip(samples, labels, folds, strict=True)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, cases in (("train", training), ("validation", validation)):
        path = directory / (name + ".csv")
        publish_csv(path, FIELDS, ((c.item, c.label, c.fold, c.distractor) for c in cases))
        manifests[name] = file_sha256(path)
    status = {name: head_status(source.url) for name, source in SOURCES.items()}
    status["competition_rules"] = head_status(RULES_URL)
    report = {
        "inputs": inputs, "manifest_sha256": manifests, "exclusions": exclusions,
        "selected_image_signature": selected_signature(root, training + validation),
        "counts": {"train": len(training), "screen": sum(c.fold == "screen" for c in validation),
                   "confirm": sum(c.fold == "confirm" for c in validation)},
        "rules": {"url": RULES_URL, "authorization_checked_by_operator": False},
        "torchvision_code": VISION_SOURCE, "torchvision_code_license": VISION_LICENSE,
        "weights": {name: {**asdict(src), "url": src.url, "data": "ImageNet-1K"}
                    for name, src in SOURCES.items()},
        "endpoint_status": status, "environment": environment(),
    }
    publish(report_path, json_bytes(report))
    print(f"{report_path}  {report['counts']}  {status}", flush=True)


def benchmark(args):
    directory = args.output / "benchmark"
    root = args.data / "train/train/train"
    report = read_json(directory / "prepare.json")
    inputs = report["inputs"]
    if inputs["hard_weights"] != HARD_WEIGHTS:
        raise ValueError("Benchmark uses different hard-condition weights")
    cases = []
    for name, folds in (("train", ("train",)), ("validation", ("screen", "confirm"))):
        path = directory / (name + ".csv")
        if file_sha256(path) != report["manifest_sha256"][name]:
            raise ValueError(f"Benchmark manifest changed: {path}")
        cases.append(read_cases(path, root, inputs["classes"], folds))
    if {c.item for c in cases[0]} & {c.item for c in cases[1]}:
        raise ValueError("Train and validation benchmark IDs overlap")
    if selected_signature(root, cases[0] + cases[1]) != report["selected_image_signature"]:
        raise ValueError("Images changed after Phase A. Prepare a new V8 benchmark.")
    return report, *cases


def open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def stress(image, condition, seed, item, secondary=None):
    if condition == "clean":
        return image
    view, _ = scene(image, "context" if condition == "impulse" else condition, seed, item, secondary)
    return add_impulses(view, seed, item) if condition == "impulse" else view


class EvaluationImages:
    def __init__(self, root, cases, condition, views, seed, denoise):
        self.root, self.cases, self.condition = root, cases, condition
        self.views, self.seed, self.denoise = views, seed, denoise

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = open_rgb(self.root / case.item)
        other = open_rgb(self.root / case.distractor) if self.condition == "distractor" else None
        image = stress(image, self.condition, self.seed, case.item, other)
        if self.denoise:
            image, _ = clean_impulses(image)
        return self.views(image), case.item


class TrainingImages:
    def __init__(self, root, cases, basic, targets, augmentation, seed):
        self.root, self.cases, self.basic = root, cases, basic
        self.targets, self.augmentation, self.seed = targets, augmentation, seed
        self.by_label = defaultdict(list)
        for c in cases:
            self.by_label[c.label].append(c.item)
        self.labels = sorted(self.by_label)
        if augmentation != "basic":
            from torchvision import transforms as T

            self.scene_transform = T.Compose([
                T.Resize((224, 224)), T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.16, contrast=0.18, saturation=0.18),
                *basic.transforms[4:],
            ])

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = open_rgb(self.root / case.item)
        if self.augmentation == "basic":
            return self.basic(image), case.label
        rng = seeded(self.seed, f"v8:train:{case.item}")
        if rng.random() >= 0.55:
            return self.basic(image), case.label
        use_distractor = rng.random() < 0.15
        if self.augmentation == "robust" and use_distractor:
            label = rng.choice([label for label in self.labels if label != case.label])
            other = open_rgb(self.root / rng.choice(self.by_label[label]))
            image = stress(image, "distractor", self.seed, case.item, other)
        else:
            image, _ = composite(image, self.targets, np.random.default_rng(rng.getrandbits(64)))
        if self.augmentation == "robust" and rng.random() < 0.10:
            image = add_impulses(image, self.seed, case.item)
        if self.augmentation == "robust" and rng.random() < 0.15:
            compressed = io.BytesIO()
            image.save(compressed, format="JPEG", quality=rng.randint(45, 85))
            with Image.open(compressed) as altered:
                image = altered.convert("RGB")
        return self.scene_transform(image), case.label
