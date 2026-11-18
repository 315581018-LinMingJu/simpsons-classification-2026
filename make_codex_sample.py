"""抽取原始圖片樣本，產生可上傳給 Codex 的小型 ZIP。"""

import argparse
import csv
import io
import json
import os
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from v2.data_checks import CLASS_COUNT, TEST_COUNT

REPO = Path(__file__).resolve().parent
MIB = 1024 * 1024
TARGET_BYTES = 25 * MIB
HARD_LIMIT_BYTES = 30 * MIB
QUOTAS = (5, 144, 96)
RANDOM = "random"
CHANGED = "v3_multiscale_changed"
IMAGE_ERRORS = (OSError, SyntaxError, ValueError, Image.DecompressionBombError)


@dataclass(frozen=True)
class Sample:
    path: Path
    source: str
    method: str
    label: str = ""


@dataclass(frozen=True)
class ArchiveImage:
    sample: Sample
    actual_format: str
    extension: str
    size: int

    @property
    def archive_path(self):
        sample = self.sample
        directory = sample.label if sample.source == "train" else sample.method
        return f"{sample.source}/{directory}/{sample.path.stem}{self.extension}"


def jpg_paths(directory):
    return sorted(path for path in directory.glob("*.jpg") if path.is_file())


def detect_format(path):
    with Image.open(path) as image:
        if image.format is None:
            raise ValueError("Pillow did not identify an image format")
        return image.format


def image_format_stats(data):
    extensions = Image.registered_extensions()
    result = {}
    for split, root, pattern in (
        ("train", data / "train/train/train", "*/*"),
        ("test", data / "test_images/test_images", "*"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(root)
        counts = Counter()
        unreadable, mismatches, examples = 0, 0, []
        for path in root.glob(pattern):
            if path.suffix.lower() not in extensions or not path.is_file():
                continue
            try:
                actual = detect_format(path)
            except IMAGE_ERRORS:
                unreadable += 1
                if len(examples) < 10:
                    examples.append(path.relative_to(data).as_posix())
                continue
            counts[actual] += 1
            mismatches += actual != extensions[path.suffix.lower()]
        detected = sum(counts.values())
        result[split] = {
            "total_files": detected + unreadable,
            "detected_files": detected,
            "format_counts": dict(sorted(counts.items())),
            "format_pct_of_detected": {fmt: round(100 * n / detected, 4)
                                       for fmt, n in sorted(counts.items())},
            "extension_mismatches": mismatches,
            "unreadable_headers": unreadable,
            "unreadable_examples": sorted(examples),
        }
    return result


def changed_test_ids(path, test_ids, classes):
    by_method = {"baseline": {}, "multiscale": {}}
    flags = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "method", "character", "changed_vs_v2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing expected columns in {path}")
        for row in reader:
            method = row["method"]
            if method not in by_method:
                continue
            item, label = row["id"], row["character"]
            if item not in test_ids or item in by_method[method] or label not in classes:
                raise ValueError(f"Unexpected ID, duplicate ID or invalid role in V3 {method}: {item!r}")
            by_method[method][item] = label
            if method == "multiscale":
                flags[item] = row["changed_vs_v2"]
    baseline, multiscale = by_method["baseline"], by_method["multiscale"]
    if baseline.keys() != test_ids or multiscale.keys() != test_ids:
        raise ValueError("Use the full V3 test_diagnostics.csv covering all test IDs")
    changed = {item for item in test_ids if baseline[item] != multiscale[item]}
    if any(flags[item] != ("1" if item in changed else "0") for item in test_ids):
        raise ValueError("V3 changed_vs_v2 flags disagree with Baseline and Multi-scale predictions")
    return changed


def sample_pools(data, v3_run, seed):
    train_root = data / "train/train/train"
    classes = sorted(path for path in train_root.iterdir() if path.is_dir())
    if len(classes) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} training class folders, found {len(classes)}")
    rng = random.Random(seed)
    train = {}
    for folder in classes:
        paths = jpg_paths(folder)
        if not paths:
            raise ValueError(f"No training .jpg files in {folder}")
        train[folder.name] = [Sample(path, "train", RANDOM, folder.name)
                              for path in rng.sample(paths, min(QUOTAS[0], len(paths)))]

    test = {path.stem: path for path in jpg_paths(data / "test_images/test_images")}
    if len(test) != TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} test .jpg files, found {len(test)}")
    changed = changed_test_ids(v3_run / "test_diagnostics.csv", test.keys(), train.keys())
    random_ids = rng.sample(sorted(test), min(QUOTAS[1], len(test) - 1))
    eligible = sorted(changed - set(random_ids))
    if not eligible:
        raise ValueError("No V3 disagreements remain outside the random sample")
    hard_ids = rng.sample(eligible, min(QUOTAS[2], len(eligible)))
    return (
        train,
        [Sample(test[item], "test", RANDOM) for item in random_ids],
        [Sample(test[item], "test", CHANGED) for item in hard_ids],
        {"test_pool": len(test), "v3_disagreement_pool": len(changed),
         "v3_disagreement_pool_excluding_random_draw": len(eligible)},
    )


def metadata_files(images, seed, counts):
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(("source", "sampling_method", "original_id", "known_character", "archive_path",
                     "original_filename", "actual_format"))
    samples = [image.sample for image in images]
    for image in images:
        sample = image.sample
        writer.writerow((sample.source, sample.method, sample.path.stem, sample.label,
                         image.archive_path, sample.path.name, image.actual_format))
    info = {
        "seed": seed,
        "requested": dict(zip(("train_per_class", "random_test", "v3_disagreements"), QUOTAS)),
        "actual": {
            "train_by_class": dict(Counter(sample.label for sample in samples if sample.source == "train")),
            "random_test": sum(sample.source == "test" and sample.method == RANDOM for sample in samples),
            "v3_disagreements": sum(sample.method == CHANGED for sample in samples),
        },
        **counts,
    }
    return {
        "manifest.csv": manifest.getvalue().encode("utf-8"),
        "sampling.json": (json.dumps(info, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    }


def plan_archive(train, random_test, hard_test, seed, counts, target, hard_limit):
    extensions = {fmt: ext for ext, fmt in sorted(Image.registered_extensions().items(), reverse=True)}
    extensions.update({"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff", "JPEG2000": ".jp2"})
    images = {}
    for group in (*train.values(), random_test, hard_test):
        for sample in group:
            try:
                actual_format = detect_format(sample.path)
            except IMAGE_ERRORS as exc:
                raise ValueError(f"Cannot identify selected image: {sample.path}: {exc}") from exc
            images[sample] = ArchiveImage(sample, actual_format,
                                          extensions.get(actual_format, sample.path.suffix),
                                          sample.path.stat().st_size)
    # 只縮減各組前綴配額，圖片的抽樣順序與單張檔案大小無關。
    steps = max(QUOTAS)
    for scale in range(steps, 0, -1):
        per_class, n_random, n_hard = ((requested * scale + steps - 1) // steps for requested in QUOTAS)
        samples = ([sample for group in train.values() for sample in group[:per_class]]
                   + random_test[:n_random] + hard_test[:n_hard])
        selected = [images[sample] for sample in samples]
        metadata = metadata_files(selected, seed, counts)
        # ZIP_STORED 不用 ZIP64。每筆有 30+46 bytes header 和兩份檔名，結尾 22 bytes。
        entries = [(image.archive_path, image.size) for image in selected]
        entries.extend((name, len(contents)) for name, contents in metadata.items())
        archive_size = 22 + sum(size + 76 + 2 * len(name.encode("utf-8")) for name, size in entries)
        if archive_size <= target and archive_size < hard_limit:
            return selected, metadata, archive_size
    if archive_size < hard_limit:
        return selected, metadata, archive_size
    raise ValueError("Even one image per class plus both test cohorts exceeds the ZIP limit")


def write_archive(output, images, metadata, planned_size, hard_limit):
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
                for entry in images:
                    path = entry.sample.path
                    contents = path.read_bytes()
                    try:
                        with Image.open(io.BytesIO(contents)) as image:
                            if image.format != entry.actual_format or len(contents) != entry.size:
                                raise ValueError("Image format or byte length changed after planning")
                            image.verify()
                        # verify() 不保證每個解碼器都讀完整張圖片，例如 JPEG。
                        with Image.open(io.BytesIO(contents)) as image:
                            image.load()
                    except IMAGE_ERRORS as exc:
                        raise ValueError(f"Cannot verify or decode selected image: {path}: {exc}") from exc
                    archive.writestr(zipfile.ZipInfo(entry.archive_path), contents)
                for name, contents in metadata.items():
                    archive.writestr(zipfile.ZipInfo(name), contents)
        size = temporary.stat().st_size
        if size >= hard_limit or size != planned_size:
            raise ValueError("ZIP byte count differs from the approved plan")
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()


def create_archive(data, v3_run, output, seed):
    output = Path(os.path.abspath(output))
    if (output.suffix.lower() != ".zip" or output.is_relative_to(REPO)
            or output.resolve().is_relative_to(REPO)):
        raise ValueError("Choose a .zip output outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    train, random_test, hard_test, counts = sample_pools(data, v3_run, seed)
    images, metadata, size = plan_archive(
        train, random_test, hard_test, seed, counts, TARGET_BYTES, HARD_LIMIT_BYTES,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_archive(output, images, metadata, size, HARD_LIMIT_BYTES)
    print(f"ZIP: {output} ({len(images)} images, {size / MIB:.2f} MiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/extracted")
    parser.add_argument("--v3-run", type=Path, default=REPO / "runs/v3_tta_155")
    parser.add_argument("--output", type=Path, default=REPO.parent / "simpsons_codex_sample.zip")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--format-stats", action="store_true", help="只讀圖片檔頭，輸出 Train／Test 格式統計")
    args = parser.parse_args()
    if args.format_stats:
        print(json.dumps(image_format_stats(args.data), indent=2, ensure_ascii=False))
    else:
        create_archive(args.data, args.v3_run, args.output, args.seed)


if __name__ == "__main__":
    main()
