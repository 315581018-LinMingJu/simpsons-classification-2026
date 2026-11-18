"""抽取原始 JPEG 樣本，產生可上傳給 Codex 的小型 ZIP。"""

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

from v2.data_checks import CLASS_COUNT, TEST_COUNT

REPO = Path(__file__).resolve().parent
MIB = 1024 * 1024
TARGET_BYTES = 25 * MIB
HARD_LIMIT_BYTES = 30 * MIB
QUOTAS = (5, 144, 96)
RANDOM = "random"
CHANGED = "v3_multiscale_changed"


@dataclass(frozen=True)
class Sample:
    path: Path
    source: str
    method: str
    label: str = ""

    @property
    def archive_path(self):
        directory = self.label if self.source == "train" else self.method
        return f"{self.source}/{directory}/{self.path.name}"


def jpeg_files(directory):
    return sorted(path for path in directory.glob("*.jpg") if path.is_file())


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
        paths = jpeg_files(folder)
        if not paths:
            raise ValueError(f"No training JPEGs in {folder}")
        train[folder.name] = [Sample(path, "train", RANDOM, folder.name)
                              for path in rng.sample(paths, min(QUOTAS[0], len(paths)))]

    test = {path.stem: path for path in jpeg_files(data / "test_images/test_images")}
    if len(test) != TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} test JPEGs, found {len(test)}")
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


def metadata_files(samples, seed, counts):
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(("source", "sampling_method", "original_id", "known_character", "archive_path"))
    for sample in samples:
        writer.writerow((sample.source, sample.method, sample.path.stem, sample.label,
                         sample.archive_path))
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
    sizes = {sample.path: sample.path.stat().st_size for group in (*train.values(), random_test, hard_test)
             for sample in group}
    # 只縮減各組前綴配額，圖片的抽樣順序與單張檔案大小無關。
    steps = max(QUOTAS)
    for scale in range(steps, 0, -1):
        per_class, n_random, n_hard = ((requested * scale + steps - 1) // steps for requested in QUOTAS)
        samples = ([sample for group in train.values() for sample in group[:per_class]]
                   + random_test[:n_random] + hard_test[:n_hard])
        metadata = metadata_files(samples, seed, counts)
        # ZIP_STORED 不用 ZIP64。每筆有 30+46 bytes header 和兩份檔名，結尾 22 bytes。
        entries = [(sample.archive_path, sizes[sample.path]) for sample in samples]
        entries.extend((name, len(contents)) for name, contents in metadata.items())
        archive_size = 22 + sum(size + 76 + 2 * len(name.encode("utf-8")) for name, size in entries)
        if archive_size <= target and archive_size < hard_limit:
            return samples, metadata, archive_size
    if archive_size < hard_limit:
        return samples, metadata, archive_size
    raise ValueError("Even one JPEG per class plus both test cohorts exceeds the ZIP limit")


def write_archive(output, samples, metadata, planned_size, hard_limit):
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
                for sample in samples:
                    contents = sample.path.read_bytes()
                    if not contents.startswith(b"\xff\xd8\xff"):
                        raise ValueError(f"Selected file is not a JPEG: {sample.path}")
                    archive.writestr(zipfile.ZipInfo(sample.archive_path), contents)
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
    samples, metadata, size = plan_archive(
        train, random_test, hard_test, seed, counts, TARGET_BYTES, HARD_LIMIT_BYTES,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_archive(output, samples, metadata, size, HARD_LIMIT_BYTES)
    print(f"ZIP: {output} ({len(samples)} JPEGs, {size / MIB:.2f} MiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/extracted")
    parser.add_argument("--v3-run", type=Path, default=REPO / "runs/v3_tta_155")
    parser.add_argument("--output", type=Path, default=REPO.parent / "simpsons_codex_sample.zip")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    create_archive(args.data, args.v3_run, args.output, args.seed)


if __name__ == "__main__":
    main()
