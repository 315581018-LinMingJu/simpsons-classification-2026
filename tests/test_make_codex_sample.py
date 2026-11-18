import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import make_codex_sample as sampler


class SampleArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data"
        self.v3 = self.root / "v3"
        self.v3.mkdir()
        self.report = self.v3 / "test_diagnostics.csv"
        self.classes = ("homer_simpson", "瑪姬")
        self.train = self.data / "train/train/train"
        for label in self.classes:
            class_dir = self.train / label
            class_dir.mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (16, 16), (i * 30, 0, 0)).save(class_dir / f"frame_{i}.jpg")
        self.test = self.data / "test_images/test_images"
        self.test.mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (13 + i, 22), (i * 30, 0, 0)).save(self.test / f"{i:03}.jpg")
        self.changed = {"000", "001", "002", "003"}
        with self.report.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "method", "character", "changed_vs_v2"))
            for i in range(8):
                item = f"{i:03}"
                changed = item in self.changed
                writer.writerow((item, "baseline", self.classes[0], 0))
                writer.writerow((item, "multiscale", self.classes[int(changed)], int(changed)))
        counts = patch.multiple(sampler, CLASS_COUNT=2, TEST_COUNT=8, QUOTAS=(3, 3, 3))
        counts.start()
        self.addCleanup(counts.stop)

    def test_integration_preserves_original_jpegs_and_id_cohorts(self):
        output = self.root / "sample.zip"
        alternative = self.root / "same_sample.zip"
        before = {path: path.read_bytes() for path in self.data.rglob("*.jpg")}
        sampler.create_archive(self.data, self.v3, output, 2026)
        sampler.create_archive(self.data, self.v3, alternative, 2026)
        self.assertEqual(output.read_bytes(), alternative.read_bytes())
        self.assertLessEqual(output.stat().st_size, sampler.TARGET_BYTES)
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            rows = list(csv.DictReader(io.StringIO(archive.read("manifest.csv").decode())))
            info = json.loads(archive.read("sampling.json"))
            self.assertEqual(info["actual"]["train_by_class"], dict.fromkeys(self.classes, 3))
            self.assertEqual(info["actual"]["random_test"], 3)
            self.assertLessEqual(info["actual"]["v3_disagreements"], 3)
            self.assertEqual(len(archive.namelist()), len(rows) + 2)
            random_test, hard_test = set(), set()
            for row in rows:
                name = row["archive_path"]
                if row["source"] == "train":
                    label = row["known_character"]
                    self.assertIn(label, self.classes)
                    original = self.train / label / (row["original_id"] + ".jpg")
                    self.assertTrue(name.startswith(f"train/{label}/"))
                else:
                    self.assertEqual(row["known_character"], "")
                    item = row["original_id"]
                    original = self.test / (item + ".jpg")
                    cohort = random_test if row["sampling_method"] == sampler.RANDOM else hard_test
                    cohort.add(item)
                self.assertEqual(archive.read(name), before[original])
            self.assertTrue(hard_test)
            self.assertTrue(hard_test <= self.changed)
            self.assertFalse(random_test & hard_test)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        with self.assertRaises(FileExistsError):
            sampler.create_archive(self.data, self.v3, output, 2026)

    def test_budget_trims_to_balanced_prefix_and_enforces_strict_limit(self):
        train, random_test, hard_test, counts = sampler.sample_pools(self.data, self.v3, 2026)
        full = sampler.plan_archive(train, random_test, hard_test, 2026, counts, 10**6, 10**6)
        minimum = sampler.plan_archive(train, random_test, hard_test, 2026, counts, 1, 10**6)
        samples, manifest, size = minimum
        self.assertLess(size, full[2])
        self.assertEqual(len(samples), len(self.classes) + 2)
        self.assertEqual(samples[:len(self.classes)], [group[0] for group in train.values()])
        self.assertEqual(samples[-2:], [random_test[0], hard_test[0]])
        accepted = sampler.plan_archive(train, random_test, hard_test, 2026, counts, size, size + 1)
        self.assertEqual(accepted, minimum)
        output = self.root / "trimmed.zip"
        sampler.write_archive(output, samples, manifest, size, size + 1)
        self.assertEqual(output.stat().st_size, size)
        with self.assertRaisesRegex(ValueError, "exceeds the ZIP limit"):
            sampler.plan_archive(train, random_test, hard_test, 2026, counts, size, size)

    def test_full_v3_report_and_consistent_flags_are_required(self):
        original = self.report.read_text().splitlines()
        for lines, error in (
            (original[:-1], "full V3"),
            (original + [original[-1]], "duplicate ID"),
            ([*original[:-1], original[-1][:-1] + "1"], "flags disagree"),
        ):
            with self.subTest(error=error):
                self.report.write_text("\n".join(lines) + "\n")
                with self.assertRaisesRegex(ValueError, error):
                    sampler.sample_pools(self.data, self.v3, 2026)

    def test_rejects_repo_path_and_does_not_publish_invalid_archive(self):
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            sampler.create_archive(self.data, self.v3, sampler.REPO / "sample.zip", 2026)
        train, random_test, hard_test, counts = sampler.sample_pools(self.data, self.v3, 2026)
        samples, metadata, size = sampler.plan_archive(
            train, random_test, hard_test, 2026, counts, 10**6, 10**6,
        )
        samples[0].path.write_bytes(b"not an image")
        output = self.root / "invalid.zip"
        with self.assertRaisesRegex(ValueError, "not a JPEG"):
            sampler.write_archive(output, samples, metadata, size, 10**6)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))
