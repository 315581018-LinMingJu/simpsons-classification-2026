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
        self.assertEqual([entry.sample for entry in samples[:len(self.classes)]],
                         [group[0] for group in train.values()])
        self.assertEqual([entry.sample for entry in samples[-2:]], [random_test[0], hard_test[0]])
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

    def test_renames_mislabeled_png_and_bmp_without_reencoding(self):
        train, random_test, _, _ = sampler.sample_pools(self.data, self.v3, 2026)
        png = random_test[0].path
        bmp = train[self.classes[0]][0].path
        Image.new("RGB", (28, 26), "red").save(png, format="PNG")
        Image.new("RGB", (18, 16), "blue").save(bmp, format="BMP")
        originals = {png: png.read_bytes(), bmp: bmp.read_bytes()}
        output = self.root / "mixed.zip"
        sampler.create_archive(self.data, self.v3, output, 2026)
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            rows = list(csv.DictReader(io.StringIO(archive.read("manifest.csv").decode())))
            for path, actual, extension in ((png, "PNG", ".png"), (bmp, "BMP", ".bmp")):
                row = next(row for row in rows if row["original_filename"] == path.name
                           and row["actual_format"] == actual)
                self.assertEqual(row["original_id"], path.stem)
                self.assertEqual(Path(row["archive_path"]).suffix, extension)
                self.assertEqual(archive.read(row["archive_path"]), originals[path])
                self.assertEqual(path.read_bytes(), originals[path])
                with Image.open(io.BytesIO(archive.read(row["archive_path"]))) as image:
                    image.verify()
                if path == png:
                    self.assertEqual(row["known_character"], "")
                    self.assertEqual(row["sampling_method"], sampler.RANDOM)
                else:
                    self.assertEqual(row["known_character"], self.classes[0])

    def test_rejects_unidentified_image_and_repo_output(self):
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            sampler.create_archive(self.data, self.v3, sampler.REPO / "sample.zip", 2026)
        train, _, _, _ = sampler.sample_pools(self.data, self.v3, 2026)
        bad = train[self.classes[0]][0].path
        bad.write_bytes(b"not an image")
        output = self.root / "invalid.zip"
        with self.assertRaisesRegex(ValueError, "Cannot identify selected image: .*frame_"):
            sampler.create_archive(self.data, self.v3, output, 2026)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_rejects_image_with_readable_header_but_truncated_pixels(self):
        train, _, _, _ = sampler.sample_pools(self.data, self.v3, 2026)
        bad = train[self.classes[0]][0].path
        bad.write_bytes(bad.read_bytes()[:-4])
        with Image.open(bad) as image:
            self.assertEqual(image.format, "JPEG")
            image.verify()
        output = self.root / "truncated.zip"
        with self.assertRaisesRegex(ValueError, "Cannot verify or decode selected image: .*frame_"):
            sampler.create_archive(self.data, self.v3, output, 2026)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_header_statistics_count_actual_formats_and_unknown_headers(self):
        Image.new("RGB", (16, 16)).save(self.test / "000.jpg", format="PNG")
        (self.test / "001.jpg").write_bytes(b"broken")
        Image.new("RGB", (15, 15)).save(self.train / self.classes[0] / "native.png")
        stats = sampler.image_format_stats(self.data)
        self.assertEqual(stats["test"]["total_files"], 8)
        self.assertEqual(stats["test"]["format_counts"], {"JPEG": 6, "PNG": 1})
        self.assertEqual(stats["test"]["extension_mismatches"], 1)
        self.assertEqual(stats["test"]["format_pct_of_detected"]["PNG"], 14.2857)
        self.assertEqual(stats["test"]["unreadable_headers"], 1)
        self.assertEqual(stats["train"]["format_counts"], {"JPEG": 6, "PNG": 1})
        self.assertEqual(stats["train"]["extension_mismatches"], 0)
