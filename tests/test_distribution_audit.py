import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from audit_distribution import audit, collect, split_summary
from distribution_audit.compare import grouped_changes, mark_novelty
from distribution_audit.features import inspect_image
from v3.reports import read_test_images


class DistributionAuditTests(unittest.TestCase):
    def test_pixels_and_header_are_separate_from_extension_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = Image.new("RGB", (24, 32), "black")
            ImageDraw.Draw(frame).rectangle((3, 4, 20, 27), fill="orange")
            mislabeled = root / "01.jpg"
            frame.save(mislabeled, format="PNG")
            corrupt = root / "02.jpg"
            corrupt.write_bytes(b"bad image")
            readings = collect([mislabeled, corrupt], workers=2, split="fixture")
            self.assertEqual(readings[0].actual_format, "PNG")
            self.assertFalse(readings[0].extension_match)
            self.assertEqual(readings[0].values[0], 1)
            self.assertEqual((readings[0].width, readings[0].height), (24, 32))
            self.assertGreater(readings[0].values[4], 0)
            self.assertIsNone(readings[1].values)
            report = split_summary(readings)
            self.assertEqual(report["decode_or_header_errors"], 1)
            self.assertEqual(report["formats_by_readable_header"], {"PNG": 1})
            self.assertEqual(report["extension_mismatch_pct"], 100)

    def test_unseen_format_does_not_inflate_visual_novelty_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (24, 32), "black")
            image.save(root / "train.jpg")
            image.save(root / "same.png")
            Image.new("RGB", (24, 32), "red").save(root / "shift.png")
            train = [inspect_image(root / "train.jpg")]
            test = [(name, inspect_image(root / f"{name}.png")) for name in ("same", "shift")]
            rows, _, bounds = mark_novelty(train, test)
            self.assertEqual(bounds["bins_from_train_p20_p80"]["brightness"], [0, 0])
            self.assertTrue(rows[0]["unseen_format"])
            self.assertEqual(rows[0]["visual_tail_distance"], 0)
            self.assertFalse(rows[0]["unseen_size"])
            self.assertEqual(rows[0]["combo_support"], "observed")
            self.assertGreater(rows[1]["visual_tail_distance"], 0)
            self.assertEqual(rows[1]["combo_support"], "unseen")

    def test_disagreement_counts_use_all_test_ids_as_denominator(self):
        rows = grouped_changes([("01", {"visual_band": "upper_tail"}),
                                ("02", {"visual_band": "inside_reference"}),
                                ("03", {"visual_band": "inside_reference"}),
                                ("04", {"visual_band": "unknown"})], {
                                    "01": True, "02": False, "03": False, "04": True,
                                })
        self.assertIn(("visual_band", "upper_tail", 1, 1, 100, 50, 2), rows)
        self.assertIn(("visual_band", "unknown", 1, 1, 100, 50, 2), rows)

    def test_cpu_report_uses_validated_folders_and_real_predictions_only_as_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            train = data / "train/train/train"
            test = data / "test_images/test_images"
            test.mkdir(parents=True)
            names = ["abraham", "bart"]
            lines = ["path,label,part"]
            for i, name in enumerate(names):
                (train / name).mkdir(parents=True)
                for label in ("train", "val"):
                    path = train / name / f"{label}.jpg"
                    image = Image.new("RGB", (32, 32), "black")
                    ImageDraw.Draw(image).rectangle((5, 4, 25, 23), fill=("orange", "blue")[i])
                    image.save(path)
                    lines.append(f"{name}/{label}.jpg,{i},{label}")
            split = ("\n".join(lines) + "\n").encode()
            split_hash = hashlib.sha256(split).hexdigest()
            for item, color in (("007", "yellow"), ("002", "red")):
                Image.new("RGB", (40, 32), color).save(test / f"{item}.jpg", format="PNG")
            (test / "004.jpg").write_bytes(b"bad")
            template = data / "sample_submission.csv"
            template.write_text("id,character\n007,placeholder\n004,invented\n002,placeholder\n")
            v2, v5 = root / "v2", root / "v5"
            v2.mkdir()
            v5.mkdir()
            (v2 / "classes.json").write_text(json.dumps(names))
            (v2 / "split.csv").write_bytes(split)
            (v5 / "settings.json").write_text(json.dumps({"classes": names, "split_sha256": split_hash}))
            old, new = root / "v3.csv", v5 / "submission_v5.csv"
            old.write_text("id,character\n007,abraham\n004,bart\n002,bart\n")
            new.write_text("id,character\n007,bart\n004,abraham\n002,bart\n")
            args = SimpleNamespace(data=data, v2_run=v2, v5_run=v5, v3_submission=old,
                                   v5_submission=new, output=root / "audit", workers=0)
            with (
                patch("audit_distribution.SPLIT_SHA256", split_hash),
                patch("v2.data_checks.CLASS_COUNT", 2),
                patch("v2.data_checks.COUNTS", {"train": 2, "val": 2}),
                patch("audit_distribution.read_test_images",
                      side_effect=lambda folder, template: read_test_images(folder, template, count=3)),
            ):
                audit(args)
            report = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(report["labels"]["allowed_characters"], names)
            self.assertFalse(report["labels"]["test_ground_truth_available"])
            self.assertEqual(report["test"]["decoded_images"], 2)
            self.assertEqual(report["test"]["non_mostly_black_edge_count"], 2)
            self.assertEqual(report["test_novelty"]["unseen_format_size"], 2)
            self.assertEqual(report["v3_v5_changes"]["changed"], 2)
            self.assertEqual(report["v3_v5_changes"]["test_images"], 3)
            self.assertTrue(report["test_novelty"]["top_rare_or_unseen_combinations"])
            with (args.output / "test_novelty.csv").open() as file:
                reader = csv.DictReader(file)
                self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
                table = {row["id"]: row for row in reader}
            self.assertEqual(set(table), {"007", "002"})
            self.assertEqual(table["007"]["v5_prediction"], "bart")
            self.assertEqual(float(table["007"]["fine_detail"]), 0.0)
            self.assertIn(table["007"]["fine_detail_bin"], ("low", "mid", "high"))
            with Image.open(args.output / "test_visual_outliers.png") as montage:
                montage.verify()
            self.assertIn("004.jpg", (args.output / "unreadable_images.csv").read_text())
            self.assertNotIn("placeholder", (args.output / "test_novelty.csv").read_text())


if __name__ == "__main__":
    unittest.main()
