import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from distribution_audit.compare import COMBO_FIELDS
from distribution_audit.novelty import MEASUREMENT_PREFIX, read_novelty
from predict_v7 import (LEDGER_COLUMNS, paired_prediction_arrays, trigger_ledger, verified_review,
                        write_candidate)
from v3.reports import read_reference
from v3.storage import file_sha256
from v7.impulse import MIN_TRIGGER_FRACTION, impulse_mask
from v7.review import audit_rows


def ledger_file(path, rows, header=LEDGER_COLUMNS):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


class V7CandidateTests(unittest.TestCase):
    def test_legacy_novelty_reader_keeps_numeric_values_and_bins_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_novelty.csv"
            header = (*MEASUREMENT_PREFIX, *COMBO_FIELDS)
            values = {
                "id": "001", "actual_format": "PNG", "width": "244", "height": "248",
                "extension_match": "False", "brightness": "0.42", "contrast": "0.22", "fine_detail": "0.031",
            }
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerow([values.get(key, "0.1") for key in MEASUREMENT_PREFIX]
                                + ["little_black", "colored", "high", "mid", "high"])
            result, = read_novelty(path)
            self.assertEqual(result["fine_detail"], "0.031")
            self.assertEqual(result["fine_detail_bin"], "high")
            self.assertEqual(result["contrast"], "0.22")
            self.assertEqual(audit_rows(path, ["001"], 1)["001"]["brightness_bin"], "high")
            path.write_text("id,fine_detail,fine_detail\n001,0.03,high\n")
            with self.assertRaisesRegex(ValueError, "Duplicate CSV columns with unknown layout"):
                list(read_novelty(path))

    def test_ledger_rejects_wrong_threshold_flags_dedup_and_reordered_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trigger.csv"
            ids = ["001", "002", "003"]
            rows = [("001", 0.003, True, "tail", "rare"), ("002", 0.002, False, "other", "known"),
                    ("003", 0.005, True, "tail", "rare")]
            ledger_file(path, rows)
            self.assertEqual(list(trigger_ledger(path, ids, 2)), ["001", "003"])
            with self.assertRaisesRegex(ValueError, "exactly 1"):
                trigger_ledger(path, ids, 1)
            ledger_file(path, rows[::-1])
            with self.assertRaisesRegex(ValueError, "template order"):
                trigger_ledger(path, ids, 2)
            ledger_file(path, [rows[0], rows[0], rows[2]])
            with self.assertRaisesRegex(ValueError, "template order"):
                trigger_ledger(path, ids, 2)
            ledger_file(path, [("001", 0.002, True, "tail", "rare"), *rows[1:]])
            with self.assertRaisesRegex(ValueError, "fixed threshold"):
                trigger_ledger(path, ids, 2)
            ledger_file(path, rows, ("id", "impulse_fraction", "filter_active", "visual_band", "visual_band"))
            with self.assertRaisesRegex(ValueError, "duplicated fields"):
                trigger_ledger(path, ids, 2)

    def test_requires_verified_review_and_completed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "review").mkdir()
            (run / "validation").mkdir()
            ledger = run / "review/test_impulse_signals.csv"
            ledger_file(ledger, [("001", 0.01, True, "tail", "rare")])
            info = {"v6_checkpoint_sha256": "abc", "test_image_signature": "sig"}
            review = {"provenance": info, "total_test": 1, "inspected": 1,
                      "groups": {"all:all": {"activated": 1}}, "trigger_fraction": MIN_TRIGGER_FRACTION}
            val = {"provenance": info, "decision": "supports_labeled_impulse_followup",
                   "trigger_fraction": MIN_TRIGGER_FRACTION, "method": "multiscale", "conditions": {
                       "textured_impulse": {"macro_gain": {"ci95_pp": [15.0, 21.67]}},
                       "clean": {"macro_gain": {"pp": 0}}, "texture_control": {"macro_gain": {"pp": 0}},
                   }}
            review_path = run / "review/review.json"
            validation_path = run / "validation/validation_report.json"
            review_path.write_text(json.dumps(review))
            validation_path.write_text(json.dumps(val))
            active, verified = verified_review(run, ["001"], info, 1)
            self.assertEqual(list(active), ["001"])
            self.assertFalse(verified["review_has_ledger_hash"])
            self.assertEqual(verified["trigger_ledger_sha256"], file_sha256(ledger))
            review_path.write_text(json.dumps({**review, "trigger_ledger_sha256": "changed"}))
            with self.assertRaisesRegex(ValueError, "ledger or trigger"):
                verified_review(run, ["001"], info, 1)
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "do not match"):
                verified_review(run, ["001"], {**info, "v6_checkpoint_sha256": "different"}, 1)
            validation_path.write_text(json.dumps({**val, "decision": "inconclusive"}))
            with self.assertRaisesRegex(ValueError, "does not support"):
                verified_review(run, ["001"], info, 1)

    def test_full_candidate_retains_inactive_ids_and_reports_archive_replay_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ids = [f"{10000 + i:06d}" for i in range(10791)][::-1]
            classes = [f"character_{i}" for i in range(50)]
            archived = {item: i % 2 for i, item in enumerate(ids)}
            active = {item: (0.02, "upper_tail", "rare") for i, item in enumerate(ids) if i % 6 == 0}
            active = dict(list(active.items())[:1728])
            first, second = list(active)[:2]
            raw = np.zeros((len(active), len(classes)), dtype=np.float32)
            filtered = np.zeros_like(raw)
            for i, item in enumerate(active):
                label = archived[item]
                raw[i, label] = 1
                filtered[i, label] = 1
            raw[0, 0], raw[0, 1] = 0.15, 0.85
            filtered[:2, 0], filtered[:2, 1] = 0.1, 0.9
            result = write_candidate(out, ids, classes, archived, active, raw, filtered, "active_raw_replay")
            self.assertEqual(result["images"], 10791)
            self.assertEqual(result["triggered"], 1728)
            self.assertEqual(result["copied_untriggered_from_v6"], 9063)
            self.assertEqual(result["changed_vs_archived_v6"], 2)
            self.assertEqual(result["raw_v6_archive_mismatch_ids"], [first])
            self.assertEqual(result["filter_changed_vs_raw_v6"], 1)
            final = read_reference(out / "submission_v7.csv", ids, classes)
            self.assertEqual(final[first], 1)
            self.assertEqual(final[second], 1)
            self.assertTrue(all(final[item] == archived[item] for item in ids if item not in active))
            self.assertEqual(result["confidence_and_margin"]["changed_vs_archive"]["images"], 2)

    def test_only_active_images_run_raw_and_filtered_multi_scale_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active_image = np.full((48, 48, 3), 110, dtype=np.uint8)
            active_image[2::6, 2::6] = (2, 250, 27)
            path = root / "001.jpg"
            image = Image.fromarray(active_image)
            image.save(path, format="PNG")
            fraction = float(impulse_mask(image).mean())
            args = SimpleNamespace(output=root, batch_size=16, workers=0)
            model = Mock()
            model.eval.return_value = model
            torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(
                DataLoader=lambda dataset, **kwargs: dataset,
            )))
            pixels = []

            def forward(mdl, dataset):
                item, = dataset
                pixels.append((item[1], np.asarray(item[0]).copy()))
                yield ["001"], np.array([[[0.9, 0.1]]], dtype=np.float32), None, 0.0

            with (
                patch.dict(sys.modules, {"torch": torch}),
                patch("predict_v7.Views", return_value=lambda value: value),
                patch("predict_v7.create_model", return_value=model),
                patch("predict_v7.load_checkpoint"),
                patch("predict_v7.infer_view_batches", side_effect=forward),
            ):
                active = {"001": (fraction, "tail", "rare")}
                test = [(path, "001"), (root / "intentionally_missing.jpg", "002")]
                raw, filtered = paired_prediction_arrays(
                    args, ["a", "b"], root / "v6.pt", "split", None, active, test, None,
                )
                np.testing.assert_array_equal(raw, filtered)
                paired_prediction_arrays(args, ["a", "b"], root / "v6.pt", "split", None,
                                         active, test, None)
            self.assertEqual([item for item, _ in pixels], ["001", "001"])
            np.testing.assert_array_equal(pixels[0][1], active_image)
            self.assertTrue((pixels[1][1] == 110).all())
