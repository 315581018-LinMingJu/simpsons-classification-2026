import csv
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from audit_v8_domain import arguments, inference_cost, run
from v2.data_checks import read_json
from v3.storage import file_sha256, json_bytes, publish_csv
from v8.data import FIELDS, Case, selected_signature
from v8.experiment import save_evaluation
from v8_audit.artifacts import load_v9, load_v9_csv, load_validation_csv, probability_metrics
from v8_audit.measure import activity_proxies, flag, group_rows
from v9.views import Example, RECIPES, REFERENCE, cache_bank


class DomainAuditTests(unittest.TestCase):
    def test_entropy_margin_and_disagreement_are_derived_from_same_view_bank(self):
        probability = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        row, = probability_metrics(probability, np.array([[0], [1]]))
        self.assertAlmostEqual(row["entropy_nats"], np.log(2), places=6)
        self.assertEqual(row["confidence"], .5)
        self.assertEqual(row["margin"], 0)
        self.assertEqual(row["view_disagreement"], .5)

    def test_npz_checks_v8_identity_and_computes_pairwise_view_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.npz"
            examples = [Example(Path("unused"), "1")]
            plan = [(True, view) for view in RECIPES[REFERENCE].views]
            meta = {"checkpoint_sha256": "official-v8", "phase": "diagnostic"}
            probs = np.tile(np.asarray([[[.8, .2]]], dtype=np.float32), (len(plan), 1, 1))
            labels = np.zeros((len(plan), 1), dtype=int)
            cache_bank(path, examples, plan, meta, 2, lambda: [
                (["1"], probs, labels, np.zeros(1), 10, 0),
            ])
            info = SimpleNamespace(provenance={"checkpoint_sha256": "official-v8"}, inputs={"classes": ["a", "b"]})
            rows, result = load_v9(path, info, examples, "diagnostic")
            self.assertEqual(result["unanimous_rate"], 1)
            self.assertEqual(result["images"], 1)
            self.assertAlmostEqual(rows["1"]["margin"], .6, places=6)
            info.provenance["checkpoint_sha256"] = "older"
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_v9(path, info, examples, "diagnostic")

    def test_csv_fallback_never_invents_entropy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            group = folder / "by_group.csv"
            publish_csv(group, ("method", "dimension", "group", "images", "mean_confidence",
                                "mean_top2_margin", "mean_view_disagreement"),
                        [(REFERENCE, "all", "all", 1, .8, .6, .1)])
            publish_csv(folder / "per_image.csv", ("id", "method", "archived_v8", "prediction", "confidence",
                                                  "top2_margin", "view_disagreement"),
                        [("1", REFERENCE, "a", "a", .8, .6, .1)])
            summary = {"provenance": {"checkpoint_sha256": "v8"}, "samples": 1,
                       "group_table_sha256": file_sha256(group), "wall_seconds": 1, "image_forwards": 10,
                       "v8_replay_mismatches": 0}
            (folder / "summary.json").write_bytes(json_bytes(summary))
            info = SimpleNamespace(provenance=summary["provenance"], inputs={"classes": ["a", "b"]},
                                   archived={"1": 0})
            rows, result = load_v9_csv(folder, info)
            self.assertNotIn("entropy_nats", rows["1"])
            self.assertEqual(result["images"], 1)
            self.assertNotIn("pairwise_agreement", result)
            group.write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_v9_csv(folder, info)

    def test_risk_counts_and_error_rates_respect_missing_values_and_ground_truth(self):
        rows, labels = {}, {}
        for i in range(64):
            item = str(i)
            rows[item] = {"prediction": int(i < 25)}
            if i != 63:
                rows[item]["brightness"] = .8 if i < 32 else .2
            labels[item] = 0
        labeled = next(row for row in group_rows(rows, labels) if row["group"] == "bright")
        unlabeled = next(row for row in group_rows(rows) if row["group"] == "bright")
        self.assertEqual((labeled["eligible"], labeled["flagged"], labeled["errors"]), (63, 32, 25))
        self.assertTrue(labeled["exploratory_signal"])
        self.assertNotIn("error_rate", unlabeled)
        self.assertIsNone(flag({"brightness": .8}, "bright_textured_little_black"))

    def test_proxy_can_find_offcenter_activity_and_preserves_missingness(self):
        image = Image.new("RGB", (280, 250))
        self.assertIsNone(activity_proxies(image)["activity_offcenter"])
        ImageDraw.Draw(image).rectangle((7, 7, 42, 49), fill="white")
        result = activity_proxies(image)
        self.assertGreater(result["activity_offcenter"], .3)
        self.assertGreater(result["center_crop_activity_loss"], .6)
        self.assertLess(result["activity_compactness"], .25)

    def test_cost_from_saved_timing_or_explicit_scenarios(self):
        actual = inference_cost(10_791, {}, {"clean": {"wall_seconds": 50, "image_forwards": 4_000, "source": "archive"}})
        self.assertEqual(actual["image_forwards"], 107_910)
        self.assertAlmostEqual(actual["projected_minutes"], 107_910 * 50 / 4_000 / 60)
        scenario = inference_cost(10_791, {}, {})
        self.assertIn("minutes_at_views_per_second_excluding_io", scenario)
        self.assertNotIn("projected_minutes", scenario)

    def test_read_only_pipeline_with_missing_test_probabilities_and_v13_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            names = ["a", "b"]
            data, out, v8 = cwd / "data", cwd / "out", cwd / "v8_full"
            out.mkdir()
            root = data / "train/train/train"
            training, val = [], []
            for label, character in enumerate(names):
                (root / character).mkdir(parents=True)
                for part in ("train", "screen"):
                    case = Case(f"{character}/{part}.jpg", label, part,
                                f"{names[1-label]}/train.jpg" if part == "screen" else "")
                    Image.new("RGB", (64, 64), ("white" if label == 0 else "black")).save(root / case.item)
                    (training if part == "train" else val).append(case)
            (v8 / "benchmark").mkdir(parents=True)
            publish_csv(v8 / "benchmark/train.csv", FIELDS,
                        [(case.item, case.label, case.fold, case.distractor) for case in training])
            (v8 / "benchmark/prepare.json").write_bytes(json_bytes({"selected_image_signature": "image_manifest"}))
            samples = []
            for i in range(3):
                path = data / f"test{i}.jpg"
                Image.new("RGB", (82, 65), (160 + i, 120, 110)).save(path)
                samples.append((path, str(i)))
            inputs = {"classes": names, "seed": 2026, "targets": {"brightness": .46, "contrast": .2, "fine_detail": .03}}
            provenance = {"checkpoint_sha256": "v8ckpt", "benchmark_sha256": "benchmark",
                          "validation_manifest_sha256": "val_manifest"}
            info = SimpleNamespace(inputs=inputs, provenance=provenance, validation=val, samples=samples,
                                   archived={"0": 0, "1": 0, "2": 1})
            args = SimpleNamespace(data=data, output=out, v8_run=v8, v9_run=cwd / "v9", v11_audit=cwd / "no.json",
                                   v10_run=cwd / "v10", v12_run=cwd / "v12", v13_run=cwd / "v13",
                                   audit_run=cwd / "absent-audit", test_sample=3, train_sample=2, workers=1)
            scores = args.v13_run / "scores/screen/reference/v8/clean"
            settings = {"fold": "screen", "method": "multiscale", "denoise": True, "precision": "fp32",
                        "checkpoint_sha256": "v8ckpt", "conditions": ["clean"], "context": {"identity": provenance},
                        "image_signature": selected_signature(root, val)}
            save_evaluation(scores, val, names, {"clean": np.array([0, 0])}, {"clean": {"seconds": 1}}, settings)
            with patch("audit_v8_domain.sources", return_value=info), redirect_stdout(StringIO()):
                run(args)
            summary = read_json(out / "summary.json")
            self.assertEqual(summary["validation_errors"]["clean"], {"images": 2, "errors": 1})
            self.assertEqual(summary["test_score_source"]["matched_archive_rows"], 0)
            self.assertEqual(summary["visual_source"]["mode"], "uniform_test_sample")
            self.assertEqual(summary["measurement_populations"]["test_visual"], 3)
            self.assertEqual(summary["test_most_predicted"][0], {"character": "a", "count": 2})
            with (out / "review.csv").open(encoding="utf-8") as file:
                test_rows = [row for row in csv.DictReader(file) if row["cohort_or_condition"] == "test"]
            self.assertTrue(test_rows)
            self.assertTrue(all(not row["validation_label"] for row in test_rows))
            with self.assertRaisesRegex(ValueError, "fold or images"):
                load_validation_csv(scores, info, list(reversed(val))[1:], "clean", root)

    def test_guard_rejects_writing_to_v8(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["--output", "runs/v8_full"])
