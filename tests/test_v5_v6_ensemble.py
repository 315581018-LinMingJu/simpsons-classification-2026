import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ensemble.data import audit_test, fresh_validation, probability_cache
from ensemble.evaluation import CONDITIONS, evaluate, fit_temperature, fuse, metrics, rescale
from ensemble_v5_v6 import accepted_candidate, run_test_inference
from v2.data_checks import Sample
from v3.reports import submission
from v3.storage import file_sha256


def mock_model_results(challenge_advantage, confirmation_advantage=True):
    labels = np.tile([0, 1], 6)
    folds = ["calibration"] * 4 + ["selection"] * 4 + ["confirmation"] * 4
    v5, v6 = [np.empty((len(CONDITIONS), len(labels), 2)) for _ in range(2)]
    for i, condition in enumerate(CONDITIONS):
        hard = condition not in ("clean", "heldout_composite")
        v6_wrong = hard if challenge_advantage else condition == "heldout_composite"
        for j, label in enumerate(labels):
            v5[i, j, label], v5[i, j, 1 - label] = 0.9, 0.1
            correct = not v6_wrong or (j >= 8 and not confirmation_advantage)
            v6_label = label if correct else 1 - label
            v6[i, j, v6_label], v6[i, j, 1 - v6_label] = 0.65, 0.35
            if j >= 8 and not confirmation_advantage and hard:
                v5[i, j, label], v5[i, j, 1 - label] = 0.1, 0.9
    return v5, v6, labels, folds


class EnsembleTests(unittest.TestCase):
    def test_soft_fusion_uses_confidence_not_votes_and_temperature_can_help(self):
        left = np.array([[0.60, 0.40]])
        right = np.array([[0.05, 0.95]])
        result = fuse(left, right, 0.5, {"v5": 1, "v6": 1})
        self.assertEqual(result.argmax(1).tolist(), [1])
        np.testing.assert_allclose(result.sum(1), 1)
        labels = np.array([0, 0, 1, 1])
        row = np.array([[0.99, 0.01], [0.01, 0.99], [0.01, 0.99], [0.99, 0.01]])
        overconfident = np.repeat(row[None, ...], len(CONDITIONS), axis=0)
        t = fit_temperature(overconfident, labels)
        self.assertGreater(t, 1)
        self.assertLess(metrics(rescale(overconfident, t), labels)["balanced"]["nll"],
                        metrics(overconfident, labels)["balanced"]["nll"])

    def test_validation_needs_independent_gain_and_more_than_v6_composite(self):
        result = evaluate(*mock_model_results(True), seed=55)
        self.assertEqual(result["decision"], "supports_ensemble")
        self.assertTrue(result["confirmation"]["ensemble_gain_vs_v6"]["families"]["v5_challenge"]["pp"] > 0)
        selection = evaluate(*mock_model_results(False), seed=55)
        self.assertEqual(selection["decision"], "no_candidate_passed_selection")
        confirm = evaluate(*mock_model_results(True, False), seed=55)
        self.assertEqual(confirm["decision"], "not_confirmed")
        self.assertEqual(confirm["selected_v5_weight"], result["selected_v5_weight"])
        self.assertEqual(len(result["selection_weights"]), 3)

    def test_validation_folds_exclude_prior_samples_and_identical_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v5 = root / "v5"
            v5.mkdir()
            train_root = root / "data"
            names = ["homer", "marge"]
            rows = []
            for label, name in enumerate(names):
                folder = train_root / name
                folder.mkdir(parents=True)
                for i in range(9):
                    path = folder / f"{i}.png"
                    color = (20 * (i if i != 2 else 1), 100 * label, 200)
                    Image.new("RGB", (12, 13), color).save(path)
                    rows.append(Sample(path, label, "train" if i == 0 else "val"))
            (v5 / "settings.json").write_text('{"train_image_signature":"sig","validation_samples":2}')
            (v5 / "validation_subset.json").write_text(
                '{"image_signature":"sig","excluded_ids":["homer/3.png","marge/3.png"]}'
            )
            (v5 / "validation_samples.csv").write_text(
                "relative_path,label\nhomer/1.png,homer\nmarge/1.png,marge\n"
            )
            sample, labels, folds, distractors, audit = fresh_validation(rows, train_root, v5, names, 5, 55)
            self.assertEqual(audit["removed_earlier_v5_validation_and_exact_copies"], 4)
            self.assertEqual(audit["removed_exact_train_matches"], 2)
            self.assertTrue(all(int(path.stem) >= 4 for path, _ in sample))
            for label in (0, 1):
                self.assertEqual({fold for fold, y in zip(folds, labels, strict=True) if y == label},
                                 {"calibration", "selection", "confirmation"})
            self.assertTrue(all(distractors[item].label != label
                                for (_, item), label in zip(sample, labels, strict=True)))

    def test_cache_has_exact_id_order_raw_probabilities_and_reuses_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            a = np.array([[[0.7, 0.3], [0.1, 0.9]], [[0.9, 0.1], [0.3, 0.7]]], dtype=np.float32)
            get = Mock(return_value=[(["009", "001"], a, None, 0)])
            ids = ["009", "001"]
            names = ["a", "b"]
            actual = probability_cache(path, ids, names, get)
            np.testing.assert_array_equal(actual, a.mean(axis=0))
            np.testing.assert_array_equal(probability_cache(path, ids, names, get), actual)
            get.assert_called_once()
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                probability_cache(path, ids[::-1], names, get)

    def test_cpu_archive_audit_tracks_shift_groups_without_treating_predictions_as_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archived_file = root / "v5.csv"
            archived_file.write_text("id,character\n001,homer\n002,marge\n")
            other = root / "v6.csv"
            other.write_text("id,character\n001,marge\n002,marge\n")
            names = ["homer", "marge"]
            (root / "summary.json").write_text(json.dumps({
                "provenance": {"split_sha256": "same", "v5_submission_sha256": file_sha256(archived_file)},
                "labels": {"allowed_characters": names},
                "test": {"files": 2, "decoded_images": 2},
            }))
            (root / "test_novelty.csv").write_text(
                "id,visual_band,combo_support,black_edge,weak_color,brightness,contrast,fine_detail\n"
                "001,upper_tail,rare,little_black,colored,high,low,high\n"
                "002,inside_reference,observed,mostly_black,colored,low,low,low\n"
            )
            args = SimpleNamespace(audit_run=root, v5_submission=archived_file, v6_submission=other)
            audit_test(root, args, ["001", "002"], names,
                       {"v5": {"001": 0, "002": 1}, "v6": {"001": 1, "002": 1}}, "same")
            result = json.loads((root / "v5_v6_test_audit.json").read_text())
            self.assertEqual(result["changed"], 1)
            self.assertEqual(result["top_transitions"], [{"v5": "homer", "v6": "marge", "count": 1}])
            upper = next(row for row in result["visual_bands"] if row["group"] == "upper_tail")
            self.assertEqual(upper["rate_ratio"], 2)

    def test_submission_gate_refuses_rejected_results_and_incompatible_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "validation_report.json"
            path.write_text('{"decision":"not_confirmed"}')
            with self.assertRaisesRegex(SystemExit, "保留 V6"):
                accepted_candidate(root, {})
            (root / "validation").mkdir()
            settings = root / "validation/settings.json"
            settings.write_text('{"inference":{"gpu":"A"}}')
            result = evaluate(*mock_model_results(True), seed=55)
            result["validation_settings_sha256"] = file_sha256(settings)
            path.write_text(json.dumps(result))
            self.assertEqual(accepted_candidate(root, {"gpu": "A"})["decision"], "supports_ensemble")
            with self.assertRaisesRegex(ValueError, "does not match"):
                accepted_candidate(root, {"gpu": "B"})


    def test_test_stage_reuses_model_probabilities_and_emits_only_one_supported_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            test = output / "test"
            test.mkdir()
            names, ids = ["homer", "marge"], ["009", "002"]
            images = [(output / f"{item}.jpg", item) for item in ids]
            for path, _ in images:
                path.touch()
            old = output / "v5_original.csv"
            new = output / "v6_original.csv"
            old.write_text("id,character\n009,homer\n002,marge\n")
            new.write_text("id,character\n009,marge\n002,marge\n")
            archived = {"v5": {"009": 0, "002": 1}, "v6": {"009": 1, "002": 1}}
            args = SimpleNamespace(output=output, v5_submission=old, v6_submission=new)
            vectors = {
                "v5": np.array([[0.99, 0.01], [0.2, 0.8]], dtype=np.float32),
                "v6": np.array([[0.48, 0.52], [0.1, 0.9]], dtype=np.float32),
            }
            for name, probs in vectors.items():
                probability_cache(test / f"{name}.npz", ids, names,
                                  lambda: [(ids, probs[None, ...], None, 0)])
            supported = {"selected_v5_weight": 0.25, "applied_temperatures": {"v5": 1, "v6": 1}}
            with (
                patch("ensemble_v5_v6.get_model") as loader,
                patch("ensemble_v5_v6.submission",
                      side_effect=lambda path, items, scores, classes: submission(
                          path, items, scores, classes, count=2)),
            ):
                run_test_inference(args, names, None, {}, {}, images, ids, archived, None)
                self.assertFalse(list(output.glob("submission_*.csv")))
                (output / "validation_report.json").write_text(json.dumps(supported))
                run_test_inference(args, names, None, {}, {}, images, ids, archived, supported)
                loader.assert_not_called()
            candidate, = output.glob("submission_*.csv")
            self.assertEqual(candidate.read_text(), "id,character\n009,homer\n002,marge\n")
            self.assertEqual(new.read_text(), "id,character\n009,marge\n002,marge\n")
