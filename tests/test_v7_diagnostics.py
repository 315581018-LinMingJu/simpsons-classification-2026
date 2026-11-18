import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from diagnose_v7 import paired_inference, review, validation_result
from v3.storage import file_sha256
from v7.impulse import CONDITIONS, add_impulses, clean_impulses, impulse_mask
from v7.review import cached_v6, review_groups


def speckled():
    image = np.full((96, 96, 3), 112, dtype=np.uint8)
    image[4:-4:6, 4:-4:6] = (2, 250, 27)
    return Image.fromarray(image)


class V7DiagnosisTests(unittest.TestCase):
    def test_isolated_noise_is_removed_without_smoothing_regular_texture(self):
        image = speckled()
        before = np.asarray(image).copy()
        result, fraction = clean_impulses(image)
        self.assertGreater(fraction, 0.01)
        self.assertTrue((np.asarray(result) == 112).all())
        np.testing.assert_array_equal(np.asarray(image), before)
        stripes = np.repeat(np.tile([35, 35, 35, 210, 210, 210], 16)[None, :, None],
                            96, axis=0)
        striped = Image.fromarray(np.repeat(stripes, 3, axis=2).astype(np.uint8))
        filtered, fraction = clean_impulses(striped)
        self.assertEqual(fraction, 0)
        self.assertIs(filtered, striped)
        self.assertFalse(impulse_mask(Image.new("RGB", (2, 2), "white")).any())

    def test_injection_repeats_without_changing_the_original(self):
        image = Image.new("RGB", (124, 96), "gray")
        a = add_impulses(image, 1, "example")
        b = add_impulses(image, 1, "example")
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotEqual(a.tobytes(), image.tobytes())
        self.assertEqual(a.size, image.size)
        self.assertTrue((np.asarray(image) == 128).all())

    def test_filter_reinference_preserves_baseline_batch_boundaries(self):
        samples = [(Path(item), item) for item in "abcdef"]
        args = SimpleNamespace(batch_size=2)
        phases = []

        def prepare(path, item):
            return speckled() if item in ("b", "f") else Image.new("RGB", (96, 96), "gray")

        def fake_infer(model, dataset, params):
            phases.append([item for _, item in dataset.samples])
            return {item: (1 if dataset.filtered else 0, 0.8, 0.4) for _, item in dataset.samples}

        with patch("diagnose_v7.infer", side_effect=fake_infer):
            raw, filtered, _ = paired_inference(None, samples, args, prepare, None)
        self.assertEqual(phases, [list("abcdef"), list("abef")])
        self.assertEqual(raw["a"], filtered["a"])
        self.assertEqual(filtered["b"][0], 1)
        self.assertEqual(filtered["f"][0], 1)

    def test_labeled_gate_needs_impulse_gain_and_both_guard_controls(self):
        labels = {"001": 0, "002": 1}
        good = {"001": (0, 0.8, 0.4), "002": (1, 0.8, 0.4)}
        bad = {"001": (1, 0.7, 0.5), "002": (0, 0.7, 0.5)}
        fractions = dict.fromkeys(labels, 0.01)
        cases = {condition: (good, good, fractions) for condition in CONDITIONS}
        self.assertEqual(validation_result(labels, cases, 2)["decision"], "no_evidence_for_filter")
        cases["textured_impulse"] = (bad, good, fractions)
        result = validation_result(labels, cases, 2)
        self.assertEqual(result["decision"], "supports_labeled_impulse_followup")
        self.assertEqual(result["conditions"]["textured_impulse"]["wins"], 2)
        cases["texture_control"] = (good, bad, fractions)
        self.assertEqual(validation_result(labels, cases, 2)["decision"], "guard_regression")

    def test_review_marks_automatically_selected_images_for_manual_confirmation(self):
        ids = {"001": 0.035, "002": 0.001, "6171": 0.006}
        features = {item: {"fine_detail": 0.05} for item in ids}
        groups = {name: (status, items) for name, status, items in review_groups(ids, features, 22)}
        self.assertEqual(groups["high_detail_low_impulse_proxy"], ("manual_confirmation_required", ["002"]))
        self.assertEqual(groups["visible_color_speckle"], ("visually_reviewed_examples", ["6171"]))

    def test_probability_cache_requires_correct_checkpoint_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "v6.csv"
            reference.write_text("id,character\n001,a\n002,b\n")
            path = root / "v6.npz"
            np.savez_compressed(path, ids=np.array(["001", "002"]),
                                probabilities=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32))
            settings = {"inference": {"classes": ["a", "b"], "method": "multiscale",
                                      "checkpoints": {"v6": "hash"}},
                        "test_image_signature": "sig", "reference_sha256": {"v6": file_sha256(reference)}}
            (root / "settings.json").write_text(json.dumps(settings))
            matrix, status = cached_v6(path, ["001", "002"], ["a", "b"], "hash", "sig", reference)
            self.assertEqual(status, "reused_verified_v6_probabilities")
            self.assertEqual(matrix.shape, (2, 2))
            self.assertEqual(cached_v6(path, ["002", "001"], ["a", "b"], "hash", "sig", reference),
                             (None, "incompatible_ids"))
            self.assertEqual(cached_v6(path, ["001", "002"], ["a", "b"], "other", "sig", reference),
                             (None, "incompatible_provenance"))

    def test_cpu_review_produces_unlabeled_real_image_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, ids = ["bart", "maggie"], ["001", "002"]
            files = [(root / f"{item}.jpg", item) for item in ids]
            speckled().save(files[0][0], format="PNG")
            Image.new("RGB", (64, 64), "green").save(files[1][0])
            (root / "summary.json").write_text(json.dumps({
                "provenance": {"split_sha256": "same"}, "labels": {"allowed_characters": names},
                "test": {"files": 2, "decoded_images": 2},
            }))
            (root / "test_novelty.csv").write_text(
                "id,fine_detail,visual_band,combo_support\n"
                "001,0.09,upper_tail,rare\n002,0.02,inside_reference,observed\n"
            )
            args = SimpleNamespace(audit_run=root, output=root / "v7", workers=0, seed=22)
            before = {path: path.read_bytes() for path, _ in files}
            review(args, names, files, ids, dict.fromkeys(ids, 1), {"split_sha256": "same"},
                   None, "unavailable")
            out = args.output / "review"
            report = json.loads((out / "review.json").read_text())
            self.assertEqual(report["groups"]["visual_band:upper_tail"]["activated"], 1)
            self.assertIsNone(report["groups"]["all:all"]["confidence"])
            self.assertEqual(report["v6_probability_cache"], "unavailable")
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            for name in ("test_categories.png", "filter_examples.png"):
                with Image.open(out / name) as image:
                    image.verify()
            with (out / "test_review.csv").open() as file:
                for item in csv.DictReader(file):
                    self.assertEqual(item["archived_v6_prediction"], "maggie")
                    self.assertEqual(item["v6_confidence"], "")
            self.assertFalse(list(args.output.rglob("*submission*.csv")))
