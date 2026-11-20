import importlib.util
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v2 import transforms_v2
from classification_v11 import arguments, sources
from v3.storage import file_sha256, json_bytes
from v8.data import Case
from v11.data import FOLDS, difference_hash, make_plan, near_hash, plan_counts, validate_plan
from v11.decision import PROFILES, accepted, choose_screen, metrics
from v11.retrieval import Feature, Index, Match, extract_batch, geometric_fit, reference_view


def strong(item, label):
    return Match(item, label, .85, 12, .85, .33, .012)


class V11Tests(unittest.TestCase):
    def test_hash_detects_flipped_panel(self):
        image = Image.new("RGB", (150, 164))
        draw = ImageDraw.Draw(image)
        draw.rectangle((26, 31, 114, 127), fill="yellow")
        draw.rectangle((38, 46, 72, 62), fill="blue")
        signature = difference_hash(image)
        flipped = difference_hash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        self.assertTrue(near_hash(signature, flipped[:1]))
        changed = np.bitwise_xor(signature[0], np.uint8(255))
        self.assertFalse(near_hash(signature[:1], changed[None]))

    def test_source_plan_filters_train_twins_and_keeps_fold_manifest(self):
        training = [Case(f"{label}/t{i}.jpg", label, "train") for label in range(2) for i in range(8)]
        validation = [Case(f"{label}/{fold}{i}.jpg", label, fold, f"{1-label}/t0.jpg")
                      for label in range(2) for fold in FOLDS for i in range(5)]
        rng = np.random.default_rng(31)
        signatures = [(str(i), rng.integers(0, 256, (2, 16), dtype=np.uint8))
                      for i in range(len(training + validation))]
        signatures[len(training)] = ("same_hash_variant", signatures[0][1].copy())
        with patch("v11.data.fingerprint_files", return_value=signatures):
            index, queries, exclusions = make_plan(training, validation, Path("data"), 2026, 1)
        self.assertTrue(plan_counts(index, queries, 2)[0])
        self.assertNotIn(validation[0].item, {row["item"] for row in queries})
        self.assertEqual(exclusions.get("near_hash_excluded", 0), 1)
        validate_plan(index, queries, training, validation)
        corrupted = [{**queries[0], "item": "test/6042.jpg"}, *queries[1:]]
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_plan(index, corrupted, training, validation)

    def test_geometry_finds_distributed_rotation_scale_inliers(self):
        rng = np.random.default_rng(4)
        left = rng.uniform(.10, .80, (15, 2))
        a = left[:, 0] + 1j * left[:, 1]
        b = .80 * np.exp(.23j) * a + .12 + .12j
        right = np.column_stack((b.real, b.imag))
        right[12:] = [[.95, .95], [0, .98], [.01, .01]]
        count, fraction, spread, error = geometric_fit(left, right)
        self.assertEqual(count, 12)
        self.assertAlmostEqual(fraction, 12 / 15)
        self.assertGreater(spread, .3)
        self.assertLess(error, 1e-7)

    def test_index_excludes_true_label_and_requires_ambiguity_abstention(self):
        cells = np.array([(x, y) for x in (.1, .35, .6, .85) for y in (.1, .4, .65)], dtype=np.float32)
        descriptors = np.eye(len(cells), dtype=np.float16)
        feature = Feature(np.array([1., 0.]), cells, descriptors)
        ref = [{"item": "a", "label": 0}, {"item": "b", "label": 1}]
        index = Index(ref, [feature, feature])
        self.assertIsNone(accepted(index.search([feature]), PROFILES["balanced"]))
        remaining = index.search([feature], absent_label=0)
        self.assertEqual(accepted(remaining, PROFILES["strict"]).item, "b")

    def evidence(self, false_accepts=0, improvements=16):
        positive = [{"item": f"source{i}", "label": i % 2, "role": "positive", "matches": [strong(f"source{i}", i % 2)],
                     "negative_matches": [strong("rival", 1 - i % 2)] if i < false_accepts else []}
                    for i in range(100)]
        val = [{"item": f"val{i}", "role": "validation", "label": i % 2,
                "baseline": 1 - i % 2 if i < improvements else i % 2,
                "matches": [strong(f"match{i}", i % 2)] if i < 24 else []} for i in range(200)]
        return positive + val

    def test_screen_locks_one_rule_and_confirm_enforces_negative_risk(self):
        report = choose_screen(self.evidence(), 1)
        self.assertEqual(report["locked_threshold"], "strict")
        confirmed = metrics(self.evidence(), report["locked_threshold"], 1, confirm=True)
        self.assertEqual(confirmed["decision"], "pass")
        self.assertEqual(confirmed["wins"], 16)
        self.assertEqual(confirmed["losses"], 0)
        self.assertGreater(confirmed["gain"]["ci95_pp"][0], 0)
        # 2/100 達點估計上限，但其 Wilson 上界依然太高。
        negative = metrics(self.evidence(false_accepts=2), "strict", 1, confirm=True)
        self.assertIn("false_acceptance_risk", negative["failures"])

    def test_no_gain_stops_before_any_confirm(self):
        report = choose_screen(self.evidence(improvements=0), 1)
        self.assertEqual(report["decision"], "stop_screen")
        self.assertIsNone(report["locked_threshold"])

    def test_exact_v8_archive_uses_digest_only_and_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(data=root / "data", v8_run=root / "v8")
            archive = args.v8_run / "candidates/convnext_tta_denoise"
            archive.mkdir(parents=True)
            checkpoint = args.v8_run / "models/convnext_all_layers/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            submission = archive / "submission_v8.csv"
            submission.write_bytes(b"archived-file")
            (archive / "settings.json").write_bytes(json_bytes({
                "checkpoint_sha256": file_sha256(checkpoint), "submission_sha256": file_sha256(submission),
                "architecture": "convnext_tiny", "method": "multiscale", "denoise": True, "images": 10791,
                "batch_size": 16,
            }))
            benchmark = args.v8_run / "benchmark"
            benchmark.mkdir()
            inputs = {"train_per_class": 0, "seed": 2026}
            report = {"inputs": inputs, "selected_image_signature": "images",
                      "manifest_sha256": {"train": "x", "validation": "y"}}
            (benchmark / "prepare.json").write_bytes(json_bytes(report))
            with (patch("classification_v11.benchmark", return_value=(report, [object()], [])),
                  patch.dict("classification_v11.COUNTS", {"train": 1})):
                self.assertEqual(sources(args).checkpoint, checkpoint)
                submission.write_bytes(b"different upload")
                with self.assertRaisesRegex(ValueError, "archive"):
                    sources(args)

    def test_output_cannot_point_to_old_runs(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["pilot", "--output", "runs/v8_full"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V11TorchTests(unittest.TestCase):
    def test_convnext_extracts_normalized_geometry_features_on_cpu(self):
        import torch

        from v3.inference import Views
        from v8.models import make_model

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(83)
            model = make_model("convnext_tiny", 50).eval()
            normalization = Views("multiscale", transforms_v2()[1]).normalize
            rng = np.random.default_rng(44)
            image = Image.fromarray(rng.integers(50, 245, (224, 224, 3), dtype=np.uint8))
            features, twin = extract_batch(model, [reference_view(image), (image, None)],
                                           normalization, device="cpu")
            self.assertEqual(features.vector.shape, (384,))
            self.assertEqual(features.descriptors.shape, (96, 192))
            self.assertAlmostEqual(float(np.linalg.norm(features.vector)), 1.0, places=6)
            matches = Index([{"item": "a", "label": 0}], [features]).search([twin])
            self.assertGreaterEqual(matches[0].inliers, 10)
        finally:
            torch.set_num_threads(threads)
