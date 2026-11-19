import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from classification_v2 import transforms_v2
from classification_v9 import candidate_gate, sample_batches, write_report
from v3.inference import Views
from v3.storage import file_sha256
from v7.impulse import clean_impulses
from v8.data import CONDITIONS
from v9.reports import frame_fraction, test_diagnostics as grouped_diagnostics, validation_report
from v9.views import (Bank, Example, RECIPES, REFERENCE, TEN, Recipe, Scores, ViewImages,
                      aggregate, cache_bank, collate_views, describe_recipes, infer, requested)


def scores(labels):
    return Scores(labels, np.full(len(labels), 0.6), np.full(len(labels), 0.3), np.zeros(len(labels)))


class V9Tests(unittest.TestCase):
    def test_frozen_selection_uses_same_condition_images_and_has_guardrails(self):
        labels = np.repeat([0, 1], 20)
        reference = labels.copy()
        reference[::4] = 1 - labels[::4]
        no_gain = "full_raw"
        good = "full_denoise"
        result = {
            cond: {REFERENCE: scores(labels if cond == "clean" else reference), good: scores(labels),
                   no_gain: scores(labels if cond == "clean" else reference)}
            for cond in CONDITIONS
        }
        report = validation_report(result, labels, [REFERENCE, no_gain, good], 2026)
        self.assertEqual(report["selected"], good)
        self.assertEqual(report["methods"][good]["wins"]["context"], 10)
        self.assertEqual(validation_report(result, labels, [REFERENCE, good], 2026, confirm=True)["decision"],
                         "passes_confirmation")
        bad_clean = labels.copy()
        bad_clean[0] = 1
        result["clean"][good] = scores(bad_clean)
        self.assertEqual(validation_report(result, labels, [REFERENCE, good], 2026, confirm=True)["decision"],
                         "no_candidate_passed")

    def test_diagnostic_compares_archive_and_replay_separately(self):
        predictions = {REFERENCE: scores(np.array([0, 1])), "full_raw": scores(np.array([0, 0]))}
        features = {"a": {"visual_band": "upper_tail", "combo_support": "rare", "black_edge_fraction": "0.3"},
                    "b": {"visual_band": "inside_reference", "combo_support": "observed", "black_edge_fraction": "0.9"}}
        dims, rows, mismatch = grouped_diagnostics(["a", "b"], predictions, [0.01, 0], features, {"a": 1, "b": 0})
        self.assertEqual(mismatch.tolist(), [0, 1])
        self.assertEqual(dims[0]["impulse_trigger"], "on")
        overall = next(row for row in rows if row[:3] == ("full_raw", "all", "all"))
        self.assertEqual(overall[-2:], (0.5, 0.5))

    def test_archive_batch_sampling_preserves_last_partial_batch(self):
        samples = list(range(107))
        self.assertEqual(sample_batches(samples, 16, 1000, 1), samples)
        picked = sample_batches(samples, 16, 33, 1)
        self.assertEqual(picked, sample_batches(samples, 16, 33, 1))
        for i in range(0, len(picked), 16):
            self.assertEqual(picked[i] % 16, 0)
            self.assertEqual(picked[i:i + 16], samples[picked[i]:picked[i] + 16])
        self.assertAlmostEqual(frame_fraction(224, 224, 288), (224 / 288)**2)
        self.assertLess(frame_fraction(284, 260, 288), frame_fraction(224, 224, 288))

    def test_cache_order_and_provenance_are_checked(self):
        examples = [Example(Path("one"), "one"), Example(Path("two"), "two")]
        plan = [(False, "full")]
        probabilities = np.array([[[0.6, 0.4], [0.2, 0.8]]], dtype=np.float32)
        batches = [(["one", "two"], probabilities, np.array([[0, 1]]), np.zeros(2), 2, 0.0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            with self.assertRaisesRegex(ValueError, "order"):
                cache_bank(path, examples[::-1], plan, {"source": 1}, 2, lambda: iter(batches))
            cache_bank(path, examples, plan, {"source": 1}, 2, lambda: iter(batches))
            reused = cache_bank(path, examples, plan, {"source": 1}, 2, lambda: self.fail("cache was not reused"))
            self.assertEqual(reused.labels.tolist(), [[0, 1]])
            with self.assertRaisesRegex(ValueError, "metadata"):
                cache_bank(path, examples, plan, {"source": 2}, 2, lambda: self.fail("incompatible cache"))

    def test_full_frame_anchor_probability_weights(self):
        plan = [(True, "full"), (True, "crop1"), (True, "crop2")]
        probs = np.array([[[0.9, 0.1]], [[0.25, 0.75]], [[0.25, 0.75]]], np.float32)
        bank = Bank(probs, probs.argmax(axis=2), np.zeros(1), 3, 0., 0.)
        mix = aggregate(bank, plan, Recipe(True, ("full", "crop1", "crop2"), True))
        ordinary = aggregate(bank, plan, Recipe(True, ("full", "crop1", "crop2")))
        self.assertEqual(mix.labels.tolist(), [0])
        self.assertEqual(ordinary.labels.tolist(), [1])
        self.assertAlmostEqual(mix.margin.item(), 0.15)

    def test_candidate_gate_requires_confirmation_and_exact_test_replay(self):
        name = "full_denoise"
        recipe = describe_recipes([name])
        provenance = {"checkpoint_sha256": "verified"}
        freeze = {"selected": name, "recipe": recipe, "screen_sha256": "frozen"}
        info = SimpleNamespace(provenance=provenance)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            confirm = args.output / "confirm/summary.json"
            review = args.output / "diagnostic/summary.json"
            review.parent.mkdir()
            table = review.parent / "by_group.csv"
            table.write_text("output", encoding="utf-8")
            evidence = {"hard_effect": {"pp": 2.0, "ci95_pp": [0.5, 3]},
                        "gain_pp": {key: 0.0 for key in CONDITIONS}}
            write_report(confirm, {**provenance, "decision": "passes_confirmation", "selected": name,
                         "recipes": recipe, "frozen_screen_sha256": "frozen", "methods": {name: evidence}})
            diagnostic = {"provenance": provenance, "group_table_sha256": file_sha256(table),
                          "replayed_complete_archive_batches": True}
            write_report(review, {**diagnostic, "v8_replay_mismatches": 2})
            with self.assertRaisesRegex(ValueError, "exact"):
                candidate_gate(args, info, freeze)
            review.unlink()
            write_report(review, {**diagnostic, "v8_replay_mismatches": 0})
            self.assertEqual(candidate_gate(args, info, freeze)["confirm_sha256"], file_sha256(confirm))
            evidence["hard_effect"]["ci95_pp"][0] = 0
            confirm.unlink()
            write_report(confirm, {**provenance, "decision": "passes_confirmation", "selected": name,
                         "recipes": recipe, "frozen_screen_sha256": "frozen", "methods": {name: evidence}})
            with self.assertRaisesRegex(ValueError, "Confirm support"):
                candidate_gate(args, info, freeze)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V9TorchTests(unittest.TestCase):
    def test_ten_pil_views_exactly_match_v8_and_share_untriggered_forward(self):
        import torch

        names = [REFERENCE, "multiscale_raw", "full_raw", "full_denoise", "center256_denoise"]
        plan = requested(names)
        rng = np.random.default_rng(45)
        impulse = np.full((260, 284, 3), 90, dtype=np.uint8)
        impulse[5:-5:6, 5:-5:6, :] = 230
        textured = rng.integers(20, 240, (260, 284, 3), dtype=np.uint8)
        images = [Image.fromarray(impulse), Image.fromarray(textured), Image.new("RGB", (250, 280), (9, 31, 47))]
        evaluation = transforms_v2()[1]
        reference = Views("multiscale", evaluation)
        with tempfile.TemporaryDirectory() as directory:
            examples = []
            for i, image in enumerate(images):
                # 使用如同 Test 的 PNG 內容與 .jpg 檔名。
                path = Path(directory) / f"{i}.jpg"
                image.save(path, format="PNG")
                examples.append(Example(path, str(i)))
            dataset = ViewImages(examples, plan, "test", 2026)
            rows = list(dataset)
        self.assertTrue(rows[0][2])
        self.assertFalse(rows[-1][2])
        for row, image in zip(rows, images, strict=True):
            filtered, _ = clean_impulses(image)
            for tensor, view in zip(reference(filtered), TEN, strict=True):
                self.assertTrue(torch.equal(tensor, row[0][plan.index((True, view))]))
            self.assertTrue(torch.equal(evaluation(filtered), row[0][plan.index((True, "center256"))]))
        class ToyModel(torch.nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        model = ToyModel().eval()
        packed = collate_views(rows)
        (ids, probs, labels, fractions, forwards, _), = infer(model, [packed], plan, device="cpu")
        self.assertEqual(ids, ["0", "1", "2"])
        expected = np.stack([model(torch.stack(views)).softmax(1).numpy()
                             for views in zip(*(reference(clean_impulses(image)[0]) for image in images), strict=True)])
        predicted = aggregate(Bank(probs, labels, fractions, forwards, 0., 0.), plan, RECIPES[REFERENCE])
        np.testing.assert_array_equal(predicted.labels, expected.mean(axis=0).argmax(axis=1))
        np.testing.assert_allclose(predicted.confidence, expected.mean(axis=0).max(axis=1), atol=0, rtol=0)
        self.assertLess(forwards, len(plan) * len(images))
