import importlib.util
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v14 import arguments, inspect_history, predict, read_history, test_comparison
from v3.inference import Prediction
from v3.reports import submission
from v3.storage import file_sha256, image_signature, json_bytes
from v6.composite import Targets
from v14.training import Budget, BudgetExhausted, choose_epoch, early_stop, learning_rate, mixture, relight, selection, train_one


class V14Tests(unittest.TestCase):
    def test_schedule_and_controlled_mixture(self):
        values = [learning_rate(i, 4) for i in range(1, 5)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(values[0], 3e-5)
        self.assertAlmostEqual(values[-1], 5e-6)
        self.assertEqual(mixture(1), (.15, .15))
        self.assertAlmostEqual(sum(mixture(4)), .45)

    def test_same_size_relight_preserves_interior_and_source(self):
        image = Image.new("RGB", (224, 224))
        ImageDraw.Draw(image).rectangle((32, 31, 190, 184), fill=(220, 140, 40))
        altered = relight(image, Targets(.46, .21, .031), np.random.default_rng(1))
        self.assertIsNotNone(altered)
        self.assertEqual(altered.size, image.size)
        np.testing.assert_array_equal(np.asarray(image)[31:185, 32:191], np.asarray(altered)[31:185, 32:191])
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertNotEqual(altered.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_selection_allows_small_loss_and_does_not_require_gain(self):
        ref = {"accuracy": {"clean": .98}, "hard_accuracy": .90}
        safety = selection({"accuracy": {"clean": .97}, "hard_accuracy": .895}, ref)
        self.assertTrue(safety["eligible"])
        self.assertLess(safety["hard_delta_pp"], 0)
        rows = [{"epoch": i, "loss": .2, "selection": safety} for i in (1, 2, 3)]
        self.assertEqual(choose_epoch(rows)["epoch"], 1)
        self.assertEqual(choose_epoch(rows, 2)["epoch"], 2)
        self.assertEqual(early_stop(rows), "early_stopping")
        rows[1]["selection"] = selection({"accuracy": {"clean": .8}, "hard_accuracy": .9}, ref)
        self.assertIsNone(choose_epoch(rows, 2))

    def test_budget_persists_and_stops_resume_after_cap(self):
        with tempfile.TemporaryDirectory() as root:
            current = [10.]
            path = Path(root) / "budget.json"
            budget = Budget(path, 60, lambda: current[0])
            current[0] += 40
            budget.check()
            current[0] += 300  # 停機不計入
            resumed = Budget(path, 60, lambda: current[0])
            self.assertEqual(resumed.elapsed, 40)
            current[0] += 20
            with self.assertRaises(BudgetExhausted):
                resumed.check()

    def test_class_changes_measure_both_directions_without_test_labels(self):
        labels = ["a", "b"]
        old, new = {"x": 0, "y": 0, "z": 1}, {"x": 1, "y": 0, "z": 0}
        rows, moves, changed = test_comparison(list(old), old, new, labels)
        self.assertEqual(changed, 2)
        self.assertEqual(rows[0], ("a", 2, 2, 1, 1, 0))
        self.assertEqual(moves, [("a", "b", 1), ("b", "a", 1)])

    def test_output_cannot_target_v8(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["prepare", "--output", "runs/v8_full"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "requires torch")
class V14TorchTests(unittest.TestCase):
    def test_cpu_mocked_prediction_publishes_verified_candidate_and_deltas(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v8 = root / "v8"
            archived = v8 / "candidates/convnext_tta_denoise/submission_v8.csv"
            archived.parent.mkdir(parents=True)
            archived.write_text("id,character\nx,a\ny,b\n", encoding="utf-8")
            template = root / "sample_submission.csv"
            template.write_text("id,character\nx,b\ny,a\n", encoding="utf-8")
            samples = [(root / f"{i}.jpg", i) for i in ("x", "y")]
            for path, _ in samples:
                Image.new("RGB", (24, 24)).save(path)
            folder = root / "a"
            folder.mkdir()
            source = root / "v8.pt"
            model = torch.nn.Linear(1, 2)
            identity = {"arm": "a"}
            torch.save({"identity": identity, "epoch": 1, "model": model.state_dict(),
                        "training": {"epoch": 1, "loss": .2, "selection": {"eligible": True, "score": .9}}},
                       folder / "epoch_01.pt")
            provenance = {"checkpoint_sha256": "v8-sha", "v8_submission_sha256": file_sha256(archived),
                          "test_image_signature": image_signature(p for p, _ in samples),
                          "template_sha256": file_sha256(template)}
            test = SimpleNamespace(samples=samples, archived={"x": 0, "y": 1}, provenance=provenance)
            args = SimpleNamespace(epoch=None, arm="a", v8_run=v8, data=root, predict_batch_size=None, workers=0)
            info = SimpleNamespace(checkpoint=source, identity=provenance, settings={"batch_size": 8},
                                   inputs={"seed": 2026, "classes": ["a", "b"]})
            with (patch("classification_v14.test_sources", return_value=test),
                  patch("classification_v14.init_cuda"), patch("classification_v14.make_model", return_value=model),
                  patch.object(torch.nn.Module, "cuda", lambda self: self),
                  patch("classification_v14.infer_loader", return_value=[
                      [("x", Prediction(1, .8, 0, 0)), ("y", Prediction(1, .9, 0, 0))]
                  ]),
                  patch("classification_v14.submission", side_effect=lambda *args: submission(*args, count=2)),
                  redirect_stdout(StringIO())):
                predict(args, info, folder, identity)
            result = folder / "candidate_epoch_01"
            self.assertEqual((result / "submission.csv").read_text(), "id,character\nx,b\ny,b\n")
            self.assertIn("a,b,1", (result / "transitions.csv").read_text())

    def test_optimizer_audit_detects_different_epoch_and_changed_weights(self):
        import torch

        best = {"epoch": 1, "best_epoch": 1, "history": [{"epoch": 1}], "model": {"a": torch.ones(1)},
                "config": {}, "architecture": "convnext_tiny", "classes": ["a"],
                "split_sha256": "split", "origin": {"kind": "scratch"}, "parent_sha256": None}
        same = {**best, "optimizer": {"state": {0: {"step": 1}}}}
        audit = inspect_history(best, same)
        self.assertTrue(audit["last_matches_best"])
        self.assertFalse(audit["best_has_optimizer"])
        self.assertFalse(inspect_history(best, {**same, "epoch": 2})["last_matches_best"])
        self.assertFalse(inspect_history(best, {**same, "model": {"a": torch.zeros(1)}})["last_matches_best"])

    def test_epoch_recovery_and_fingerprint(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"arm": "a"}
            for index in (1, 2):
                torch.save({"identity": identity, "epoch": index, "training": {"epoch": index}},
                           root / f"epoch_{index:02d}.pt")
            p1 = root / "epoch_01.pt"
            (root / "history.json").write_bytes(json_bytes([{"epoch": 1, "checkpoint_sha256": file_sha256(p1)}]))
            self.assertEqual(len(read_history(root, identity)), 2)
            torch.save({"identity": identity, "epoch": 1, "training": {"epoch": 1},
                        "model": {"x": torch.ones(1)}}, p1)
            with self.assertRaisesRegex(ValueError, "changed"):
                read_history(root, identity)

    def test_cpu_optimizer_tail_covers_all_images_and_counts_augmentations(self):
        import torch

        model = torch.nn.Linear(3, 2)
        initial = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters())
        args = SimpleNamespace(batch_size=8, workers=0)
        dataset = [(torch.rand(3), i % 2, "v8_recipe" if i % 2 else "relight") for i in range(77)]
        with (patch.object(torch.Tensor, "cuda", lambda self, **_: self),
              patch.object(torch, "autocast", lambda *args, **kwargs: nullcontext())):
            result = train_one(model, optimizer, dataset, args, 2026, SimpleNamespace(check=lambda: None))
        self.assertFalse(torch.equal(initial, model.weight))
        self.assertEqual(result["images"], 77)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(sum(result["augmentation_counts"].values()), 77)
