import hashlib
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v2 import transforms_v2
from classification_v8 import arguments
from v2.data_checks import Sample
from v3.storage import image_signature, json_bytes
from v8.data import (CONDITIONS, HARD_WEIGHTS, Case, TrainingImages, benchmark, prepare,
                     stress, train_cases)
from v8.experiment import (accumulation_size, comparison, labels_for, load_evaluation,
                           save_evaluation, train_epoch)
from v8.models import SOURCES, make_model, permit_weights, trainable_modules, verify_weight


class V8Tests(unittest.TestCase):
    def test_train_sampling_uses_the_original_train_partition(self):
        root = Path("data")
        rows = [Sample(root / name / f"{i}.jpg", label, part)
                for label, name in enumerate(("a", "b"))
                for i, part in enumerate(("train", "val", "train"))]
        cases = train_cases(rows, root, 1, 42)
        self.assertEqual(cases, train_cases(list(reversed(rows)), root, 1, 42))
        self.assertEqual([0, 1], [case.label for case in cases])
        self.assertTrue(all(case.fold == "train" and not case.item.endswith("1.jpg") for case in cases))

    def test_fixed_challenge_keeps_clean_and_pairs_impulse_with_context(self):
        image = Image.new("RGB", (224, 224))
        ImageDraw.Draw(image).rectangle((35, 30, 170, 203), fill=(230, 155, 70))
        sample = "bart/1.jpg"
        self.assertIs(stress(image, "clean", 42, sample), image)
        for condition in CONDITIONS[1:]:
            with self.subTest(condition=condition):
                result = stress(image, condition, 42, sample, image)
                same = stress(image, condition, 42, sample, image)
                self.assertEqual(result.size, (264, 264))
                self.assertEqual(result.tobytes(), same.tobytes())
        scene = np.array(stress(image, "context", 42, sample))
        impulse = np.array(stress(image, "impulse", 42, sample))
        changed = (scene != impulse).any(axis=2).mean()
        self.assertTrue(0.01 < changed < 0.25)
        self.assertAlmostEqual(sum(HARD_WEIGHTS.values()), 1)

    def test_positive_confirmation_requires_paired_hard_improvement(self):
        labels = np.repeat([0, 1], 40)
        reference = {key: labels.copy() for key in CONDITIONS}
        for key in HARD_WEIGHTS:
            reference[key][::5] = 1 - labels[::5]
        good = {key: labels.copy() for key in CONDITIONS}
        self.assertTrue(comparison(labels, good, reference, 42)["passes_confirmation_threshold"])
        self.assertFalse(comparison(labels, reference, reference, 42)["passes_confirmation_threshold"])
        good["clean"][0] = 1
        self.assertFalse(comparison(labels, good, reference, 42)["passes_confirmation_threshold"])

    def test_single_condition_regression_blocks_confirmation(self):
        labels = np.repeat([0, 1], 40)
        candidate = {key: labels.copy() for key in CONDITIONS}
        reference = {key: labels.copy() for key in CONDITIONS}
        for key in ("context", "impulse", "clipped"):
            reference[key][::3] = 1 - labels[::3]
        candidate["distractor"][::7] = 1 - labels[::7]
        result = comparison(labels, candidate, reference, 42)
        self.assertGreater(result["hard_effect"]["ci95_pp"][0], 0)
        self.assertFalse(result["passes_confirmation_threshold"])

    def test_provenance_permission_and_exact_bytes(self):
        with self.assertRaisesRegex(ValueError, "allow-imagenet"):
            permit_weights({"kind": "imagenet1k"}, False)
        permit_weights({"kind": "imagenet1k"}, True)
        permit_weights({"kind": "scratch"}, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"synthetic")
            source = replace(SOURCES["convnext_tiny"], sha256=hashlib.sha256(b"synthetic").hexdigest())
            self.assertEqual(verify_weight(path, source), path)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_weight(path, SOURCES["convnext_tiny"])

    def test_tail_accumulation_counts_actual_images(self):
        self.assertEqual(accumulation_size(129, 16, 4), 64)
        self.assertEqual(accumulation_size(17, 16, 4), 17)
        self.assertEqual(accumulation_size(1, 16, 4), 1)

    def test_outputs_keep_unique_ids_and_support_original_clean(self):
        cases = [Case("a/1.jpg", 0, "original"), Case("b/1.jpg", 1, "original")]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            settings = {"checkpoint_sha256": "v8", "conditions": ["clean"]}
            save_evaluation(directory, cases, ["a", "b"], {"clean": labels_for(cases)}, {}, settings)
            predictions, report = load_evaluation(directory, cases, settings, ["a", "b"])
            self.assertEqual(predictions["clean"].tolist(), [0, 1])
            self.assertEqual(report["per_class"]["b"]["clean"], 1)
            self.assertIsNone(report["hard_accuracy"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_evaluation(directory, cases, {**settings, "checkpoint_sha256": "other"}, ["a", "b"])
            (directory / "predictions.csv").write_text("forged", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_evaluation(directory, cases, settings, ["a", "b"])

    def test_prepare_creates_read_only_repeatable_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            args = SimpleNamespace(data=parent / "data", output=parent / "output", v1_run=parent / "v1",
                                   v2_run=parent / "v2", v5_run=parent / "v5", audit=parent / "summary.json",
                                   per_class=4, train_per_class=1, seed=123)
            root = args.data / "train/train/train"
            classes = ["a", "b"]
            rows, val = [], []
            for index, name in enumerate(classes):
                (root / name).mkdir(parents=True)
                for i in range(5):
                    path = root / name / f"{i}.jpg"
                    Image.new("RGB", (16, 16), (index * 130, i * 40, 60)).save(path)
                    rows.append(Sample(path, index, "val" if i else "train"))
                    if i:
                        val.append((path, path.relative_to(root).as_posix()))
            args.v5_run.mkdir()
            (args.v5_run / "settings.json").write_bytes(json_bytes({
                "classes": classes, "split_sha256": "original",
                "train_image_signature": image_signature(row.path for row in rows),
            }))
            for name in ("validation_subset.json", "validation_samples.csv"):
                (args.v5_run / name).touch()
            args.audit.write_bytes(json_bytes({
                "provenance": {"split_sha256": "original"},
                "labels": {"allowed_characters": classes, "test_ground_truth_available": False},
                "train": {"decoded_images": len(rows)},
                "test": {"decoded_images": 10791, "feature_medians": {
                    "brightness": 0.45, "contrast": 0.20, "fine_detail": 0.03,
                }},
            }))
            folds = ["calibration", "selection", "confirmation", "confirmation"] * 2
            distractors = {item: rows[5] if item.startswith("a/") else rows[0] for _, item in val}
            with (patch("v8.data.load_comparison_split", return_value=(classes, rows, "original")),
                  patch("v8.data.fresh_validation",
                        return_value=(val, [0] * 4 + [1] * 4, folds, distractors, {})),
                  patch("v8.data.head_status", return_value={"http": 200})):
                prepare(args)
                prepare(args)
                report, train, validation = benchmark(args)
            self.assertEqual(report["counts"], {"train": 2, "screen": 4, "confirm": 4})
            self.assertFalse({case.item for case in train} & {case.item for case in validation})
            rows[0].path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Images changed"):
                benchmark(args)

    def test_cli_rejects_old_outputs_and_ambiguous_resume(self):
        with self.assertRaises(SystemExit):
            arguments(["prepare", "--output", "runs/v6_composite"])
        with self.assertRaises(SystemExit):
            arguments(["train", "--tag", "trial", "--resume", "--init", "imagenet"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V8TorchSmokeTests(unittest.TestCase):
    def test_real_backbones_forward_and_partial_scope(self):
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for name in SOURCES:
                with self.subTest(architecture=name):
                    model = make_model(name, 50).eval()
                    with torch.inference_mode():
                        logits = model(torch.zeros(1, 3, 224, 224))
                    self.assertEqual(tuple(logits.shape), (1, 50))
                    self.assertTrue(logits.isfinite().all().item())
                    head, partial = trainable_modules(model, name, "partial")
                    self.assertIn(head, partial)
                    self.assertTrue(len(partial) > 1)
                    self.assertNotIn(model, partial)
                    del model
        finally:
            torch.set_num_threads(threads)

    def test_train_epoch_steps_for_final_partial_group(self):
        import torch

        torch.manual_seed(32)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        data = torch.utils.data.TensorDataset(torch.randn(5, 2), torch.tensor([0, 0, 1, 1, 1]))
        args = SimpleNamespace(batch_size=2, workers=0, accum=2, amp="off")
        with (patch.object(torch.Tensor, "cuda", lambda self, **_: self),
              patch.object(optimizer, "step", wraps=optimizer.step) as step):
            loss = train_epoch(model, [model], optimizer, scaler, data, args, 123)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(step.call_count, 2)

    def test_real_augmentations_output_224_tensor(self):
        from v6.composite import Targets

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = []
            for i, name in enumerate(("a", "b")):
                (root / name).mkdir()
                for j in range(8):
                    item = f"{name}/{j}.jpg"
                    image = Image.new("RGB", (224, 224))
                    ImageDraw.Draw(image).ellipse((15, 25, 195, 211), fill=(255, 128, i * 110))
                    image.save(root / item)
                    cases.append(Case(item, i, "train"))
            dataset = TrainingImages(root, cases, transforms_v2()[0], Targets(.45, .18, .03), "robust", 77)
            for i in range(len(dataset)):
                tensor, label = dataset[i]
                self.assertEqual(tuple(tensor.shape), (3, 224, 224))
                self.assertTrue(tensor.isfinite().all().item())
                self.assertEqual(label, cases[i].label)
