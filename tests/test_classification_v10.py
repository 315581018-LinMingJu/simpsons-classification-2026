import importlib.util
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v10 import arguments, load_adapted, run, verified_report
from v2.data_checks import read_json
from v3.storage import file_sha256, json_bytes, publish_csv
from v6.composite import Targets
from v8.data import CONDITIONS, FIELDS, HARD_WEIGHTS, Case, selected_signature
from v10.augment import (MIN_OFFSET_FRACTION, MIN_VISIBLE_EDGE, MIN_VISIBLE_FRACTION,
                         SIZE, TrainingImages, render_scene)
from v10.experiment import DENOISE, METHOD, comparison, fit, gate, subset

TARGETS = Targets(.45852, .21, .03137)


def framed_image():
    image = Image.new("RGB", (224, 224))
    ImageDraw.Draw(image).rectangle((34, 29, 188, 186), fill=(224, 170, 49))
    return image


class V10Tests(unittest.TestCase):
    def test_scene_is_reproducible_preserves_color_and_keeps_visible_cues_in_frame(self):
        image = framed_image()
        result = render_scene(image, TARGETS, 2027, "character/image.jpg")
        again = render_scene(image, TARGETS, 2027, "character/image.jpg")
        self.assertEqual(result.reason, "accepted")
        self.assertEqual(result.image.tobytes(), again.image.tobytes())
        self.assertGreaterEqual(result.visible_fraction, MIN_VISIBLE_FRACTION)
        self.assertGreaterEqual(result.offset_fraction, MIN_OFFSET_FRACTION)
        x1, y1, x2, y2 = result.cue_box
        self.assertGreater(min(x1, y1), 0)
        self.assertLess(max(x2, y2), SIZE)
        self.assertGreaterEqual(min(x2 - x1, y2 - y1), MIN_VISIBLE_EDGE)
        self.assertEqual(result.image.getpixel(((x1 + x2) // 2, (y1 + y2) // 2)), (224, 170, 49))

    def test_ambiguous_matte_and_tiny_character_fall_back(self):
        no_matte = Image.new("RGB", (224, 224), "yellow")
        self.assertEqual(render_scene(no_matte, TARGETS, 1, "a").reason, "uncertain_matte")
        image = Image.new("RGB", (224, 224))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 190, 190), fill=(28, 28, 28))
        draw.rectangle((110, 110, 115, 115), fill=(240, 210, 90))
        result = render_scene(image, TARGETS, 1, "b")
        self.assertEqual(result.reason, "insufficient_cues")
        self.assertIs(result.image, image)

    def test_training_fallback_keeps_original_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "a/image.jpg"
            file.parent.mkdir()
            Image.new("RGB", (224, 224), "yellow").save(file)
            cases = [Case("a/image.jpg", 1, "train")]
            images = TrainingImages(root, cases, [("fallback", 1)], lambda p: p, TARGETS, 1)
            with patch("v10.augment.SCENE_PROBABILITY", 1.0):
                self.assertEqual(images[0], ("fallback", 1))

    def test_training_subset_is_fixed_and_class_balanced(self):
        cases = [Case(f"{label}/{i}.jpg", label, "train") for label in range(3) for i in range(10)]
        selected = subset(cases, 4, 2027, "train")
        self.assertEqual(selected, subset(list(reversed(cases)), 4, 2027, "train"))
        self.assertEqual([sum(row.label == label for row in selected) for label in range(3)], [4] * 3)

    def test_paired_win_uses_hard_conditions_and_separate_confirm_gate(self):
        labels = np.repeat([0, 1], 20)
        ref, adapted = ({key: labels.copy() for key in CONDITIONS} for _ in range(2))
        for name in ("context", "clipped"):
            ref[name][::4] = 1 - labels[::4]
        result = comparison("confirm", labels, ref, adapted, 2026)
        self.assertEqual(result["decision"], "eligible_for_candidate")
        self.assertEqual(result["conditions"]["context"]["wins"], 10)
        self.assertEqual(result["conditions"]["context"]["losses"], 0)
        gains = result["condition_gain_pp"]
        borderline = {"pp": 2, "ci95_pp": [0, 4]}
        self.assertEqual(gate("screen", borderline, gains), [])
        self.assertEqual(gate("confirm", borderline, gains), ["hard_ci_includes_no_improvement"])
        bad = {**gains, "clean": -0.75, "context": -0.25, "negative": -4.5}
        self.assertEqual(gate("confirm", {"pp": 2, "ci95_pp": [1, 4]}, bad),
                         ["clean_regression", "context_regression", "other_condition_regression"])

    def test_report_guard_binds_screen_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder))
            info = SimpleNamespace(provenance={"protocol": "v10"})
            path = args.output / "confirm/comparison.json"
            path.parent.mkdir()
            report = {
                "provenance": info.provenance, "v10_checkpoint_sha256": "weights", "audit_sha256": "audit",
                "screen_sha256": "screen", "phase": "confirm", "method": METHOD, "denoise": DENOISE,
                "decision": "eligible_for_candidate", "hard_effect": {"pp": 2, "ci95_pp": [1, 4]},
                "condition_gain_pp": {key: 0 for key in CONDITIONS},
            }
            path.write_bytes(json_bytes(report))
            self.assertEqual(verified_report(args, "confirm", info, "weights", "audit", "screen"),
                             file_sha256(path))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                verified_report(args, "confirm", info, "different", "audit", "screen")
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                verified_report(args, "confirm", info, "weights", "audit", "different")
            path.write_bytes(json_bytes({**report, "hard_effect": {"pp": 2, "ci95_pp": [0, 4]}}))
            with self.assertRaises(SystemExit):
                verified_report(args, "confirm", info, "weights", "audit", "screen")

    def test_cli_rejects_output_in_existing_runs(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["train", "--output", "runs/v8_full"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V10TorchTests(unittest.TestCase):
    def test_cpu_audit_reads_exact_full_mode_checkpoint_without_test_images(self):
        import torch

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            args = SimpleNamespace(data=base / "data", output=base / "out", stage="audit",
                                   v8_run=base / "v8", distribution_audit=base / "distribution.json",
                                   allow_imagenet=False)
            root = args.data / "train/train/train"
            classes = ["a", "b"]
            train, validation = [], []
            for i, name in enumerate(classes):
                (root / name).mkdir(parents=True)
                for j, fold in enumerate(("train", "screen", "confirm")):
                    item = f"{name}/{j}.jpg"
                    framed_image().save(root / item)
                    other = "" if j == 0 else f"{classes[1 - i]}/0.jpg"
                    (train if j == 0 else validation).append(Case(item, i, fold, other))
            benchmark = args.v8_run / "benchmark"
            benchmark.mkdir(parents=True)
            manifests = {}
            for name, cases in (("train", train), ("validation", validation)):
                manifest = benchmark / f"{name}.csv"
                publish_csv(manifest, FIELDS, ((x.item, x.label, x.fold, x.distractor) for x in cases))
                manifests[name] = file_sha256(manifest)
            config = {"benchmark_sha256": manifests, "mode": "full", "augmentation": "robust",
                      "train_image_signature": "all-train", "targets": asdict(TARGETS)}
            inputs = {"classes": classes, "split_sha256": "split", "seed": 2026,
                      "hard_weights": HARD_WEIGHTS, "train_image_signature": "all-train"}
            (benchmark / "prepare.json").write_bytes(json_bytes({
                "inputs": inputs, "manifest_sha256": manifests,
                "selected_image_signature": selected_signature(root, train + validation),
            }))
            checkpoint = args.v8_run / "models/convnext_all_layers/best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"format": "simpsons-v8-v1", "architecture": "convnext_tiny", "classes": classes,
                        "split_sha256": "split", "origin": {"kind": "scratch"}, "config": config}, checkpoint)
            args.distribution_audit.write_bytes(json_bytes({
                "provenance": {"split_sha256": "split"},
                "labels": {"allowed_characters": classes, "test_ground_truth_available": False},
                "test": {"decoded_images": 10791, "feature_medians": asdict(TARGETS)},
            }))
            with redirect_stdout(StringIO()), patch("torch.cuda.is_available", side_effect=AssertionError("no CUDA")):
                run(args)
            report = read_json(args.output / "audit/summary.json")
            self.assertEqual(report["training_images"], 2)
            self.assertEqual(report["decision"], "ready")
            self.assertEqual(report["provenance"]["v8_checkpoint_sha256"], file_sha256(checkpoint))
            self.assertTrue((args.output / "audit/preview.png").is_file())

    def test_cpu_smoke_one_optimizer_step_preserves_frozen_stage_and_saves_lineage(self):
        import torch

        class ToyConvNeXt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4))
                self.classifier = torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.classifier(self.features(x.mean(dim=(2, 3))))

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            args = SimpleNamespace(data=base / "data", workers=0, batch_size=4)
            root = args.data / "train/train/train"
            cases = []
            for i, name in enumerate(("a", "b")):
                (root / name).mkdir(parents=True)
                for j in range(2):
                    item = f"{name}/{j}.jpg"
                    framed_image().save(root / item)
                    cases.append(Case(item, i, "train"))
            checkpoint = base / "v8.pt"
            ref = ToyConvNeXt()
            torch.save(ref.state_dict(), checkpoint)
            provenance = {"training_seed": 2027, "v8_checkpoint_sha256": file_sha256(checkpoint),
                          "v10_training_signature": selected_signature(root, cases)}
            info = SimpleNamespace(provenance=provenance, path=checkpoint, classes=["a", "b"],
                                   parent={"config": {"augmentation": "robust"}, "model": ref.state_dict()})
            output = base / "v10.pt"
            with (patch("v10.experiment.init_cuda"),
                  patch("v10.experiment.gpu_environment", return_value={"device": "cpu smoke"}),
                  patch("v10.experiment.make_model", side_effect=lambda *_: ToyConvNeXt()),
                  patch.object(torch.nn.Module, "cuda", lambda self: self),
                  patch.object(torch.Tensor, "cuda", lambda self, **_: self),
                  patch.object(torch.cuda, "is_bf16_supported", return_value=True),
                  patch.object(torch.cuda, "reset_peak_memory_stats"),
                  patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "max_memory_reserved", return_value=0),
                  patch.object(torch, "autocast", lambda *a, **k: nullcontext())):
                result = fit(args, info, cases, TARGETS, output, "audit")
            saved = load_adapted(output, info, "audit")
            self.assertEqual(result["images"], 4)
            self.assertTrue(np.isfinite(result["loss"]))
            self.assertEqual(saved["configuration"]["head_lr"], 3e-5)
            self.assertTrue(torch.equal(saved["model"]["features.0.weight"], ref.features[0].weight))
            self.assertFalse(torch.equal(saved["model"]["classifier.weight"], ref.classifier.weight))
            self.assertEqual(file_sha256(checkpoint), provenance["v8_checkpoint_sha256"])
            with self.assertRaisesRegex(ValueError, "audited"):
                load_adapted(output, info, "incompatible")
