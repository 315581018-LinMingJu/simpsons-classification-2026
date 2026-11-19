import hashlib
import importlib.util
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from classification_v12 import arguments, expected_decision, read_base, run, shifted_cohort, train_phase
from v3.storage import file_sha256, json_bytes
from v8.data import Case, selected_signature
from v12.model import SOURCE, gpu_probe, make_base, permitted, weights_path
from v12.model import configure as configure_v12
from v12.protocol import ALL_CONDITIONS, SHIFT, comparison


def scores():
    labels = np.repeat([0, 1], 30)
    cohorts = {name: [Case(str(i), int(label), "screen") for i, label in enumerate(labels)]
               for name in ALL_CONDITIONS}
    baseline = {name: labels.copy() for name in ALL_CONDITIONS}
    current = {name: labels.copy() for name in ALL_CONDITIONS}
    return labels, cohorts, baseline, current


class V12Tests(unittest.TestCase):
    def test_hard_only_gain_stops_when_heldout_does_not_improve(self):
        labels, cohorts, before, after = scores()
        before["context"][::4] = 1 - labels[::4]
        result = comparison("screen", before, after, cohorts, 2026)
        self.assertGreater(result["hard_effect"]["pp"], 2)
        self.assertEqual(result["decision"], "stop_screen")
        self.assertIn("heldout_gain_below_minimum", result["reasons"])

    def test_paired_confirm_requires_both_positive_lower_bounds_and_clean_guard(self):
        labels, cohorts, before, after = scores()
        before["context"][::4] = 1 - labels[::4]
        before[SHIFT][::4] = 1 - labels[::4]
        result = comparison("confirm", before, after, cohorts, 2026)
        self.assertEqual(result["decision"], "eligible_for_candidate")
        self.assertGreater(result["heldout_effect"]["ci95_pp"][0], 0)
        after["clean"][0] = 1
        self.assertIn("clean_regression", comparison("confirm", before, after, cohorts, 2026)["reasons"])

    def test_weak_head_runs_no_tune(self):
        args = SimpleNamespace(stage="train", allow_imagenet=True)
        checkpoint = Path("head.pt")
        head = {"phase": "head", "decision": "stop_head", "reasons": ["weak_head_shift"]}
        with (patch("classification_v12.sources"), patch("classification_v12.prepared"),
              patch("classification_v12.reference_checkpoint"), patch("classification_v12.stage_settings"),
              patch("classification_v12.gpu_probe", return_value={"status": "ready"}),
              patch("classification_v12.train_phase", return_value=(checkpoint, {})) as train_mock,
              patch("classification_v12.read_base"), patch("classification_v12.evaluate", return_value=head),
              patch("classification_v12.expected_decision", side_effect=AssertionError("unexpected tune")),
              redirect_stdout(StringIO())):
            run(args)
        train_mock.assert_called_once()

    def test_v11_cohort_is_optional_and_verifies_original_records(self):
        validation = [Case(f"{fold}/{label}", label, fold, f"train/{1-label}")
                      for fold in ("screen", "confirm") for label in (0, 1)]
        identity = {"checkpoint_sha256": "v8", "benchmark_sha256": "bench",
                    "v8_image_signature": "images", "validation_seed": 2026}
        info = SimpleNamespace(validation=validation, identity=identity, inputs={"classes": ["a", "b"]})
        self.assertIs(shifted_cohort(None, info), validation)
        queries = [{"item": c.item, "role": "validation", "fold": c.fold, "label": c.label,
                    "other": c.distractor} for c in validation]
        with tempfile.TemporaryDirectory() as folder:
            audit = Path(folder) / "audit.json"
            audit.write_bytes(json_bytes({"decision": "ready", "identity": identity, "queries": queries}))
            self.assertEqual(shifted_cohort(audit, info), validation)
            queries[0]["fold"] = "confirm"
            audit.write_bytes(json_bytes({"decision": "ready", "identity": identity, "queries": queries}))
            with self.assertRaisesRegex(ValueError, "manifest"):
                shifted_cohort(audit, info)

    def test_confirm_gate_binds_checkpoint_and_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            path = out / "tune.pt"
            path.write_bytes(b"contents")
            args = SimpleNamespace(output=out)
            info = SimpleNamespace(identity={"v8": "ref"})
            screen = out / "screen.json"
            screen.write_bytes(json_bytes({
                "identity": info.identity, "prepare_sha256": "prepare", "v12_checkpoint_sha256": file_sha256(path),
                "phase": "screen", "method": "multiscale", "denoise": True, "decision": "advance_to_confirm",
            }))
            self.assertEqual(expected_decision(args, "screen", info, "prepare", path, "advance_to_confirm"),
                             file_sha256(screen))
            with self.assertRaisesRegex(ValueError, "provenance"):
                expected_decision(args, "screen", info, "old", path, "advance_to_confirm")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                expected_decision(args, "screen", info, "prepare", path, "advance_to_confirm")

    def test_permission_and_output_guard(self):
        with self.assertRaises(SystemExit):
            permitted(False)
        permitted(True)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["train", "--output", "runs/v8_full"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V12TorchTests(unittest.TestCase):
    def test_head_stage_writes_fixed_checkpoint_and_reuses_completed_epoch(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(3, 4))
                self.classifier = torch.nn.Linear(4, 2)

            def forward(self, image):
                return self.classifier(self.features(image.mean(dim=(2, 3))))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = [Case(f"{class_id}/a.jpg", class_id, "train") for class_id in (0, 1)]
            for case in cases:
                path = root / case.item
                path.parent.mkdir()
                Image.new("RGB", (224, 224), (110, 100, case.label * 180)).save(path)
            reference = root / "v8.pt"
            reference.write_bytes(b"unchanged")
            args = SimpleNamespace(output=root, batch_size=4, workers=0)
            info = SimpleNamespace(root=root, training=cases, targets=None, checkpoint=reference,
                                   inputs={"classes": ["a", "b"]},
                                   identity={"checkpoint_sha256": file_sha256(reference),
                                             "v12_train_signature": selected_signature(root, cases)})
            settings = {"phase": "head", "seed": 2027, "amp": "bf16", "accum": 16, "augmentation": "basic"}
            with (patch("classification_v12.init_cuda"), patch("classification_v12.weights_path"),
                  patch("classification_v12.make_base", side_effect=lambda *_: Toy()) as make,
                  patch.object(torch.nn.Module, "cuda", lambda self: self),
                  patch.object(torch.Tensor, "cuda", lambda self, **_: self),
                  patch.object(torch.cuda, "reset_peak_memory_stats"),
                  patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "max_memory_reserved", return_value=0),
                  patch.object(torch, "autocast", lambda *a, **k: nullcontext()),
                  redirect_stdout(StringIO())):
                args.weights_file = None
                path, stats = train_phase(args, info, settings)
                reused, again = train_phase(args, info, settings)
            self.assertEqual(path, reused)
            self.assertEqual(stats, again)
            self.assertEqual(stats["images"], 2)
            self.assertTrue(np.isfinite(stats["loss"]))
            make.assert_called_once()
            self.assertEqual(read_base(path, settings)["training"], stats)
            self.assertEqual(reference.read_bytes(), b"unchanged")

    def test_official_download_has_atomic_verified_cache(self):
        import torch

        payload = b"synthetic weights"
        source = replace(SOURCE, sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (patch("v12.model.SOURCE", source),
                  patch.object(torch.hub, "download_url_to_file",
                               side_effect=lambda url, dst, **_: Path(dst).write_bytes(payload)) as download):
                result = weights_path(root)
                self.assertEqual(result.read_bytes(), payload)
                self.assertEqual(weights_path(root), result)
                download.assert_called_once()
                self.assertEqual(list(root.iterdir()), [result])
                result.write_bytes(b"bad bytes")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    weights_path(root)

    def test_cpu_probe_reports_unavailable_cuda(self):
        env = {"cuda_available": False, "devices_visible": 0, "convnext_base": True}
        with patch("v12.model.environment", return_value=env):
            self.assertEqual(gpu_probe()["status"], "requires_one_bf16_cuda_gpu_and_torchvision_convnext")

    def test_base_real_forward_and_two_stage_freezing(self):
        import torch
        from torchvision.models import ConvNeXt_Base_Weights

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            self.assertEqual(SOURCE.url, ConvNeXt_Base_Weights.IMAGENET1K_V1.url)
            model = make_base(50).eval()
            with torch.inference_mode():
                logits = model(torch.zeros((1, 3, 224, 224)))
            self.assertEqual(tuple(logits.shape), (1, 50))
            self.assertTrue(logits.isfinite().all().item())
            _, warmup = configure_v12(model, "head")
            self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
            self.assertEqual(warmup.param_groups[0]["lr"], 1e-3)
            _, optimizer = configure_v12(model, "tune")
            self.assertFalse(any(p.requires_grad for p in model.features[:4].parameters()))
            self.assertTrue(all(p.requires_grad for p in model.features[4:].parameters()))
            self.assertEqual([g["lr"] for g in optimizer.param_groups], [2e-4, 2e-5])
        finally:
            torch.set_num_threads(threads)
