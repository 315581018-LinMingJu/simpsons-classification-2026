import importlib.util
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from classification_v13 import arguments, short_report, train_pair, verify_report
from v3.storage import file_sha256, json_bytes
from v8.data import Case
from v12.protocol import ALL_CONDITIONS, SHIFT
from v13.protocol import paired_report, screen_decision
from v13.training import (ARMS, BUDGET, EFFECTIVE_BATCH, BudgetExceeded, checkpoint_stats,
                          configure_last_stage, counted_epoch, manifests)


class V13Tests(unittest.TestCase):
    def test_manifests_are_fixed_balanced_disjoint_and_have_exact_budget(self):
        labels = [f"class{i}" for i in range(50)]
        training = [Case(f"{label}/{i}.jpg", index, "train")
                    for index, label in enumerate(labels) for i in range(260)]
        plan = manifests(training, labels, 2026)
        self.assertEqual(plan, manifests(list(reversed(training)), labels, 2026))
        self.assertEqual({name: len(cases) for name, cases in plan.items()},
                         {name: updates * EFFECTIVE_BATCH for name, (updates, _) in BUDGET.items()})
        self.assertEqual(np.bincount([c.label for c in plan["pilot"]]).tolist(), [128] * 50)
        self.assertEqual(len({c.label for c in plan["smoke"]}), 50)
        self.assertFalse({c.item for c in plan["pilot"]} & {c.item for c in plan["extend"]})

    def results(self, base_size=80, tiny_size=75, v8_size=90):
        labels = np.repeat([0, 1], 50)
        cohorts = {key: [Case(str(i), int(label), "screen") for i, label in enumerate(labels)]
                   for key in ALL_CONDITIONS}

        def scores(size):
            values = labels.copy()
            values[:100 - size] = 1 - values[:100 - size]
            return {key: values.copy() for key in ALL_CONDITIONS}

        base, tiny, v8 = map(scores, (base_size, tiny_size, v8_size))
        return (
            paired_report("screen", v8, tiny, cohorts, 1, "v8", "tiny"),
            paired_report("screen", v8, base, cohorts, 1, "v8", "base"),
            paired_report("screen", tiny, base, cohorts, 1, "tiny", "base"),
        )

    def test_matched_budgets_allow_one_extension_only_when_promising(self):
        checks = self.results()
        self.assertEqual(screen_decision("pilot", *checks)["decision"], "advance_to_extend")
        self.assertEqual(screen_decision("extend", *checks)["decision"], "stop")
        weak = self.results(base_size=65, tiny_size=75)
        self.assertEqual(screen_decision("pilot", *weak)["decision"], "stop")

    def test_base_needs_advantage_over_deployment_and_equal_budget_tiny(self):
        checks = self.results(base_size=100, tiny_size=90, v8_size=90)
        self.assertEqual(screen_decision("pilot", *checks)["candidate"], "base")
        tied = self.results(base_size=100, tiny_size=100, v8_size=90)
        self.assertEqual(screen_decision("pilot", *tied)["candidate"], "tiny")
        isolated = self.results(base_size=95, tiny_size=94, v8_size=95)
        self.assertIsNone(screen_decision("pilot", *isolated)["candidate"])
        summary = short_report({"comparisons": {"base_vs_tiny": checks[2]}})
        self.assertEqual(summary["comparisons"]["base_vs_tiny"]["heldout_accuracy_pct"]["base"], 100)

    def test_time_cap_stops_pair_without_advancing_with_one_arm(self):
        with patch("classification_v13.train_arm", side_effect=[{"optimizer_steps": 100}, BudgetExceeded(30, 721)]):
            result = train_pair(None, None, None, "pilot")
        self.assertEqual(result["decision"], "stop_budget")
        self.assertEqual(result["failed_arm"], "base")
        self.assertEqual(result["optimizer_steps_before_stop"], 30)
        self.assertEqual(list(result["training"]), ["tiny"])

    def test_cached_confirm_binds_checkpoint_and_screen_report(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            screen = out / "pilot.json"
            screen.write_bytes(b"locked")
            checkpoint = out / "checkpoints/pilot_base.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            context = {"code": "pinned"}
            result = {"context": context, "phase": "confirm", "checkpoint_phase": "pilot",
                      "screen_sha256": file_sha256(screen), "confirmed_checkpoint_sha256": {
                          "base": file_sha256(checkpoint),
                      }}
            (out / "confirm.json").write_bytes(json_bytes(result))
            args = SimpleNamespace(output=out)
            self.assertEqual(verify_report(args, context, "confirm"), result)
            screen.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Screen lock"):
                verify_report(args, context, "confirm")

    def test_output_rejects_v12_directory(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["pilot", "--output", "runs/v12"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class V13TorchTests(unittest.TestCase):
    def test_real_adamw_updates_are_counted_last_stage_and_head_change(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
                self.classifier = torch.nn.Sequential(torch.nn.Linear(4, 2))

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

        torch.manual_seed(42)
        model = Toy()
        modules, optimizer = configure_last_stage(model)
        data = torch.utils.data.TensorDataset(torch.randn(128, 4), torch.arange(128) % 2)
        args = SimpleNamespace(batch_size=8, accum=8, amp="bf16", workers=0)
        original = [param.detach().clone() for param in model.parameters()]
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        updates = []

        def observe():
            updates.append([bool(torch.isfinite(layer.weight.grad).all()) for layer in modules
                            for layer in (layer[-1] if isinstance(layer, torch.nn.Sequential) else layer,)])

        with (patch.object(torch.Tensor, "cuda", lambda self, **_: self),
              patch.object(torch, "autocast", lambda *a, **k: nullcontext())):
            loss, steps = counted_epoch(model, modules, optimizer, scaler, data, args, 123, 2, 120, observe,
                                         clock=lambda: 0)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(steps, 2)
        self.assertEqual(updates, [[True, True], [True, True]])
        self.assertEqual([torch.equal(old, actual) for old, actual in zip(original, model.parameters())],
                         [True, True, False, False, False, False])
        self.assertEqual({int(value["step"].item()) for value in optimizer.state.values()}, {2})

    def test_smoke_records_frozen_stem_and_both_updated_parameters(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
                self.classifier = torch.nn.Sequential(torch.nn.Linear(4, 2))

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

        torch.manual_seed(73)
        model = Toy()
        modules, optimizer = configure_last_stage(model)
        data = torch.utils.data.TensorDataset(torch.randn(128, 4), torch.arange(128) % 2)
        info = SimpleNamespace(root=None, targets=None, augmentation="basic")
        args = SimpleNamespace(batch_size=8, workers=0)
        with (patch("v13.training.init_cuda"), patch("v13.training.TrainingImages", return_value=data),
              patch.object(torch.Tensor, "cuda", lambda self, **_: self),
              patch.object(torch.cuda, "reset_peak_memory_stats"), patch.object(torch.cuda, "synchronize"),
              patch.object(torch.cuda, "max_memory_reserved", return_value=0),
              patch.object(torch, "autocast", lambda *a, **k: nullcontext())):
            result = checkpoint_stats("base", model, modules, optimizer, data, info, 123, args, "smoke")
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["stem_max_delta"], 0)
        self.assertEqual(result["gradients_ok"], {"last_stage": True, "head": True})
        self.assertGreater(result["head_max_delta"], 0)
        self.assertGreater(result["last_stage_max_delta"], 0)

    def test_time_budget_is_checked_at_measured_optimizer_update(self):
        import torch

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        args = SimpleNamespace(batch_size=8, accum=8)
        timer = iter((0.0, 1.5))

        def run(*_):
            optimizer.step()
            raise AssertionError("expected cap before second update")

        with patch("v13.training.train_epoch", side_effect=run):
            with self.assertRaises(BudgetExceeded) as raised:
                counted_epoch(model, [model], optimizer, None, range(128), args, 2026, 2, 1,
                              clock=lambda: next(timer))
        self.assertEqual(raised.exception.count, 1)

    def test_official_model_schemas_use_the_same_final_stage_policy(self):
        import torch
        from torchvision.ops import StochasticDepth

        from v13.training import make_arm

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for arm in ARMS:
                model = make_arm(arm, 50, Path("unused"))
                modules, optimizer = configure_last_stage(model)
                self.assertIs(modules[0], model.features[-1])
                self.assertTrue(all(p.requires_grad for p in model.features[-1].parameters()))
                self.assertFalse(any(p.requires_grad for p in model.features[:-1].parameters()))
                self.assertTrue(all(layer.p == 0 for layer in model.modules() if isinstance(layer, StochasticDepth)))
                self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-3, 1e-4])
                del modules, optimizer, model
        finally:
            torch.set_num_threads(threads)
