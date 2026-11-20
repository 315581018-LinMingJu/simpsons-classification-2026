import importlib.util
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from compare_v8_factors import arguments, earlier_screens, prepare, read_prepared, save_validation
from v3.storage import file_sha256, json_bytes
from v8.data import Case, selected_signature
from v8_controlled.conditions import ARMS, CANVAS, RIM, VIEW_NAMES, blind_views, controlled_frames, crop_retention, kept_views
from v8_controlled.report import CONTRASTS, effects, exact_mcnemar, geometry_summary, holm


def original():
    image = Image.new("RGB", (224, 224))
    draw = ImageDraw.Draw(image)
    draw.ellipse((34, 42, 193, 179), fill=(220, 136, 53))
    draw.rectangle((80, 70, 118, 110), fill=(71, 130, 220))
    return image


class ControlledTests(unittest.TestCase):
    def test_single_factor_masks_and_equal_luminance_detail(self):
        sample = original()
        ex = controlled_frames(sample, 2026, "a/1.jpg")
        self.assertTrue(ex.valid)
        again = controlled_frames(sample, 2026, "a/1.jpg")
        self.assertEqual([ex.frames[key].tobytes() for key in ARMS],
                         [again.frames[key].tobytes() for key in ARMS])
        reference = np.asarray(ex.frames["neutral"])
        rim = np.ones((CANVAS, CANVAS), bool)
        rim[RIM:-RIM, RIM:-RIM] = False
        x0, y0, x1, y1 = ex.bounds["neutral"]
        self.assertTrue(np.all(np.asarray(ex.frames["black_rim"])[~rim] == reference[~rim]))
        for key in ("bright_background", "outer_detail"):
            self.assertTrue(np.all(np.asarray(ex.frames[key])[rim] == reference[rim]))
            self.assertTrue(np.all(np.asarray(ex.frames[key])[y0:y1, x0:x1] == reference[y0:y1, x0:x1]))
        self.assertGreater(np.asarray(ex.frames["bright_background"]).mean(), reference.mean())
        self.assertEqual(np.asarray(ex.frames["outer_detail"]).sum(), reference.sum())

    def test_denoise_change_inside_protected_region_is_rejected(self):
        def fake(image):
            array = np.array(image)
            if array[0, 0, 0] == 0:
                array[CANVAS//2, CANVAS//2] = 0
            return Image.fromarray(array), 0.0

        with patch("v8_controlled.conditions.clean_impulses", side_effect=fake):
            ex = controlled_frames(original(), 2026, "a/1.jpg")
        self.assertFalse(ex.valid)
        self.assertEqual(ex.invalid_reason, "denoise_changes_outside_black_rim")

    def test_blind_control_matches_each_scale_and_does_not_need_another_view(self):
        ex = controlled_frames(original(), 2026, "a/1.jpg")
        selected = kept_views(ex.retention)
        matched = blind_views(selected, 2026, "a/1.jpg")
        for i in (0, 5):
            self.assertEqual(selected[i:i+5].sum(), matched[i:i+5].sum())
        self.assertEqual(matched.tolist(), blind_views(selected, 2026, "a/1.jpg").tolist())
        self.assertLess(np.count_nonzero(selected), len(VIEW_NAMES))
        stats = geometry_summary(ex.retention[None], selected[None], matched[None], [ex.cue_retention])
        self.assertGreater(stats["mean_patch_gain_pp"], 0)
        self.assertGreaterEqual(stats["mean_patch_retained_selected_pct"], 98)
        self.assertEqual(kept_views([.2, .6, .3]).tolist(), [False, True, False])

    def test_paired_effect_holm_and_count_control_stop(self):
        labels = np.repeat([0, 1], 50)
        bad = labels.copy()
        bad[::4] = 1 - bad[::4]
        predictions = {name: labels.copy() for name in (*ARMS, "retained", "matched_count")}
        predictions["bright_background"] = bad
        predictions["offcenter"] = bad
        geometry = {"mean_patch_gain_pp": 10, "mean_patch_gain_vs_matched_pp": 8}
        report = effects(labels, predictions, 2026, geometry)
        self.assertEqual(set(report), set(CONTRASTS))
        self.assertEqual(report["brightness"]["decision"], "accuracy_decreased")
        self.assertEqual(report["content_retention"]["decision"], "improved_vs_v8_but_selection_effect_unresolved")
        self.assertEqual(report["brightness"]["losses"], 25)
        self.assertLess(report["brightness"]["accuracy_effect"]["ci95_pp"][1], 0)
        self.assertEqual(report["black_rim"]["decision"], "stop_no_clear_effect")
        predictions["matched_count"] = bad
        self.assertEqual(effects(labels, predictions, 2026, geometry)["content_retention"]["decision"],
                         "accuracy_improved")
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(holm({"a": .01, "b": .03})["a"], .02)

    def test_flips_with_equal_correctness_stay_separate(self):
        labels = np.repeat([0, 1], 50)
        base = labels.copy()
        base[::5] = 2
        changed = base.copy()
        changed[::5] = 3
        predictions = {name: base.copy() for name in (*ARMS, "retained", "matched_count")}
        predictions["black_rim"] = changed
        report = effects(labels, predictions, 2026,
                         {"mean_patch_gain_pp": 10, "mean_patch_gain_vs_matched_pp": 8})
        self.assertEqual(report["black_rim"]["accuracy_effect"]["pp"], 0)
        self.assertAlmostEqual(report["black_rim"]["flip_rate"]["pp"], 20)
        self.assertEqual(report["black_rim"]["decision"], "prediction_sensitive_accuracy_unresolved")

    def test_prepare_binds_case_labels_and_checks_previous_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root, v9_run=root / "v9", v10_run=root / "v10")
            args.v9_run.joinpath("screen").mkdir(parents=True)
            report = args.v9_run / "screen/summary.json"
            report.write_bytes(json_bytes({"checkpoint_sha256": "old", "benchmark_sha256": "bench",
                                          "decision": "no_candidate_passed"}))
            cases = [Case(f"{arm}{i}.jpg", label, "screen") for label, arm in enumerate(("a", "b")) for i in range(4)]
            for case in cases:
                original().save(root / case.item)
            identity = {"checkpoint_sha256": "ours", "benchmark_sha256": "bench"}
            info = SimpleNamespace(identity=identity, root=root, cases=cases,
                                   inputs={"seed": 2026, "classes": ["a", "b"]})
            self.assertEqual(earlier_screens(args, info)["v9"]["status"], "different_reference")
            with redirect_stdout(StringIO()):
                prepare(args, info)
            result = read_prepared(args, info)
            self.assertEqual(result["decision"], "ready")
            self.assertEqual(result["planned_image_forwards"], 400)
            info.cases[0] = Case(cases[0].item, 1, "screen")
            with self.assertRaisesRegex(ValueError, "labels or IDs"):
                read_prepared(args, info)

    def test_guard_keeps_existing_runs_read_only(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            arguments(["prepare", "--output", "runs/v8_full"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "requires torch and torchvision")
class ControlledTorchTests(unittest.TestCase):
    def test_mask_retention_reproduces_v8_resize_and_fivecrop_order(self):
        from torchvision import transforms as T
        from v3.inference import SCALES

        data = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        data[88:209, 9:176] = 255
        mask = Image.fromarray(data)
        expected = []
        for scale in SCALES:
            scaled = T.Resize(scale)(mask)
            expected.extend(np.asarray(crop).sum() / np.asarray(scaled).sum() for crop in T.FiveCrop(224)(scaled))
        np.testing.assert_allclose(crop_retention(data), expected, atol=0, rtol=0)

    def test_all_three_crop_arms_share_one_forward_bank(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["a.jpg", "b.jpg"]
            cases = [Case(item, i, "screen") for i, item in enumerate(files)]
            for case in cases:
                original().save(root / case.item)
            weight = root / "v8.pt"
            weight.write_bytes(b"synthetic")
            info = SimpleNamespace(root=root, cases=cases, checkpoint=weight,
                                   identity={"checkpoint_sha256": file_sha256(weight),
                                             "screen_sample_signature": selected_signature(root, cases)},
                                   inputs={"seed": 2026, "classes": ["a", "b"]})
            args = SimpleNamespace(allow_imagenet=True, output=root, batch_size=8, workers=0)
            model = torch.nn.Linear(2, 2)
            probabilities = np.tile(np.eye(2, dtype=np.float32)[None], (len(VIEW_NAMES), 1, 1))
            stats = {"image_forwards": 20, "wall_seconds": 1.0}
            with (patch("compare_v8_factors.reference_checkpoint", return_value={"model": model.state_dict()}),
                  patch("compare_v8_factors.make_model", return_value=model),
                  patch("compare_v8_factors.init_cuda"),
                  patch.object(torch.nn.Module, "cuda", lambda self: self),
                  patch.object(torch.cuda, "reset_peak_memory_stats"),
                  patch.object(torch.cuda, "max_memory_reserved", return_value=0),
                  patch("compare_v8_factors.evaluate_bank", return_value=(probabilities, stats)) as forward,
                  redirect_stdout(StringIO())):
                result = save_validation(args, info, {"eligible_ids": files})
            self.assertEqual(forward.call_count, len(ARMS))
            self.assertEqual(result["total_image_forwards"], 100)
            self.assertEqual(result["marginal_image_forwards_vs_reference"]["content_retention"], 0)
            self.assertEqual(result["marginal_image_forwards_vs_reference"]["retention_vs_count_control"], 0)
