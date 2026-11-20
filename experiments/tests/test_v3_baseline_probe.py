import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from v3.baseline_probe import locate_batch, probe, read_cache, top_scores
from v3.inference import Prediction
from v3.storage import Cache, baseline_cache_issues


class BaselineProbeTests(unittest.TestCase):
    def test_locate_original_batches_preserves_id_text_and_short_tail(self):
        ids = ["20", "003", "155", "11", "9"]
        self.assertEqual(locate_batch(ids, "155", 2), (2, 4, 0))
        self.assertEqual(locate_batch(ids, "9", 2), (4, 5, 0))
        with self.assertRaisesRegex(ValueError, "Unknown test ID"):
            locate_batch(ids, "3", 2)

    def test_top5_and_close_margin(self):
        logits = np.array([1.0, 1.00001, -1.5, 0.2, 0.5, -1.0])
        probs = np.exp(logits) / np.exp(logits).sum()
        classes = list("ABCDEF")
        report = top_scores(logits, probs, classes, 0)
        self.assertEqual([row["character"] for row in report["top5"]], list("BAEDF"))
        self.assertAlmostEqual(report["top1_top2_logit_margin"], 0.00001)
        self.assertGreater(report["top1_top2_probability_margin"], 0)
        self.assertEqual(report["historical_label_current_score"]["character"], "A")
        logits[1] = logits[0]
        self.assertEqual(top_scores(logits, probs, classes, 0)["top5"][0]["character"], "A")

    def test_read_only_audit_distinguishes_preview_version_and_batch_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test.sqlite3"
            ids, classes = ["105", "155", "9"], ["kent", "timothy"]
            metadata = {"cache_version": 2, "reference_baseline_batch_size": 2}
            with closing(sqlite3.connect(path)) as conn:
                cache = Cache(conn, metadata)
                first, second = Prediction(0, 0.40, 0, 0.003), Prediction(1, 0.16, 0, 0.003)
                cache.add("baseline", [(ids[0], first), (ids[1], second)])
                cache.add("baseline_preview", [(ids[1], first)])
            (root / "baseline_check.json").write_text(
                '{"settings":{"v2_reference_sha256":"reference"}}'
            )
            original = path.read_bytes()
            checked = read_cache(root, metadata, ids, 2, classes, "155", "reference")
            self.assertTrue(checked["compatible"])
            self.assertEqual(checked["integrity_check"], ["ok"])
            self.assertEqual(checked["baseline_target"]["character"], "timothy")
            self.assertEqual(checked["preview_target"]["character"], "kent")
            self.assertTrue(checked["prior_baseline_check_reference_matches"])
            self.assertEqual(path.read_bytes(), original)
            older = read_cache(root, {**metadata, "cache_version": 3}, ids, 2, classes, "155", "new")
            self.assertFalse(older["compatible"])
            self.assertEqual(older["changed_metadata_keys"], ["cache_version"])
            self.assertFalse(older["prior_baseline_check_reference_matches"])
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("DELETE FROM predictions WHERE method='baseline' AND id='105'")
            gap = read_cache(root, metadata, ids, 2, classes, "155", "reference")
            self.assertFalse(gap["compatible"])
            self.assertIn("complete original batches", gap["baseline_structure_issues"][0])

    def test_cache_prefix_accepts_full_short_tail_but_rejects_invalid_fields(self):
        good = Prediction(0, 0.25, 0, 0.002)
        ids = ["a", "b", "c"]
        self.assertEqual(baseline_cache_issues(dict.fromkeys(ids, good), ids, 2, 2), [])
        self.assertTrue(baseline_cache_issues({"a": good}, ids, 2, 2))
        bad = Prediction(4, float("inf"), 0, 0.1)
        self.assertTrue(baseline_cache_issues({"a": bad, "b": good}, ids, 2, 2))

    def test_probe_replays_original_batch_with_both_loaders_and_prints_json(self):
        classes = list("ABCDEF")
        ids = ["8", "1", "155", "2", "9"]
        samples = [(Path(item + ".jpg"), item) for item in ids]
        config = {"batch_size": 2, "seed": 2026, "workers": 4}
        args = SimpleNamespace(output=Path("unused"), probe_id="155", replay_prefix=True,
                               probe_path="both", workers=2)
        logits = np.array([[0.51, 0.52, -0.1, 0.2, 0.1, 0.0],
                           [1, -1, 0.1, 0.2, 0, -0.5]], dtype=np.float32)
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        cudnn = SimpleNamespace(benchmark=False, deterministic=True, allow_tf32=True)
        torch = SimpleNamespace(
            backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False))),
            manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()),
            get_default_dtype=lambda: "torch.float32", get_float32_matmul_precision=lambda: "highest",
        )
        stdout = io.StringIO()
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch("v3.baseline_probe.read_cache", return_value={
                "baseline_target": {"label": 1},
            }),
            patch("v3.baseline_probe.v2_loader") as v2_loader,
            patch("v3.baseline_probe.view_loader") as v3_loader,
            patch("v3.baseline_probe.replay", return_value=("equal", logits, probabilities)) as replay,
            redirect_stdout(stdout),
        ):
            probe(args, samples, classes, dict.fromkeys(ids, 0), config, object(), {}, "reference",
                  Mock())
        v2_loader.assert_called_once()
        self.assertEqual(v2_loader.call_args.args[0], samples[:4])
        self.assertEqual(v3_loader.call_args.args[0], samples[:4])
        self.assertEqual(replay.call_args.args[-2:], (1, ["155", "2"]))
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["replay_settings"]["dataset_last_batch_size"], 1)
        self.assertEqual(result["historical_v2_label"], "A")
        self.assertFalse(result["historical_v2_scores_available"])
        self.assertEqual(result["replays"]["v2"]["top1"], "B")
        self.assertTrue(result["replays"]["v3"]["target_matches_cached_baseline"])
        self.assertTrue(result["replay_comparison"]["batch_input_bitwise_equal"])
        self.assertEqual(result["replay_comparison"]["target_logit_max_abs_diff"], 0)
