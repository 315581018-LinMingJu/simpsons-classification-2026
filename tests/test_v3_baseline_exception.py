import copy
import hashlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from classification_v3 import arguments, check_test_baseline
from v3.baseline_exception import ALLOWED_ID, CURRENT_LABEL, HISTORICAL_LABEL, authorize_exception, baseline_result
from v3.baseline_probe import read_cache
from v3.inference import Prediction
from v3.reports import submission
from v3.storage import Cache, json_bytes


class BaselineExceptionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.output = self.root / "new"
        self.output.mkdir()
        self.cache = self.source / "test.sqlite3"
        self.probe_path = self.root / "probe.json"
        self.classes = [CURRENT_LABEL, HISTORICAL_LABEL, "a", "b", "c"]
        self.ids = ["003", ALLOWED_ID, "004"]
        self.samples = [(Path(id_ + ".jpg"), id_) for id_ in self.ids]
        self.reference = {item: 1 if item == ALLOWED_ID else 0 for item in self.ids}
        self.predictions = {item: Prediction(0, 0.163805, 0.0, 0.001) for item in self.ids}
        self.metadata = {
            "cache_version": 2, "reference_baseline_batch_size": 2, "classes": self.classes,
            "v2_checkpoint_sha256": "a" * 64,
        }
        self.reference_hash = "b" * 64
        self.settings = {**self.metadata, "v2_reference_sha256": self.reference_hash}
        with closing(sqlite3.connect(self.cache)) as connection:
            Cache(connection, self.metadata).add("baseline", self.predictions.items())
        (self.source / "baseline_check.json").write_bytes(json_bytes(
            baseline_result(self.ids, self.predictions, self.reference, self.settings)
        ))
        audit = read_cache(self.source, self.metadata, self.ids, 2, self.classes, ALLOWED_ID,
                           self.reference_hash)
        replay = {
            "input_tensor_sha256": "f" * 64, "target_matches_cached_baseline": True,
            "batch_reference_mismatch_ids": [ALLOWED_ID], "logits_dtype": "float32",
            "top1": CURRENT_LABEL,
            "top5": [{"character": name, "logit": 0.5 - i * 0.001,
                      "probability": 0.164 - i * 0.0004} for i, name in enumerate(self.classes)],
        }
        self.report = {
            "id": ALLOWED_ID, "historical_v2_label": HISTORICAL_LABEL,
            "v2_reference_sha256": self.reference_hash, "expected_cache_metadata": self.metadata,
            "cache": audit,
            "replay_settings": {
                "batch_size": 2, "actual_batch_size": 2, "batch_ids": self.ids[:2],
                "batch_start_index_zero_based": 0, "position_in_batch_zero_based": 1,
                "cudnn_benchmark": True, "cudnn_deterministic": False,
            },
            "replay_comparison": {
                "batch_input_bitwise_equal": True, "batch_logits_bitwise_equal": True,
                "batch_logit_max_abs_diff": 0.0, "target_logit_max_abs_diff": 0.0,
            },
            "replays": {"v2": replay, "v3": copy.deepcopy(replay)},
        }

    def authorize(self, report=None, allowed_id=ALLOWED_ID):
        self.probe_path.write_bytes(json_bytes(self.report if report is None else report))
        return authorize_exception(self.source, self.probe_path, allowed_id, self.ids,
                                   self.reference, self.metadata, self.reference_hash)

    def test_verified_exception_writes_audit_and_keeps_model_prediction(self):
        original = {path.name: path.read_bytes() for path in self.source.iterdir()}
        baseline, authorization = self.authorize()
        self.assertEqual(baseline, self.predictions)
        check_test_baseline(self.output, self.samples, baseline, self.reference, self.settings,
                            authorization)
        filename = self.output / "baseline_exception_authorization.json"
        record = json.loads(filename.read_bytes())
        self.assertEqual(record["allowed_id"], ALLOWED_ID)
        self.assertEqual(record["historical_label"], HISTORICAL_LABEL)
        self.assertEqual(record["current_label"], CURRENT_LABEL)
        self.assertEqual(record["v2_checkpoint_sha256"], self.metadata["v2_checkpoint_sha256"])
        self.assertEqual(record["probe_report_sha256"],
                         hashlib.sha256(self.probe_path.read_bytes()).hexdigest())
        candidate = self.output / "submission_v3_baseline.csv"
        submission(candidate, self.ids, baseline, self.classes, count=len(self.ids))
        self.assertIn(f"155,{CURRENT_LABEL}", candidate.read_text())
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, original)
        with self.assertRaisesRegex(RuntimeError, "Baseline differs"):
            check_test_baseline(self.output, self.samples, baseline, self.reference, self.settings)

    def test_other_id_or_second_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only ID 155"):
            self.authorize(allowed_id="003")
        self.reference["004"] = 1
        with self.assertRaisesRegex(ValueError, "sole mismatch"):
            self.authorize()

    def test_missing_or_inconsistent_probe_evidence_is_rejected(self):
        for group, key, bad_value, error in (
            ("replay_comparison", "batch_input_bitwise_equal", False, "bitwise"),
            ("replay_comparison", "batch_logit_max_abs_diff", False, "bitwise"),
            ("replay_comparison", "batch_logits_bitwise_equal", None, "bitwise"),
            ("cache", "integrity_check", [], "cache field"),
            ("cache", "baseline_target", None, "cache field"),
            ("replays", "v2", None, "both replays"),
            ("replay_settings", "batch_ids", list(reversed(self.ids[:2])), "original batch"),
            ("expected_cache_metadata", "v2_checkpoint_sha256", "c" * 64, "checkpoint"),
        ):
            with self.subTest(group=group, key=key):
                report = copy.deepcopy(self.report)
                if bad_value is None:
                    del report[group][key]
                else:
                    report[group][key] = bad_value
                with self.assertRaisesRegex(ValueError, error):
                    self.authorize(report)
        report = copy.deepcopy(self.report)
        report["replays"]["v2"].pop("top5")
        with self.assertRaisesRegex(ValueError, "top-five"):
            self.authorize(report)

    def test_live_cache_changes_are_rejected_even_if_probe_passed(self):
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE predictions SET confidence = 0.2 WHERE id = '155'")
        with self.assertRaisesRegex(ValueError, "baseline_target"):
            self.authorize()

    def test_cli_requires_explicit_and_separate_source(self):
        with patch.object(sys, "argv", ["program", "predict"]):
            self.assertIsNone(arguments().allow_historical_mismatch_id)
        options = ["--allow-historical-mismatch-id", ALLOWED_ID, "--baseline-probe-report",
                   str(self.probe_path), "--baseline-source", str(self.source)]
        for argv in (
            ["program", "predict", *options[:2]],
            ["program", "predict", *options, "--output", str(self.source)],
            ["program", "predict", "--allow-historical-mismatch-id", "003"],
            ["program", "diagnostic", *options],
        ):
            with (self.subTest(argv=argv), patch.object(sys, "argv", argv),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                arguments()
        with patch.object(sys, "argv", ["program", "predict", *options, "--output", str(self.output)]):
            self.assertEqual(arguments().allow_historical_mismatch_id, ALLOWED_ID)
