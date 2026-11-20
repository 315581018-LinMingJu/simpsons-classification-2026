import io
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import classification_v16 as v16
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import Prediction
from v3.reports import read_reference
from v3.storage import file_sha256, json_bytes
from v8.data import CONDITIONS
from v16.probabilities import Progress, cache_or_infer, fuse, read_cache
from v16.report import screen


def view_batches(ids, probabilities):
    for i, item in enumerate(ids):
        yield [item], np.tile(probabilities[i:i + 1], (10, 1, 1)), None, 0


class ProbabilitiesTest(unittest.TestCase):
    def test_exact_mean_reuse_and_identity(self):
        ids = ["9", "2"]
        views = np.zeros((10, 2, 2), dtype=np.float32)
        views[:6] = [.6, .4]
        views[6:] = [.1, .9]
        with tempfile.TemporaryDirectory() as storage:
            path = Path(storage) / "arm.npz"
            metadata = {"arm": "a"}
            got, _, reused = cache_or_infer(
                path, ids, 2, metadata, lambda: [(ids, views, None, 0)], "fixture")
            self.assertFalse(reused)
            np.testing.assert_array_equal(got, views.mean(axis=0))
            with self.assertRaises(ValueError):
                read_cache(path, ids[::-1], 2, metadata)
            with self.assertRaises(ValueError):
                read_cache(path, ids, 2, {"arm": "b"})
            with patch("v16.probabilities.Progress", side_effect=AssertionError("inference on cache hit")):
                same, _, reused = cache_or_infer(path, ids, 2, metadata, None, "fixture")
            self.assertTrue(reused)
            np.testing.assert_array_equal(same, got)
            with np.load(path, allow_pickle=False) as saved:
                options = dict(saved)
            options["probabilities"] = options["probabilities"][:, ::-1]
            np.savez(path, **options)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_cache(path, ids, 2, metadata)

    def test_incomplete_or_out_of_order_does_not_publish(self):
        p = np.array([[.2, .8], [.3, .7]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as storage:
            path = Path(storage) / "arm.npz"
            for broken, error in ((["a"], "Incomplete"), (["b", "a"], "batch ID order")):
                with self.assertRaisesRegex(ValueError, error):
                    cache_or_infer(path, ["a", "b"], 2, {}, lambda: view_batches(broken, p), "fixture")
                self.assertFalse(path.exists())

    def test_progress_tracks_completed_images(self):
        stream = io.StringIO()
        clock = iter([0, 2, 5]).__next__
        progress = Progress("V16 B test", 5, stream=stream, clock=clock)
        progress.advance(2)
        progress.advance(3)
        lines = stream.getvalue().splitlines()
        self.assertIn("0/5", lines[0])
        self.assertIn("2/5 40.00%", lines[1])
        self.assertIn("ETA 3.0s", lines[1])
        self.assertIn("5/5 100.00%", lines[2])
        self.assertIn("ETA 0.0s", lines[2])

    def test_soft_voting_keeps_probability_magnitudes(self):
        a = np.array([[.54, .46], [.9, .1]], dtype=np.float32)
        b = np.array([[.1, .9], [.1, .9]], dtype=np.float32)
        result = fuse(a, b, .8)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result.argmax(1), [1, 0])


class ScreenTest(unittest.TestCase):
    def test_default_and_predeclared_alternative(self):
        labels = np.tile([0, 1], 12)
        p = np.eye(2, dtype=np.float32)[labels]
        matrices = {condition: p for condition in CONDITIONS}
        default = screen(matrices, matrices, labels, 3)
        self.assertEqual(default["weight_a"], .8)
        self.assertTrue(default["reused_screen"])
        self.assertFalse(default["independent_validation"])
        a = {**matrices, "context": p * .25 + (1 - p) * .75}
        b = {**matrices, "context": p * .96 + (1 - p) * .04}
        alternative = screen(a, b, labels, 3)
        self.assertEqual(alternative["weight_a"], .5)
        row = alternative["trials_vs_a"]["a50"]["conditions"]["context"]
        self.assertEqual((row["wins"], row["losses"]), (24, 0))
        self.assertGreater(row["accuracy_effect"]["ci95_pp"][0], 0)
        self.assertEqual(alternative["trials_vs_a"]["b"]["conditions"]["context"]["disagreement"]["pp"], 100)

    def test_clean_regression_blocks_alternative(self):
        labels = np.tile([0, 1], 10)
        p = np.eye(2, dtype=np.float32)[labels]
        matrices = {condition: p for condition in CONDITIONS}
        a = {**matrices, "context": p * .25 + (1 - p) * .75, "clean": .62 * p + .38 * (1 - p)}
        b = {**matrices, "context": p * .96 + (1 - p) * .04, "clean": .04 * p + .96 * (1 - p)}
        self.assertEqual(screen(a, b, labels, 3)["weight_a"], .8)


class ArchiveTest(unittest.TestCase):
    def test_checkpoint_and_archive_sha_contract(self):
        import torch

        with tempfile.TemporaryDirectory() as storage:
            folder = Path(storage) / "a"
            archive = folder / "candidate_epoch_03"
            archive.mkdir(parents=True)
            epoch = folder / "epoch_03.pt"
            settings = {"arm": "a", "protocol": v16.v14.FORMAT, "prepare_sha256": "parent", "environment": {}}
            training = {"epoch": 3, "loss": .04}
            torch.save({"epoch": 3, "training": training, "identity": settings,
                        "model": {"classifier.2.weight": torch.zeros(2, 768)}}, epoch)
            (folder / "settings.json").write_bytes(json_bytes(settings))
            (folder / "history.json").write_bytes(json_bytes([{**training, "checkpoint_sha256": file_sha256(epoch)}]))
            csv = archive / "submission.csv"
            csv.write_text("id,character\n9,x\n2,y\n")
            test = SimpleNamespace(provenance={"template_sha256": "template"})
            summary = {"arm": "a", "epoch": 3, "checkpoint_sha256": file_sha256(epoch),
                       "submission_sha256": file_sha256(csv), "test_provenance": test.provenance,
                       "images": 2, "method": "multiscale", "denoise": True, "batch_size": 8}
            (archive / "summary.json").write_bytes(json_bytes(summary))
            args = SimpleNamespace(v14_run=Path(storage), archive_a=None)
            loaded = v16.load_arm(args, "a", "parent", test, ["9", "2"], ["x", "y"])
            self.assertEqual(loaded.archive, {"9": 0, "2": 1})
            self.assertEqual(loaded.fingerprint["test_batch_size"], 8)
            csv.write_text("id,character\n9,y\n2,y\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                v16.load_arm(args, "a", "parent", test, ["9", "2"], ["x", "y"])
            epoch.write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                v16.load_arm(args, "a", "parent", test, ["9", "2"], ["x", "y"])

    def test_full_size_output_has_exact_order_and_legal_classes(self):
        ids = [str(i) for i in range(TEST_COUNT - 1, -1, -1)]
        classes = [f"character_{i:02d}" for i in range(50)]
        predictions = {item: Prediction(i % 50, .6, 0, 0) for i, item in enumerate(ids)}
        with tempfile.TemporaryDirectory() as storage:
            csv = Path(storage) / "submission_v16.csv"
            v16.submission(csv, ids, predictions, classes)
            indexed = read_reference(csv, ids, classes)
            self.assertEqual(list(indexed), ids)
            self.assertEqual(set(indexed.values()), set(range(50)))
            with self.assertRaises(ValueError):
                v16.submission(csv, ids[:-1], predictions, classes)


class CandidateTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name)
        self.args = SimpleNamespace(output=output)
        (output / "prepare.json").write_text("{}")
        self.labels = ["x", "y"]
        self.ids = ["9", "02"]
        self.a = np.array([[.54, .46], [.9, .1]], dtype=np.float32)
        self.b = np.array([[.1, .9], [.1, .9]], dtype=np.float32)
        arm_a = SimpleNamespace(archive={"9": 0, "02": 0}, fingerprint={"checkpoint_sha256": "a"})
        arm_b = SimpleNamespace(archive={"9": 1, "02": 1}, fingerprint={"checkpoint_sha256": "b"})
        self.source = SimpleNamespace(ids={"test": self.ids, "screen": self.ids}, classes=self.labels,
                                      arms={"a": arm_a, "b": arm_b},
                                      identity={"screen_labels": [0, 1], "screen_batch_size": 8},
                                      info=SimpleNamespace(inputs={"seed": 4}))
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.collect = stack.enter_context(patch.object(v16, "collect", side_effect=self.collect_fixture))
        stack.enter_context(patch.object(v16, "verify_unchanged"))
        stack.enter_context(patch.object(v16, "TEST_COUNT", 2))
        stack.enter_context(patch.object(v16, "read_cache", return_value=(self.a, 1)))
        v16.validate(self.args, self.source)

    def collect_fixture(self, args, source, name, scope):
        if scope == "screen":
            return ({condition: self.a for condition in CONDITIONS}, {condition: 1 for condition in CONDITIONS})
        return ({None: self.a if name == "a" else self.b}, {"all": 1})

    def test_candidate_after_both_replays(self):
        v16.predict(self.args, self.source)
        path = self.args.output / "candidate"
        self.assertEqual(read_reference(path / "submission_v16.csv", self.ids, self.labels), {"9": 1, "02": 0})
        report = read_json(path / "summary.json")
        self.assertEqual(report["replay_mismatches"], {"a": 0, "b": 0})
        self.assertEqual(report["changed_vs_a"], 1)
        self.assertIn("x,y,1", (path / "transitions.csv").read_text())

    def test_b_mismatch_stops_before_writing_candidate(self):
        self.source.arms["b"].archive["02"] = 0
        with self.assertRaisesRegex(SystemExit, "B Top-1"):
            v16.predict(self.args, self.source)
        self.assertFalse((self.args.output / "candidate").exists())
        self.assertIn("02,x,y", (self.args.output / "replay_mismatch_b.csv").read_text())

    def test_edited_weight_stops_before_test(self):
        path = self.args.output / "validation.json"
        report = read_json(path)
        report["weight_a"] = .5
        path.write_bytes(json_bytes(report))
        self.collect.reset_mock()
        with self.assertRaisesRegex(ValueError, "verified Screen"):
            v16.predict(self.args, self.source)
        self.collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
