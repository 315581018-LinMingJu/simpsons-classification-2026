import io
import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import classification_v16 as v16
import classification_v9 as v9
from v3 import inference
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import Prediction
from v3.reports import read_reference, read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes
from v8.data import CONDITIONS, Case, selected_signature
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
            test = SimpleNamespace(provenance={"template_sha256": "template"}, archive_settings={"batch_size": 8})
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


class V14WriterTest(unittest.TestCase):
    def test_real_writer_and_missing_denoise_archive_reach_each_stage(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                                                      torch.nn.Linear(768, 50))

            def forward(self, images):
                channels = images.mean(dim=(2, 3))
                return self.classifier(torch.nn.functional.pad(channels, (0, 765)))

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as storage, ExitStack() as stack:
            root = Path(storage)
            stack.callback(os.chdir, Path.cwd())
            os.chdir(root)
            data, v8_run, v14_run = Path("data/extracted"), Path("runs/v8_full"), Path("runs/v14_continuation")
            output = Path("runs/v16/test_run").resolve()
            v8_csv = v8_run / "candidates/convnext_tta_denoise/submission_v8.csv"
            v8_csv.parent.mkdir(parents=True)
            v8_checkpoint = v8_run / "models/convnext_all_layers/best.pt"
            v8_checkpoint.parent.mkdir(parents=True)
            v8_checkpoint.write_bytes(b"synthetic parent")
            (v8_run / "benchmark").mkdir()
            (v8_run / "benchmark/prepare.json").write_bytes(b"{}")
            v14_run.mkdir()
            ids = ["9", "02", "21", "7", "3"]
            classes = [f"character_{i:02d}" for i in range(50)]
            test_dir = data / "test_images/test_images"
            test_dir.mkdir(parents=True)
            for i, item in enumerate(ids):
                image = Image.new("RGB", (250 + i, 290), (10 * i, 20 + i, 5))
                ImageDraw.Draw(image).rectangle((21 + 9 * i, 64, 190, 230), fill=(220, 140, 40))
                image.save(test_dir / f"{item}.jpg", format="PNG")
            template = data / "sample_submission.csv"
            template.write_text("id,character\n" + "".join(f"{item},{classes[0]}\n" for item in ids))
            shutil.copyfile(template, v8_csv)
            batch_sizes = {"a": 3, "b": 4}
            v8_settings = {"checkpoint_sha256": file_sha256(v8_checkpoint), "template_sha256": file_sha256(template),
                           "test_image_signature": image_signature(test_dir / f"{item}.jpg" for item in ids),
                           "submission_sha256": file_sha256(v8_csv), "architecture": "convnext_tiny",
                           "method": "multiscale", "denoise": True, "images": len(ids),
                           "batch_size": batch_sizes["a"], "origin": "synthetic"}
            (v8_csv.parent / "settings.json").write_bytes(json_bytes(v8_settings))
            cases = [Case("9.jpg", 0, "screen", "02.jpg"), Case("02.jpg", 1, "screen", "9.jpg")]
            inputs = {"classes": classes, "split_sha256": "split", "seed": 316}
            context = SimpleNamespace(settings=v8_settings, checkpoint=v8_checkpoint, screen=cases, root=test_dir,
                                      inputs=inputs, identity={
                                          "checkpoint_sha256": v8_settings["checkpoint_sha256"],
                                          "v8_submission_sha256": v8_settings["submission_sha256"],
                                          "screen_signature": selected_signature(test_dir, cases)})
            plan = v14_run / "prepare.json"
            plan.write_bytes(json_bytes({"identity": context.identity, "eval_batch_size": 8,
                                         "screen_images": len(cases), "screen_ids": [c.item for c in cases]}))
            report = {"inputs": inputs, "manifest_sha256": {"validation": "validation"}}

            cpu_zeros = torch.zeros
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(v9, "benchmark", return_value=(report, [], cases)))
            stack.enter_context(patch.object(v9, "read_test_images", side_effect=lambda *a: read_test_images(*a, count=len(ids))))
            stack.enter_context(patch.object(v16.v14, "context", return_value=context))
            stack.enter_context(patch.object(v16.v14, "init_cuda"))
            stack.enter_context(patch.object(v16, "init_cuda"))
            for module in (v16.v14, v16):
                stack.enter_context(patch.object(module, "make_model", side_effect=lambda *_: SmallModel()))
            stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda self: self))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda self, **_: self))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(patch.object(torch, "zeros", side_effect=lambda shape, **_: cpu_zeros(shape)))
            stack.enter_context(patch.object(v16.v14, "submission", side_effect=lambda *a: submission(*a, count=len(ids))))
            stack.enter_context(patch.object(v16, "TEST_COUNT", len(ids)))

            reference, current = [], []
            infer = inference.infer_view_batches

            def trace(records, *args, **kwargs):
                for batch in infer(*args, **kwargs):
                    if kwargs.get("warmup", True):
                        records.append((list(batch[0]), batch[1].copy()))
                    yield batch

            stack.enter_context(patch.object(inference, "infer_view_batches",
                                             side_effect=lambda *a, **kw: trace(reference, *a, **kw)))
            stack.enter_context(patch.object(v16, "infer_view_batches", side_effect=lambda *a, **kw: trace(current, *a, **kw)))
            archives = {}
            for arm, epoch in v16.EPOCHS.items():
                folder = v14_run / arm
                folder.mkdir()
                settings = {"arm": arm, "prepare_sha256": file_sha256(plan), "protocol": v16.v14.FORMAT}
                (folder / "settings.json").write_bytes(json_bytes(settings))
                model = SmallModel()
                for i in range(1, epoch + 1):
                    torch.save({"epoch": i, "identity": settings, "model": model.state_dict(),
                                "training": {"epoch": i, "loss": .1, "selection": {"eligible": True, "score": .9}}},
                               folder / f"epoch_{i:02d}.pt")
                (folder / "history.json").write_bytes(json_bytes(v16.v14.read_history(folder, settings)))
                args = SimpleNamespace(arm=arm, epoch=epoch, v8_run=v8_run, data=data, workers=0,
                                       predict_batch_size=None if arm == "a" else batch_sizes[arm])
                v16.v14.predict(args, context, folder, settings)
                archives[arm] = folder / f"candidate_epoch_{epoch:02d}"

            full = read_json(archives["b"] / "summary.json")
            self.assertEqual(set(full), {"arm", "epoch", "checkpoint_sha256", "submission_sha256", "images",
                                        "changed_vs_v8", "changed_pct", "seconds", "submission", "method",
                                        "denoise", "batch_size", "test_provenance"})
            self.assertIs(full["denoise"], True)
            compat = root / "legacy_b"
            shutil.copytree(archives["b"], compat)
            summary = compat / "summary.json"
            legacy = {k: value for k, value in full.items() if k != "denoise"}
            argv = ["--output", str(output), "--archive-b", str(compat), "--allow-imagenet", "--workers", "0"]
            for bad in (False, None, 1):
                with self.subTest(denoise=bad):
                    summary.write_bytes(json_bytes({**legacy, "denoise": bad}))
                    with self.assertRaisesRegex(ValueError, "archived denoise"):
                        v16.prepare_inputs(v16.arguments(["prepare", *argv]))
            summary.write_bytes(json_bytes(legacy))
            original = {p: file_sha256(p) for p in (summary, *[d / "summary.json" for d in archives.values()])}
            for stage in ("prepare", "probe", "validate", "predict"):
                with (patch("sys.argv", ["classification_v16.py", stage, *argv]),
                      patch.object(v16, "environment", return_value={"cuda_available": True, "devices_visible": 1}),
                      patch.object(torch.cuda, "mem_get_info", return_value=(2**33, 2**34))):
                    self.assertEqual(v16.main(), 0)

            prepare = read_json(output / "prepare.json")
            self.assertEqual(prepare["arms"]["a"]["denoise_source"], "summary")
            self.assertEqual(prepare["arms"]["b"]["denoise_source"], "verified_v14_protocol")
            self.assertEqual([len(items) for items, _ in reference], [3, 2, 4, 1])
            for (old_ids, old), (new_ids, new) in zip(reference, current, strict=True):
                self.assertEqual(old_ids, new_ids)
                self.assertEqual(new.shape, (10, len(new_ids), 50))
                self.assertEqual(new.dtype, np.float32)
                np.testing.assert_array_equal(old, new)
            for arm, start in (("a", 0), ("b", 2)):
                with np.load(output / f"cache/test/{arm}/all.npz", allow_pickle=False) as cache:
                    expected = np.concatenate([p.mean(axis=0) for _, p in reference[start:start + 2]])
                    np.testing.assert_array_equal(cache["probabilities"], expected)
            result = read_json(output / "candidate/summary.json")
            self.assertEqual(result["replay_mismatches"], {"a": 0, "b": 0})
            self.assertEqual(list(read_reference(output / "candidate/submission_v16.csv", ids, classes)), ids)
            self.assertEqual({p: file_sha256(p) for p in original}, original)


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
