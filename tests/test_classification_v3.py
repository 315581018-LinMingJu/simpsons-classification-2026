import csv
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v3 import main, predict_cached, validation_subset
from v2.data_checks import Sample
from v3.inference import Prediction, Views, summarize_views
from v3.reports import read_reference, read_test_images, submission, test_report, validation_report
from v3.storage import Cache, image_signature


def sample_prediction(label=0, confidence=0.75, disagreement=0.0):
    return Prediction(label, confidence, disagreement, 0.02)


class ClassificationV3Tests(unittest.TestCase):
    def test_baseline_uses_original_transform_and_view_counts(self):
        class Resize:
            def __init__(self, size):
                self.size = size

            def __call__(self, image):
                if isinstance(self.size, tuple):
                    height, width = self.size
                else:
                    width, height = (round(axis * self.size / min(image.size)) for axis in image.size)
                return image.resize((width, height))

        class FiveCrop:
            def __init__(self, size):
                self.size = size

            def __call__(self, image):
                size = self.size
                right, bottom = image.width - size, image.height - size
                positions = [(0, 0), (right, 0), (0, bottom), (right, bottom),
                             (right // 2, bottom // 2)]
                return [image.crop((x, y, x + size, y + size)) for x, y in positions]

        def compose(transforms):
            def apply(image):
                for transform in transforms:
                    image = transform(image)
                return image
            return apply

        evaluation = Mock(return_value="original V2 tensor")
        evaluation.transforms = [None, None, lambda image: image]
        image = Image.new("RGB", (320, 240), "navy")
        ImageDraw.Draw(image).rectangle((0, 0, 112, 96), fill="yellow")
        transforms = SimpleNamespace(Resize=Resize, FiveCrop=FiveCrop, Compose=compose)
        with patch.dict(sys.modules, {"torchvision": SimpleNamespace(transforms=transforms)}):
            self.assertEqual(Views("baseline", evaluation)(image), ("original V2 tensor",))
            evaluation.assert_called_once_with(image)
            full, = Views("full_resize", evaluation)(image)
            self.assertEqual(full.size, (224, 224))
            five = Views("five_crop", evaluation)(image)
            multiscale = Views("multiscale", evaluation)(image)
            self.assertEqual((len(five), len(multiscale)), (5, 10))
            self.assertTrue(all(crop.size == (224, 224) for crop in (*five, *multiscale)))
            self.assertNotEqual(five[0].tobytes(), five[1].tobytes())
            self.assertNotEqual(multiscale[0].tobytes(), multiscale[5].tobytes())

    def test_softmax_probability_average_and_single_view_argmax(self):
        probabilities = np.array([[[0.51, 0.49]], [[0.51, 0.49]], [[0.01, 0.99]]])
        pred, = summarize_views(probabilities, np.array([[0], [0], [1]]), 0.5)
        self.assertEqual(pred.label, 1)
        self.assertAlmostEqual(pred.confidence, (0.49 + 0.49 + 0.99) / 3)
        self.assertAlmostEqual(pred.view_disagreement, 1 / 3)
        self.assertEqual(pred.seconds, 0.5)
        # Softmax 捨入平手仍沿用 logits 的 top-1。
        tied, = summarize_views(np.array([[[0.5, 0.5]]]), np.array([[1]]), 0.2)
        self.assertEqual(tied.label, 1)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize_views(np.array([[[np.nan, 0.5]]]), np.array([[1]]), 0.2)

    def test_cache_reuse_between_diagnostic_predict_and_rerun(self):
        samples = [(Path("003.jpg"), "003"), (Path("002.jpg"), "002")]
        make_model = Mock(return_value=object())
        inference = Mock(side_effect=lambda model, rows, method: [
            [(item, sample_prediction()) for _, item in rows]
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite3"
            for subset in (samples[:1], samples, samples):
                with closing(sqlite3.connect(path)) as connection:
                    cache = Cache(connection, {"checkpoint": "a"})
                    results = predict_cached(cache, subset, ("baseline",), make_model, inference)
                    self.assertEqual(list(results["baseline"]), [item for _, item in subset])
            self.assertEqual(inference.call_count, 2)
            self.assertEqual(inference.call_args.args[1], samples[1:])
            self.assertEqual(make_model.call_count, 2)
            with closing(sqlite3.connect(path)) as connection:
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    Cache(connection, {"checkpoint": "b"})

    def test_validation_subset_reuses_exact_duplicate_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [Sample(root / name, 0, part) for name, part in
                    (("train.png", "train"), ("duplicate.png", "val"), ("unique.png", "val"))]
            for row, color in zip(rows, ("red", "red", "blue"), strict=True):
                Image.new("RGB", (4, 4), color).save(row.path)
            signature = image_signature(row.path for row in rows)
            samples, targets, counts = validation_subset(root, rows, signature, root)
            self.assertEqual(samples, [(rows[2].path, "unique.png")])
            self.assertEqual(targets, {"unique.png": 0})
            self.assertEqual(counts["cross_partition_pairs"], 1)
            with patch("classification_v3.duplicate_diagnostic", side_effect=AssertionError):
                self.assertEqual(validation_subset(root, rows, signature, root)[0], samples)
            Image.new("RGB", (4, 4), "red").save(rows[2].path)
            with self.assertRaisesRegex(ValueError, "images changed"):
                validation_subset(root, rows, image_signature(row.path for row in rows), root)

    def test_submission_preserves_all_ids_and_refuses_overwrite(self):
        ids = [f"{i:05d}" for i in reversed(range(10791))]
        predictions = {item: sample_prediction(i % 2) for i, item in enumerate(ids)}
        classes = ["homer", "marge"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            submission(path, ids, predictions, classes)
            original = path.read_bytes()
            submission(path, ids, predictions, classes)
            self.assertEqual(list(read_reference(path, ids, classes).items()),
                             [(item, predictions[item].label) for item in ids])
            with path.open(newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0], ["id", "character"])
            self.assertEqual([row[0] for row in rows[1:]], ids)
            bad = dict(predictions, **{ids[0]: sample_prediction(2)})
            with self.assertRaisesRegex(ValueError, "class index"):
                submission(path, ids, bad, classes)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                submission(path, [ids[1], *ids[1:]], predictions, classes)
            predictions[ids[0]] = sample_prediction(1)
            with self.assertRaises(FileExistsError):
                submission(path, ids, predictions, classes)
            self.assertEqual(path.read_bytes(), original)

    def test_test_manifest_ignores_sample_characters_and_checks_image_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (16, 16))
            for item in ("001", "000"):
                image.save(root / f"{item}.jpg")
            template = root / "sample.csv"
            template.write_text("id,character\n001,unused\n000,nonsense\n")
            self.assertEqual(read_test_images(root, template, count=2),
                             [(root / "001.jpg", "001"), (root / "000.jpg", "000")])
            image.save(root / "002.jpg")
            with self.assertRaisesRegex(ValueError, "extra"):
                read_test_images(root, template, count=2)

    def test_cli_reuses_diagnostic_and_gates_candidates_on_v2_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            test = data / "test_images/test_images"
            test.mkdir(parents=True)
            ids = ["009", "002"]
            for item in ids:
                Image.new("RGB", (16, 16), "yellow").save(test / f"{item}.jpg")
            (data / "sample_submission.csv").write_text("id,character\n009,unused\n002,unused\n")
            classes = [f"character_{i}" for i in range(50)]
            v1, v2 = (root / name for name in ("v1", "v2"))
            for run in v1, v2:
                run.mkdir()
                (run / "classes.json").write_text(json.dumps(classes))
                (run / "split.csv").write_bytes(b"test split")
                (run / "best.pt").write_bytes(run.name.encode())
            reference = v2 / "submission_v2.csv"
            reference.write_text("id,character\n009,character_0\n002,character_0\n")
            evaluation = "test transform"
            (v2 / "run_config.json").write_text(json.dumps({
                "eval_transform": repr(evaluation), "batch_size": 64,
            }))
            cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "fake CUDA")
            cudnn = SimpleNamespace(version=lambda: 1)
            torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(cudnn=cudnn),
                                    version=SimpleNamespace(cuda="test"), __version__="test")
            vision = SimpleNamespace(__version__="test")
            def infer(model, samples, method, evaluation, batch_size, workers, *, warmup=True):
                self.assertEqual(cudnn.benchmark, not warmup)
                self.assertEqual(cudnn.deterministic, warmup)
                self.assertEqual(batch_size, 16 if warmup else 64)
                return [[(item, sample_prediction(int(method == "full_resize")))
                         for _, item in samples]]

            fake_infer = Mock(side_effect=infer)

            def run(stage, output):
                argv = ["classification_v3.py", stage, "--data", str(data), "--v1-run", str(v1),
                        "--v2-run", str(v2), "--output", str(output), "--limit", "1"]
                with patch.object(sys, "argv", argv):
                    main()

            stages = "classification_v3."
            with (
                patch.dict(sys.modules, {"torch": torch, "torchvision": vision}),
                patch(stages + "SPLIT_SHA256", hashlib.sha256(b"test split").hexdigest()),
                patch(stages + "transforms_v2", return_value=(None, evaluation)),
                patch(stages + "create_model"),
                patch(stages + "load_checkpoint"),
                patch(stages + "infer_batches", fake_infer),
                patch(stages + "read_test_images",
                      side_effect=lambda folder, template: read_test_images(folder, template, count=2)),
                patch(stages + "submission",
                      side_effect=lambda path, ids, preds, classes: submission(
                          path, ids, preds, classes, count=2)),
                patch("v3.reports.montage"),
            ):
                output = root / "v3"
                run("diagnostic", output)
                self.assertEqual(fake_infer.call_count, 5)
                run("baseline-check", output)
                self.assertEqual(fake_infer.call_count, 6)
                baseline = fake_infer.call_args
                self.assertEqual([item for _, item in baseline.args[1]], ids)
                self.assertEqual(baseline.args[2], "baseline")
                self.assertEqual(baseline.kwargs, {"warmup": False})
                self.assertFalse(list(output.glob("submission_v3_*.csv")))
                check = json.loads((output / "baseline_check.json").read_text())
                self.assertEqual(check["mismatch_count"], 0)
                run("predict", output)
                self.assertEqual(fake_infer.call_count, 10)
                self.assertTrue(all(len(call.args[1]) == 1 for call in fake_infer.call_args_list[6:]))
                self.assertEqual(len(list(output.glob("submission_v3_*.csv"))), 4)
                run("predict", output)
                self.assertEqual(fake_infer.call_count, 10)
                reference.write_text("id,character\n009,character_1\n002,character_0\n")
                mismatch = root / "mismatch"
                with self.assertRaisesRegex(RuntimeError, "Baseline differs"):
                    run("predict", mismatch)
                self.assertEqual(fake_infer.call_count, 11)
                self.assertEqual(fake_infer.call_args.args[2], "baseline")
                self.assertTrue(cudnn.deterministic)
                self.assertFalse(cudnn.benchmark)
                self.assertFalse(list(mismatch.glob("submission_v3_*.csv")))
                self.assertFalse((mismatch / "test_summary.json").exists())
                check = json.loads((mismatch / "baseline_check.json").read_text())
                self.assertEqual(check["mismatch_ids"], ["009"])


    def test_reports_render_and_measure_without_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "001.jpg"
            Image.new("RGB", (30, 20), "yellow").save(path)
            base = {"001": sample_prediction()}
            changed = {"001": sample_prediction(1, 0.6, 0.4)}
            predictions = {"baseline": base, "full_resize": changed,
                           "five_crop": changed, "multiscale": base}
            results = validation_report(root, {"001": 1}, predictions, {})
            self.assertEqual(results[1]["delta_vs_baseline_pp"], 100)
            self.assertEqual(results[1]["changed_pct"], 100)
            self.assertEqual(results[1]["mean_confidence"], 0.6)
            summary = test_report(root, [(path, "001")], {**predictions, "v1_baseline": changed},
                                  ["homer", "marge"], {"001": 1}, {})
            self.assertEqual(summary["baseline_reference_mismatch_ids"], ["001"])
            self.assertEqual(summary["v1_v2_changed_count"], 1)
            self.assertEqual(summary["methods"]["five_crop"]["view_disagreements"], 1)
            for filename in ("low_confidence.png", "prediction_disagreements.png",
                             "tta_disagreements.png", "v1_v2_disagreements.png"):
                with Image.open(root / filename) as image:
                    image.verify()
            self.assertEqual(json.loads((root / "comparison.json").read_text())["results"], results)


if __name__ == "__main__":
    unittest.main()
