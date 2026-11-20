import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v5 import BackgroundValidation, paired_outcome, predict, train
from v2.data_checks import Sample
from v3.inference import Prediction
from v3.reports import submission
from v3.storage import file_sha256
from v5.background import exterior_mask, replace_background


def sample_image():
    image = Image.new("RGB", (100, 100))
    painter = ImageDraw.Draw(image)
    painter.rectangle((15, 15, 84, 84), fill=(226, 191, 42))
    painter.rectangle((45, 45, 53, 56), fill=(0, 0, 0))
    return image


def label_prediction(label):
    return Prediction(label, 0.85, 0.0, 0.01)


class BackgroundExperimentTests(unittest.TestCase):
    def test_only_connected_near_black_padding_changes(self):
        image = sample_image()
        data = np.asarray(image)
        mask = exterior_mask(data)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[49, 49])
        self.assertTrue((data[mask] <= 24).all())
        new = np.asarray(replace_background(image, np.random.default_rng(41)))
        np.testing.assert_array_equal(new[~mask], data[~mask])
        np.testing.assert_array_equal(np.asarray(image), data)
        self.assertTrue((new[mask] != data[mask]).any())
        for color in ("black", "white"):
            uniform = Image.new("RGB", (100, 100), color)
            self.assertFalse(exterior_mask(np.asarray(uniform)).any())
            self.assertIs(replace_background(uniform, np.random.default_rng(41)), uniform)

    def test_heldout_background_is_deterministic_and_distinct(self):
        image = sample_image()
        train_view = replace_background(image, np.random.default_rng(41))
        heldout = replace_background(image, np.random.default_rng(41), held_out=True)
        self.assertNotEqual(train_view.tobytes(), heldout.tobytes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            image.save(path)
            dataset = BackgroundValidation([(path, "character/frame.png")], lambda im: im, 41, True)
            a, item = dataset[0]
            b, _ = dataset[0]
            self.assertEqual(item, "character/frame.png")
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_paired_outcome_requires_stress_improvement_and_clean_guard(self):
        targets = {"a": 0, "b": 1}
        good = {item: label_prediction(label) for item, label in targets.items()}
        bad = {item: label_prediction(1 - label) for item, label in targets.items()}
        baseline = {"clean": good, "heldout_background": bad}
        candidate = {"clean": good, "heldout_background": good}
        report = paired_outcome(targets, baseline, candidate, 41)
        self.assertEqual(report["decision"], "supports_background_adaptation")
        self.assertEqual(report["conditions"]["heldout_background"]["wins"], 2)
        self.assertEqual(report["conditions"]["heldout_background"]["gain_macro_pp"]["ci95_pp"],
                         [100, 100])
        self.assertEqual(paired_outcome(targets, baseline, baseline, 41)["decision"],
                         "does_not_support_background_adaptation")
        self.assertEqual(paired_outcome(targets, baseline, {**candidate, "clean": bad}, 41)["decision"],
                         "clean_regression")

    def test_mocked_train_and_predict_write_new_outputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "v5"
            args = SimpleNamespace(output=run, data=root, seed=2026, batch_size=16, workers=0,
                                   per_class=1)
            source = root / "v2.pt"
            source.write_bytes(b"V2 model")
            val_path = root / "val.png"
            sample_image().save(val_path)
            classes = ["homer", "marge"]
            targets = {"homer/frame.png": 0, "marge/frame.png": 1}
            samples = [(val_path, item) for item in targets]
            train_rows = [Sample(val_path, 0, "train")]
            good = {item: label_prediction(label) for item, label in targets.items()}
            scores = {"clean": good, "heldout_background": good}
            device = SimpleNamespace(get_device_name=lambda: "synthetic CPU")
            fake_torch = SimpleNamespace(
                cuda=device, optim=SimpleNamespace(AdamW=Mock()),
                nn=SimpleNamespace(CrossEntropyLoss=Mock()),
                __version__="fake", version=SimpleNamespace(cuda="fake"),
            )
            model = Mock()
            model.eval.return_value = model

            def write_model(path, checkpoint):
                path.write_text(str(checkpoint["epoch"]))

            test_ids = [(val_path, "001"), (val_path, "002")]
            test_scores = [("001", label_prediction(1)), ("002", label_prediction(0))]
            with (
                patch.dict(sys.modules, {"torch": fake_torch, "torchvision": SimpleNamespace(__version__="fake")}),
                patch("classification_v5.paired_data", return_value=(samples, targets, {}, "sig", "fake")),
                patch("classification_v5.create_model", return_value=model),
                patch("classification_v5.load_checkpoint"),
                patch("classification_v5.loader"),
                patch("classification_v5.train_epoch", side_effect=[0.3, 0.2]) as step,
                patch("classification_v5.validate", return_value=scores),
                patch("classification_v5.save_checkpoint", side_effect=write_model),
                patch("classification_v5.read_test_images", return_value=test_ids),
                patch("classification_v5.infer_batches", return_value=[test_scores]),
                patch("classification_v5.submission",
                      side_effect=lambda path, ids, preds, names: submission(path, ids, preds, names, count=2)),
            ):
                train(args, root, classes, train_rows, "split", source, "train", "eval")
                self.assertEqual(step.call_count, 2)
                self.assertEqual(source.read_bytes(), b"V2 model")
                report = json.loads((run / "validation.json").read_text())
                self.assertEqual(report["checkpoint_sha256"], file_sha256(run / "epoch_02.pt"))
                self.assertTrue((run / "validation_samples.csv").is_file())
                with self.assertRaises(FileExistsError):
                    train(args, root, classes, train_rows, "split", source, "train", "eval")
                predict(args, classes, "split", "eval")
                self.assertEqual((run / "submission_v5.csv").read_text().splitlines(),
                                 ["id,character", "001,marge", "002,homer"])
                with self.assertRaises(FileExistsError):
                    predict(args, classes, "split", "eval")
                (run / "epoch_02.pt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    predict(args, classes, "split", "eval")
