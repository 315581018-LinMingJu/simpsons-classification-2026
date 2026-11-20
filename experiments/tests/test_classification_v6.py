import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from classification_v6 import CONDITIONS, CompositeTrain, paired_results, predict, preview, train
from distribution_audit.features import FEATURES, visual_features
from v3.inference import Prediction
from v3.reports import submission
from v3.storage import file_sha256
from v6.composite import Targets, composite, training_project, warp


def frame():
    image = Image.new("RGB", (224, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 26, 199, 196), fill="gold")
    draw.ellipse((50, 40, 187, 150), fill="red")
    return image


def prediction(label):
    return Prediction(label, 0.9, 0, 0)


class V6ExperimentTests(unittest.TestCase):
    def test_composite_is_deterministic_and_preserves_full_panel(self):
        source = frame()
        before = source.tobytes()
        targets = Targets(0.46, 0.21, 0.031)
        a, area = composite(source, targets, np.random.default_rng(47))
        b, repeat = composite(source, targets, np.random.default_rng(47))
        held, _ = composite(source, targets, np.random.default_rng(47), held_out=True)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(area, repeat)
        self.assertNotEqual(a.tobytes(), held.tobytes())
        self.assertEqual(source.tobytes(), before)
        self.assertGreaterEqual(area, 0.1)
        self.assertLess(area, 0.7)
        measurements = dict(zip(FEATURES, visual_features(a), strict=True))
        self.assertLess(measurements["black_edge_fraction"], 0.2)
        self.assertAlmostEqual(measurements["brightness"], targets.brightness, delta=0.09)
        rgba = Image.new("RGBA", (100, 60), "red")
        rendered = warp(rgba, training_project(rgba.size, np.random.default_rng(50)))
        x0, y0, x1, y1 = rendered.getchannel("A").getbbox()
        self.assertTrue(0 < x0 < x1 < 264 and 0 < y0 < y1 < 264)

    def test_augmented_inputs_are_not_randomly_recropped(self):
        training_transform = Mock()
        training_transform.transforms = list(range(7))
        compose = Mock(return_value=Mock(return_value="tensor"))
        torchvision = SimpleNamespace(transforms=SimpleNamespace(Compose=compose,
                                                                Resize=lambda size: ("resize", size)))
        with patch.dict(sys.modules, {"torchvision": torchvision}), patch("random.random", return_value=0.3):
            using_scene = CompositeTrain(training_transform, Targets(0.46, 0.21, 0.031))
            self.assertEqual(using_scene(frame()), "tensor")
        self.assertEqual(compose.call_args.args[0], [("resize", (224, 224)), 4, 5, 6])
        training_transform.assert_not_called()

    def test_decision_compares_with_both_references_and_guards_ceiling(self):
        labels = {"a": 0, "b": 1}
        good = {item: prediction(y) for item, y in labels.items()}
        bad = {item: prediction(1 - y) for item, y in labels.items()}
        scores = {(name, condition): good for name in ("v5", "control", "v6") for condition in CONDITIONS}
        self.assertEqual(paired_results(labels, scores, 7)["decision"], "heldout_near_ceiling")
        scores["control", "heldout_composite"] = bad
        scores["v5", "heldout_composite"] = bad
        supported = paired_results(labels, scores, 7)
        self.assertEqual(supported["decision"], "supports_composite")
        self.assertEqual(supported["conditions"]["heldout_composite"]["vs_control"]["wins"], 2)
        scores["v6", "clean"] = bad
        self.assertEqual(paired_results(labels, scores, 7)["decision"], "clean_regression")
        scores["v6", "clean"] = good
        scores["v6", "heldout_composite"] = bad
        self.assertEqual(paired_results(labels, scores, 7)["decision"],
                         "no_evidence_beyond_extra_training")

    def test_preview_reports_common_feature_groups_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            frame().save(path)
            args = SimpleNamespace(output=Path(directory) / "v6", seed=20261119)
            summary = {"reference_fit": {"bins_from_train_p20_p80": {
                "brightness": [0.2, 0.35], "contrast": [0.15, 0.27], "fine_detail": [0.005, 0.012],
            }}}
            preview(args, [(path, "homer/frame.png")], Targets(0.46, 0.21, 0.031), summary)
            report_path = args.output / "preview/preview.json"
            report = json.loads(report_path.read_text())
            self.assertEqual(report["images"], 1)
            self.assertLess(report["comparisons"]["heldout_composite"]["medians"]["black_edge_fraction"],
                            report["comparisons"]["clean"]["medians"]["black_edge_fraction"])
            self.assertTrue((args.output / "preview/composite_examples.png").is_file())

    def test_two_arms_reload_same_weights_and_candidate_uses_only_v6(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root / "v6", data=root, seed=11, batch_size=32,
                                   workers=0, train_per_class=2, per_class=1)
            source = root / "v5.pt"
            source.write_bytes(b"original checkpoint")
            images = [(root / "a.jpg", "1"), (root / "b.jpg", "2")]
            labels = {"1": 0, "2": 1}
            training = [(path, item, labels[item]) for path, item in images]
            classes = ["homer", "bart"]
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(get_device_name=lambda: "synthetic CPU"),
                optim=SimpleNamespace(AdamW=Mock()), nn=SimpleNamespace(CrossEntropyLoss=Mock()),
                manual_seed=Mock(), version=SimpleNamespace(cuda="fake"), __version__="fake",
            )
            fake_model = Mock()
            fake_model.eval.return_value = fake_model
            scores = {condition: {"1": prediction(0), "2": prediction(1)} for condition in CONDITIONS}

            def write_model(path, checkpoint):
                path.write_bytes(path.name.encode())

            with (
                patch.dict(sys.modules, {"torch": fake_torch, "torchvision": SimpleNamespace(__version__="fake")}),
                patch("classification_v6.create_model", return_value=fake_model),
                patch("classification_v6.load_checkpoint") as load,
                patch("classification_v6.CompositeTrain"),
                patch("classification_v6.loader") as loaders,
                patch("classification_v6.train_epoch", side_effect=[0.3, 0.2]) as epochs,
                patch("classification_v6.validate", return_value=scores),
                patch("classification_v6.save_checkpoint", side_effect=write_model),
                patch("classification_v6.read_test_images", return_value=images),
                patch("classification_v6.infer_batches", return_value=[[("1", prediction(1)), ("2", prediction(0))]]),
                patch("classification_v6.submission",
                      side_effect=lambda path, ids, preds, names: submission(path, ids, preds, names, count=2)),
            ):
                train(args, classes, "split", training, images, labels, source,
                      Targets(0.46, 0.21, 0.031), "train", "eval", {})
                self.assertEqual(epochs.call_count, 2)
                self.assertEqual([call.args[1] for call in load.call_args_list], [source] * 3)
                self.assertEqual(loaders.call_args_list[0].args[0], loaders.call_args_list[1].args[0])
                self.assertEqual(loaders.call_args_list[0].args[2], loaders.call_args_list[1].args[2])
                self.assertEqual(source.read_bytes(), b"original checkpoint")
                experiment = args.output / "experiment"
                with self.assertRaises(FileExistsError):
                    train(args, classes, "split", training, images, labels, source,
                          Targets(0.46, 0.21, 0.031), "train", "eval", {})
                report = json.loads((experiment / "validation.json").read_text())
                self.assertEqual(report["arms"]["v6"]["checkpoint_sha256"], file_sha256(experiment / "v6.pt"))
                self.assertFalse((experiment / "submission_v6.csv").exists())
                predict(args, classes, "split", "eval")
                self.assertEqual(load.call_args.args[1], experiment / "v6.pt")
                self.assertEqual((experiment / "submission_v6.csv").read_text().splitlines(),
                                 ["id,character", "1,bart", "2,homer"])
                with self.assertRaises(FileExistsError):
                    predict(args, classes, "split", "eval")


if __name__ == "__main__":
    unittest.main()
