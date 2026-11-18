import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from diagnose_v5 import run_challenge, validation_inputs
from v2.data_checks import Sample
from v3.inference import Prediction
from v3.storage import file_sha256, image_signature
from v5.challenge import PARENTS, SIZE, scene
from v5.diagnostics import evaluate_challenge, legacy_audit


def source_image(color):
    image = Image.new("RGB", (224, 224))
    ImageDraw.Draw(image).rectangle((20, 28, 193, 187), fill=color)
    return image


class V5ChallengeTests(unittest.TestCase):
    def test_context_variants_change_only_the_designed_region(self):
        source = source_image("gold")
        secondary = source_image("blue")
        variants = {case: scene(source, case, 20261119, "homer/frame", secondary) for case in PARENTS}
        original, _ = variants["clean"]
        self.assertIs(original, source)
        cgeom, ccontext, clipped, distractor, negative = (
            (np.asarray(variants[name][0]), variants[name][1])
            for name in ("geometry", "context", "clipped", "distractor", "negative")
        )
        self.assertEqual(cgeom[0].shape, (SIZE, SIZE, 3))
        area = ccontext[1]["primary_alpha_fraction"]
        self.assertEqual(cgeom[1]["primary_alpha_fraction"], area)
        self.assertEqual(negative[1]["primary_alpha_fraction"], area)
        self.assertEqual(distractor[1]["primary_alpha_fraction"], area)
        self.assertLess(clipped[1]["primary_alpha_fraction"], area)
        self.assertGreater(distractor[1]["secondary_alpha_fraction"], 0)
        self.assertFalse(np.array_equal(ccontext[0], cgeom[0]))
        self.assertTrue(np.any(distractor[0] != ccontext[0]))
        np.testing.assert_array_equal(ccontext[0][0], clipped[0][0])
        np.testing.assert_array_equal(ccontext[0][0], negative[0][0])
        np.testing.assert_array_equal(ccontext[0][0], distractor[0][0])
        again, _ = scene(source, "context", 20261119, "homer/frame")
        np.testing.assert_array_equal(again, ccontext[0])

    def test_audit_finds_different_error_ids_at_equal_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paired_predictions.csv").write_text(
                "condition,relative_path,true_character,v2_prediction,v5_prediction\n"
                "clean,a.jpg,homer,marge,homer\n"
                "clean,b.jpg,marge,marge,homer\n"
                "heldout_background,a.jpg,homer,homer,marge\n"
                "heldout_background,b.jpg,marge,homer,marge\n"
            )
            (root / "validation_samples.csv").write_text(
                "relative_path,label,matte_fraction\na.jpg,homer,0.24\nb.jpg,marge,0.4\n"
            )
            report = legacy_audit(root)
            self.assertEqual(report["paired_images"], 2)
            self.assertEqual(report["conditions"]["clean"]["wins"][0]["true_character"], "homer")
            self.assertEqual(report["conditions"]["clean"]["wins"][0]["matte_fraction"], 0.24)
            self.assertEqual(report["conditions"]["heldout_background"]["losses"][0]["relative_path"],
                             "a.jpg")
            self.assertFalse(report["background_changes_by_model"]["v2"]["same_error_ids"])
            self.assertEqual(report["background_changes_by_model"]["v2"]["prediction_changes"], 2)
            with (root / "paired_predictions.csv").open("a") as file:
                file.write("clean,a.jpg,homer,marge,homer\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                legacy_audit(root)

    def test_challenge_statistics_report_ceiling_and_paired_drops(self):
        targets = {"a": 0, "b": 1}
        correct = {item: Prediction(label, 0.8, 0, 0) for item, label in targets.items()}
        wrong = {item: Prediction(1 - label, 0.6, 0, 0) for item, label in targets.items()}
        predictions = {(condition, name): correct for condition in PARENTS for name in ("v2", "v5")}
        predictions["context", "v2"] = wrong
        result = evaluate_challenge(targets, predictions, 44, {"a": 0.1, "b": 0.24})
        self.assertTrue(result["geometry"]["near_ceiling"])
        self.assertFalse(result["context"]["near_ceiling"])
        self.assertEqual(result["context"]["v2_drop_vs_parent_macro_pp"]["pp"], 100)
        self.assertEqual(result["context"]["v5_gain_change_vs_parent_macro_pp"]["ci95_pp"], [100, 100])
        self.assertEqual(result["context"]["foreground_size_slices"]["alpha_at_most_12_pct"]["images"], 1)

    def test_challenge_uses_original_split_and_writes_only_new_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            train_root = data / "train/train/train"
            classes = ["homer", "marge"]
            rows = []
            for label, name in enumerate(classes):
                class_dir = train_root / name
                class_dir.mkdir(parents=True)
                for part in ("val", "train"):
                    path = class_dir / f"{part}.png"
                    source_image(("gold", "blue")[label]).save(path)
                    rows.append(Sample(path, label, part))
            v2, v5 = root / "v2", root / "v5"
            v2.mkdir()
            v5.mkdir()
            (v2 / "best.pt").write_bytes(b"v2")
            (v5 / "epoch_02.pt").write_bytes(b"v5")
            signature = image_signature(row.path for row in rows)
            evaluation = "original_eval"
            settings = {
                "validation_samples": 2, "train_image_signature": signature, "classes": classes,
                "split_sha256": "split", "method": "multiscale", "eval_transform": repr(evaluation),
                "v2_checkpoint_sha256": file_sha256(v2 / "best.pt"),
            }
            (v5 / "settings.json").write_text(json.dumps(settings))
            (v5 / "validation.json").write_text(json.dumps({"checkpoint_sha256": file_sha256(v5 / "epoch_02.pt")}))
            (v5 / "validation_subset.json").write_text(json.dumps({"image_signature": signature,
                                                                  "excluded_ids": []}))
            manifest = v5 / "validation_samples.csv"
            good_manifest = "relative_path,label,matte_fraction\nhomer/val.png,homer,0.3\nmarge/val.png,marge,0.2\n"
            manifest.write_text(good_manifest.replace("homer/val.png", "homer/train.png"))
            args = SimpleNamespace(data=data, v1_run=root / "v1", v2_run=v2, v5_run=v5,
                                   output=root / "challenge", seed=44, per_class=1,
                                   batch_size=2, workers=0)
            with self.assertRaisesRegex(ValueError, "Invalid V5 validation item"):
                validation_inputs(args, rows, train_root, classes, settings)
            manifest.write_text(good_manifest)
            immutable = {path: path.read_bytes() for folder in (v2, v5) for path in folder.iterdir()}
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "synthetic"),
                utils=SimpleNamespace(data=SimpleNamespace(DataLoader=lambda dataset, **kwargs: dataset)),
                backends=SimpleNamespace(cudnn=SimpleNamespace()),
                version=SimpleNamespace(cuda="test"), __version__="test",
            )
            model = Mock()
            model.eval.return_value = model

            def infer(model, dataset):
                return [[(item, Prediction(int(item.startswith("marge")), 0.8, 0, 0))
                         for _, item in dataset]]

            with (
                patch.dict(sys.modules, {"torch": fake_torch, "torchvision": SimpleNamespace(__version__="test")}),
                patch("diagnose_v5.load_comparison_split", return_value=(classes, rows, "split")),
                patch("diagnose_v5.transforms_v2", return_value=(None, evaluation)),
                patch("diagnose_v5.Views", return_value=lambda image: image),
                patch("diagnose_v5.create_model", return_value=model),
                patch("diagnose_v5.load_checkpoint"),
                patch("diagnose_v5.infer_loader", side_effect=infer) as mocked_infer,
            ):
                run_challenge(args)
            self.assertEqual(mocked_infer.call_count, len(PARENTS) * 2)
            self.assertEqual({path: path.read_bytes() for path in immutable}, immutable)
            with (args.output / "sample_manifest.csv").open() as file:
                self.assertTrue(all(row["source_label"] != row["distractor_label"]
                                    for row in csv.DictReader(file)))
            self.assertEqual(json.loads((args.output / "challenge.json").read_text())["samples"], 2)
            self.assertTrue((args.output / "challenge_examples.png").is_file())
