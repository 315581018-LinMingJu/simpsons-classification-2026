import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

import compare_v1_v2 as compare
from v2 import data_checks


def example_image():
    image = Image.new("RGB", (96, 80), "royalblue")
    ImageDraw.Draw(image).ellipse((20, 15, 69, 59), fill="gold")
    return image


class ComparisonTests(unittest.TestCase):
    def test_split_and_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1, v2, train = (root / name for name in ("v1", "v2", "images"))
            v1.mkdir()
            v2.mkdir()
            (train / "character").mkdir(parents=True)
            classes = ["character"]
            split = (b"path,label,part\ncharacter/a.png,0,train\n"
                     b"character/b.png,0,val\ncharacter/c.png,0,val\n")
            digest = hashlib.sha256(split).hexdigest()
            for run in v1, v2:
                (run / "split.csv").write_bytes(split)
                (run / "classes.json").write_text(json.dumps(classes))
            example_image().save(train / "character/a.png", compress_level=0)
            example_image().save(train / "character/b.png", compress_level=9)
            Image.new("RGB", (96, 80), "white").save(train / "character/c.png")

            with patch.multiple(data_checks, CLASS_COUNT=1, COUNTS={"train": 1, "val": 2}):
                found_classes, rows, found_hash = compare.load_comparison_split(train, v1, v2, digest)
                self.assertEqual((found_classes, found_hash), (classes, digest))
                report, _, exposed = data_checks.duplicate_diagnostic(rows)
                self.assertEqual(report["cross_partition_pairs"], 1)
                self.assertEqual(exposed, {str(train / "character/b.png")})

                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    compare.load_comparison_split(train, v1, v2, "0" * 64)
                (v2 / "split.csv").write_bytes(split + b"\n")
                with self.assertRaisesRegex(ValueError, "V2 split"):
                    compare.load_comparison_split(train, v1, v2, digest)
                (v2 / "split.csv").write_bytes(split)
                (v2 / "classes.json").write_text('["another_character"]')
                with self.assertRaisesRegex(ValueError, "class mappings"):
                    compare.load_comparison_split(train, v1, v2, digest)

    def test_v2_training_input_loader_still_uses_validated_split(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            v1 = data / "v1"
            train = data / "train/train/train/character"
            test = data / "test_images/test_images"
            for folder in v1, train, test:
                folder.mkdir(parents=True)
            for path in train / "a.jpg", train / "b.jpg", test / "001.jpg":
                example_image().save(path)
            (v1 / "split.csv").write_text(
                "path,label,part\ncharacter/a.jpg,0,train\ncharacter/b.jpg,0,val\n"
            )
            (v1 / "classes.json").write_text('["character"]')
            (v1 / "run_config.json").write_text(json.dumps({
                "architecture": "resnet18", "optimizer": "AdamW", "lr": 0.0001,
                "weight_decay": 0.0001, "seed": 2026, "epochs": 1, "batch_size": 2,
                "train_transform": "RandomResizedCrop224+RandomHorizontalFlip",
                "eval_transform": "Resize256+CenterCrop224",
            }))
            (v1 / "history.json").write_text('[{"epoch":1,"val_accuracy":1,"seconds":1}]')
            template = data / "sample_submission.csv"
            template.write_text("id,character\n001,ignored\n")
            with patch.multiple(data_checks, CLASS_COUNT=1, TEST_COUNT=1,
                                COUNTS={"train": 1, "val": 1}):
                classes, _, _, rows, test_ids, split = data_checks.load_inputs(data, v1, template)
            self.assertEqual(classes, ["character"])
            self.assertEqual([row.part for row in rows], ["train", "val"])
            self.assertEqual(test_ids, [(test / "001.jpg", "001")])
            self.assertEqual(split, (v1 / "split.csv").read_bytes())

    def test_checkpoint_formats_and_metadata(self):
        classes = ["character"]
        state = {"fc.weight": object()}
        v1 = {"model": state, "classes": classes, "epoch": 1, "best_acc": 0.9}
        v2 = {**v1, "split_sha256": "expected"}
        model = Mock()
        checkpoint_load = Mock(side_effect=[v1, v2, v1, {**v2, "classes": ["wrong"]}])
        with patch.dict(sys.modules, {"torch": SimpleNamespace(load=checkpoint_load)}):
            self.assertEqual(compare.load_checkpoint(model, Path("v1.pt"), classes), v1)
            self.assertEqual(compare.load_checkpoint(model, Path("v2.pt"), classes, "expected"), v2)
            with self.assertRaisesRegex(ValueError, "different split"):
                compare.load_checkpoint(model, Path("v2.pt"), classes, "expected")
            with self.assertRaisesRegex(ValueError, "Class order"):
                compare.load_checkpoint(model, Path("v2.pt"), classes, "expected")
        model.load_state_dict.assert_called_with(state, strict=True)
        self.assertEqual(model.load_state_dict.call_count, 2)
        checkpoint_load.assert_called_with(Path("v2.pt"), map_location="cpu", weights_only=True)

    def test_perturbations_are_repeatable_and_precede_preprocessing(self):
        original = example_image()
        for condition in compare.PERTURBATIONS:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "example.png"
                original.save(path)
                preprocess = Mock(side_effect=lambda image: image.resize((48, 40)))
                samples = [data_checks.Sample(path, 0, "val")]
                dataset = compare.ValidationImages(samples, root, condition, 2026, preprocess)
                result, label = dataset[0]
                disturbed = preprocess.call_args.args[0]
                self.assertEqual(label, 0)
                self.assertEqual(disturbed.size, original.size)
                self.assertEqual(result.size, (48, 40))
                self.assertNotEqual(disturbed.tobytes(), original.tobytes())
                self.assertEqual(dataset[0][0].tobytes(), result.tobytes())
                self.assertNotEqual(
                    disturbed.tobytes(), compare.perturb(original, condition, 42, path.name).tobytes()
                )

    def test_paired_scoring_and_drop(self):
        images = object()
        models = [object(), object()]
        labels = [0, 1, 2, 3]
        batches = [(images, labels)]

        def calculate(condition, scope, predictions, clean=None):
            forward = Mock(side_effect=[(predictions[0], 0.2), (predictions[1], 0.4)])
            result = compare.score_condition(models, batches, forward, condition, scope, clean)
            self.assertIs(forward.call_args_list[0].args[1], images)
            self.assertIs(forward.call_args_list[1].args[1], images)
            return result

        clean = calculate("clean", "no_exact_train_match", ([0, 1, 2, 3], [0, 1, 1, 3]))
        self.assertIsNone(clean["v2_drop_from_clean_pp"])
        result = calculate("grayscale", "no_exact_train_match", ([0, 0, 1, 3], [0, 1, 0, 3]), clean)
        values = {
            "samples": 4, "v1_accuracy_pct": 50, "v2_accuracy_pct": 75,
            "v2_minus_v1_pp": 25, "prediction_disagreement_pct": 50,
            "v1_drop_from_clean_pp": 50, "v2_drop_from_clean_pp": 0,
            "v1_inference_seconds": 0.2, "v2_inference_seconds": 0.4,
        }
        for key, expected in values.items():
            self.assertEqual(result[key], expected)

    def test_reports_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.png"
            example_image().save(path)
            examples = [data_checks.Sample(path, 0, "val")]
            scope = "no_exact_train_match"
            batches = [(object(), [0])]
            models = [0, 1]

            def forward(model, _):
                return [model], 0.01

            clean = compare.score_condition(models, batches, forward, "clean", scope)
            results = [compare.score_condition(models, batches, forward, "clean_full", "full"), clean]
            results.extend(compare.score_condition(models, batches, forward, condition, scope, clean)
                           for condition in compare.PERTURBATIONS)
            output = root / "reports"
            args = (output, {"seed": 2026}, results, examples, root, 2026, lambda image: image)
            compare.write_reports(*args)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"comparison.csv", "comparison.json", "comparison.png", "examples.png"})
            self.assertEqual(json.loads((output / "comparison.json").read_text())["results"], results)
            with (output / "comparison.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), len(results))
            for name in ("comparison.png", "examples.png"):
                with Image.open(output / name) as image:
                    image.verify()
            with self.assertRaises(FileExistsError):
                compare.write_reports(*args)


if __name__ == "__main__":
    unittest.main()
