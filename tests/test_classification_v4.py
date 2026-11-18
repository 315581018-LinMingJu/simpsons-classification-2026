import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from classification_v4 import benchmark, diagnostic
from v2.data_checks import Sample
from v3.inference import Prediction
from v3.storage import image_signature, json_bytes
from v4.analysis import changes_by_aspect, compare, paired_effect, read_v3_changes
from v4.geometry import CONDITIONS, METHODS, layout, render, retained_fraction, stratified_samples


class V4Tests(unittest.TestCase):
    def test_only_position_changes_between_matched_geometry_conditions(self):
        image = Image.new("RGB", (224, 224), (248, 17, 12))
        aspects = [0.75, 1, 1.33]
        params = layout(image.size, aspects, 2026, "bart/7.jpg")
        self.assertEqual(params, layout(image.size, aspects, 2026, "bart/7.jpg"))
        size, patch, center, offcenter, fill = params
        self.assertNotEqual(center, offcenter)
        self.assertEqual(patch[0], patch[1])
        self.assertIs(render(image, "clean", aspects, 2026, "bart/7.jpg"), image)
        canvases = [render(image, case, aspects, 2026, "bart/7.jpg") for case in CONDITIONS[1:]]
        for canvas in canvases:
            self.assertEqual(canvas.size, size)
            foreground = np.all(np.asarray(canvas) == (248, 17, 12), axis=2)
            self.assertEqual(int(foreground.sum()), patch[0] * patch[1])
            self.assertEqual(canvas.getpixel((0, 0)), fill)
        self.assertGreater(retained_fraction(size, patch, center),
                           retained_fraction(size, patch, offcenter))

    def test_sampling_is_stratified_repeatable_and_unmodified(self):
        samples = [(Path(f"{i}.jpg"), str(i)) for i in range(6)]
        labels = {str(i): int(i > 3) for i in range(6)}
        picked = stratified_samples(samples, labels, 2, 2026, 2)
        self.assertEqual([labels[item] for _, item in picked], [0, 0, 1, 1])
        self.assertEqual(set(picked), set(stratified_samples(samples[::-1], labels, 2, 2026, 2)))
        with self.assertRaisesRegex(ValueError, "validation image"):
            stratified_samples(samples, labels, 2, 2026, 3)

    def test_macro_effect_treats_classes_equally(self):
        effect = paired_effect([1, 1, 1, 0], [0, 0, 0, 1], 10)
        self.assertEqual(effect, {"pp": 50, "ci95_pp": [50, 50]})

    def test_decision_checks_paired_interaction_and_clean_guard(self):
        targets = {"id1": 0, "id2": 1}
        correct = {item: Prediction(y, 0.7, 0, 0.01) for item, y in targets.items()}
        wrong = {item: Prediction(1 - y, 0.7, 0, 0.01) for item, y in targets.items()}
        preds = {(case, method): correct for case in CONDITIONS for method in METHODS}
        _, outcome = compare(targets, preds, 2026)
        self.assertEqual(outcome["decision"], "does_not_support_offcenter_hypothesis")
        preds["offcenter", "baseline"] = wrong
        scores, outcome = compare(targets, preds, 2026)
        self.assertEqual(outcome["decision"], "supports_offcenter_hypothesis")
        self.assertEqual(outcome["offcenter_minus_center_gain"]["pp"], 100)
        score = next(s for s in scores if (s["condition"], s["method"]) == ("offcenter", "multiscale"))
        self.assertEqual((score["wins_vs_baseline"], score["losses_vs_baseline"]), (2, 0))
        preds["clean", "multiscale"] = wrong
        self.assertNotEqual(compare(targets, preds, 2026)[1]["decision"], "supports_offcenter_hypothesis")

    def test_label_free_v3_summary_preserves_ids_and_rejects_partial_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            rows = ["id,method,changed_vs_v2,v1_v2_changed,confidence"]
            for method in ("full_resize", "five_crop", "multiscale"):
                rows.extend((f"001,{method},0,1,0.9", f"002,{method},1,0,0.4"))
            path.write_text("\n".join(rows))
            stats = read_v3_changes(path, ["001", "002"])
            result = changes_by_aspect({"001": (80, 80), "002": (160, 80)}, stats)
            self.assertIn(("landscape", "multiscale", 1, 100, 0, 0.4), result)
            self.assertIn(("near_square", "multiscale", 1, 0, 100, 0.9), result)
            path.write_text("\n".join(rows[:-1]))
            with self.assertRaisesRegex(ValueError, "full V3"):
                read_v3_changes(path, ["001", "002"])

    def test_cpu_diagnostic_and_mocked_benchmark_write_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root / "v4", v2_run=root / "v2", v3_run=root / "v3",
                                   sample_size=2, per_class=1, batch_size=2, workers=0, seed=20)
            args.v2_run.mkdir()
            args.v3_run.mkdir()
            train = root / "train"
            train.mkdir()
            data = [
                ("train_c", "train", 0, (32, 32), "red"),
                ("val_c", "val", 0, (32, 32), "blue"),
                ("train_v", "train", 1, (32, 32), "green"),
                ("val_v", "val", 1, (32, 32), "yellow"),
                ("test_v", "test", None, (96, 40), "orange"),
                ("test_c", "test", None, (50, 96), "cyan"),
            ]
            rows, test = [], []
            for item, part, label, size, color in data:
                path = train / (item + ".png")
                Image.new("RGB", size, color).save(path)
                if part == "test":
                    test.append((path, item))
                else:
                    rows.append(Sample(path, label, part))
            classes = ["character1", "character2"]
            provenance = {
                "split_sha256": "split", "v2_checkpoint_sha256": "checkpoint",
                "test_image_signature": image_signature(path for path, _ in test),
            }
            (args.v3_run / "validation_subset.json").write_bytes(json_bytes({
                "image_signature": image_signature(row.path for row in rows),
                "excluded_ids": [], "duplicate_counts": {"cross_partition_pairs": 0},
            }))
            (args.v3_run / "test_summary.json").write_bytes(json_bytes({"settings": {
                "image_signature": provenance["test_image_signature"],
                "v2_checkpoint_sha256": provenance["v2_checkpoint_sha256"],
            }}))
            rows_csv = ["id,method,changed_vs_v2,v1_v2_changed,confidence"]
            for method in ("full_resize", "five_crop", "multiscale"):
                rows_csv.extend(f"{item},{method},1,0,0.5" for _, item in test)
            (args.v3_run / "test_diagnostics.csv").write_text("\n".join(rows_csv))
            diagnostic(args, classes, rows, train, test, provenance)
            out = args.output / "diagnostic"
            report = json.loads((out / "diagnostic.json").read_text())
            self.assertEqual(report["v3_diagnostics"]["status"], "matched")
            with (out / "sample_stats.csv").open() as file:
                self.assertTrue(all(not row["known_label"] for row in csv.DictReader(file)
                                    if row["split"] == "test"))
            with (out / "test_review.csv").open() as file:
                review = list(csv.DictReader(file))
            self.assertEqual({row["id"] for row in review if row["selection"] == "random"},
                             {"test_v", "test_c"})
            self.assertTrue(all(row["multiple_characters"] == "" for row in review))
            for name in ("split_montage.png", "test_cases.png"):
                with Image.open(out / name) as image:
                    image.verify()
            evaluation = "V2 eval"
            (args.v2_run / "run_config.json").write_text(json.dumps({"eval_transform": repr(evaluation)}))
            torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "Synthetic"),
                backends=SimpleNamespace(cudnn=SimpleNamespace()),
                utils=SimpleNamespace(data=SimpleNamespace(DataLoader=lambda dataset, **_: dataset)),
                version=SimpleNamespace(cuda="test"), __version__="test",
            )
            vision = SimpleNamespace(__version__="test")

            def inferred(model, dataset):
                return [[(item, Prediction(int(item == "val_v.png"), 0.9, 0, 0.01))
                         for _, item in dataset]]

            with (
                patch.dict(sys.modules, {"torch": torch, "torchvision": vision}),
                patch("classification_v4.transforms_v2", return_value=(None, evaluation)),
                patch("classification_v4.create_model"),
                patch("classification_v4.load_checkpoint"),
                patch("classification_v4.Views", return_value=lambda im: im),
                patch("classification_v4.infer_loader", side_effect=inferred) as infer,
            ):
                benchmark(args, classes, rows, train, provenance)
                with self.assertRaises(FileExistsError):
                    benchmark(args, classes, rows, train, provenance)
            self.assertEqual(infer.call_count, 9)
            out = args.output / "benchmark"
            final = json.loads((out / "benchmark.json").read_text())
            self.assertEqual(final["settings"]["selected_by_class"], dict.fromkeys(classes, 1))
            self.assertEqual(final["settings"]["validation_subset_source"],
                             str(args.v3_run / "validation_subset.json"))
            self.assertEqual(len(final["scores"]), 9)
            with Image.open(out / "paired_examples.png") as image:
                image.verify()


if __name__ == "__main__":
    unittest.main()
