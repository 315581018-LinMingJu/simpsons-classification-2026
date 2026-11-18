"""使用既有 ResNet18 權重比較多視角推論。"""

import argparse
import platform
import random
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from classification_v2 import create_model, load_checkpoint, transforms_v2
from compare_v1_v2 import SPLIT_SHA256, load_comparison_split
from v2.data_checks import CLASS_COUNT, duplicate_diagnostic, read_json
from v3.inference import SCALES, VIEWS, infer_batches
from v3.reports import read_reference, read_test_images, submission, test_report, validation_report
from v3.storage import Cache, file_sha256, image_signature, json_bytes, publish


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("diagnostic", "validation", "predict"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v2-submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/v3_tta"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=128, help="diagnostic 樣本數")
    parser.add_argument("--seed", type=int, default=2026, help="diagnostic 抽樣種子")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 1 or args.workers < 0:
        parser.error("batch size and diagnostic limit must be positive; workers must be nonnegative")
    args.v2_submission = args.v2_submission or args.v2_run / "submission_v2.csv"
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in (
        args.data, args.v1_run, args.v2_run, Path("runs/v2_robustness"),
    )):
        parser.error("--output must be outside data and previous run directories")
    return args


def predict_cached(cache, samples, methods, make_model, inference):
    results = {}
    model = None
    for method in methods:
        predictions = cache.read(method)
        missing = [(path, item) for path, item in samples if item not in predictions]
        print(f"{method}: 沿用 {len(samples) - len(missing)} 張，待推論 {len(missing)} 張", flush=True)
        if missing:
            if model is None:
                model = make_model()
            view_method = "baseline" if method == "v1_baseline" else method
            for batch in inference(model, missing, view_method):
                cache.add(method, batch)
                predictions.update(batch)
        results[method] = {item: predictions[item] for _, item in samples}
    return results


def validation_subset(output, rows, signature, train_root):
    manifest = output / "validation_subset.json"
    valid = {row.path.relative_to(train_root).as_posix(): row for row in rows if row.part == "val"}
    if manifest.exists():
        selection = read_json(manifest)
        if selection["image_signature"] != signature:
            raise ValueError("Validation images changed; choose a new --output directory")
    else:
        print("逐張解碼訓練與驗證圖片，檢查跨集合像素完全重複……", flush=True)
        report, _, exposed = duplicate_diagnostic(rows)
        selection = {
            "image_signature": signature,
            "excluded_ids": [item for item, row in valid.items() if str(row.path) in exposed],
            "duplicate_counts": {key: value for key, value in report.items() if key != "groups"},
        }
        publish(manifest, json_bytes(selection))
    excluded = set(selection["excluded_ids"])
    if not excluded <= valid.keys():
        raise ValueError("Cached exclusions contain unknown validation IDs")
    samples = [(row.path, item) for item, row in valid.items() if item not in excluded]
    if not samples:
        raise ValueError("No validation images remain after duplicate filtering")
    return samples, {item: valid[item].label for _, item in samples}, selection["duplicate_counts"]


def main():
    args = arguments()

    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V3 inference")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, evaluation = transforms_v2()
    saved_eval = read_json(args.v2_run / "run_config.json")["eval_transform"]
    if saved_eval != repr(evaluation):
        raise ValueError("V2 saved evaluation transform differs from the current runtime")

    validating = args.stage == "validation"
    v2_checkpoint = args.v2_run / "best.pt"
    v1_checkpoint = args.v1_run / "best.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    if validating:
        train_root = args.data / "train/train/train"
        classes, rows, split_hash = load_comparison_split(train_root, args.v1_run, args.v2_run)
        signature = image_signature(row.path for row in rows)
        samples, targets, duplicates = validation_subset(args.output, rows, signature, train_root)
    else:
        classes = read_json(args.v2_run / "classes.json")
        if (not isinstance(classes, list) or len(classes) != CLASS_COUNT
                or not all(isinstance(name, str) and name for name in classes)
                or len(set(classes)) != CLASS_COUNT):
            raise ValueError("Expected 50 distinct ordered class names")
        if read_json(args.v1_run / "classes.json") != classes:
            raise ValueError("V1 and V2 class mappings differ")
        split_hash = file_sha256(args.v2_run / "split.csv")
        if split_hash != SPLIT_SHA256 or file_sha256(args.v1_run / "split.csv") != split_hash:
            raise ValueError("V1/V2 splits differ from the expected original split")
        samples = read_test_images(args.data / "test_images/test_images",
                                   args.data / "sample_submission.csv")
        signature = image_signature(path for path, _ in samples)
        reference = read_reference(args.v2_submission, [item for _, item in samples], classes)
        if args.stage == "diagnostic":
            indices = sorted(random.Random(args.seed).sample(
                range(len(samples)), min(args.limit, len(samples))))
            samples = [samples[index] for index in indices]

    metadata = {
        "cache_version": 1, "image_signature": signature, "split_sha256": split_hash,
        "classes": classes, "v2_checkpoint_sha256": file_sha256(v2_checkpoint),
        "eval_transform": repr(evaluation), "multiscale_short_edges": SCALES, "views": VIEWS,
        "batch_size": args.batch_size, "precision": "float32",
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": True, "gpu": torch.cuda.get_device_name(), "pillow": pillow_version,
        "inference_timing": "GPU transfer, forward, softmax and D2H; excludes loader and warm-up",
    }
    if not validating:
        metadata["v1_checkpoint_sha256"] = file_sha256(v1_checkpoint)

    def model_for(checkpoint, expected_hash=None):
        model = create_model().eval()
        load_checkpoint(model, checkpoint, classes, expected_hash)
        return model

    def inference(model, missing, method):
        return infer_batches(model, missing, method, evaluation, args.batch_size, args.workers)

    cache_path = args.output / ("validation.sqlite3" if validating else "test.sqlite3")
    with closing(sqlite3.connect(cache_path)) as connection:
        cache = Cache(connection, metadata)
        predictions = predict_cached(cache, samples, VIEWS,
                                     lambda: model_for(v2_checkpoint, split_hash), inference)
        if not validating:
            predictions.update(predict_cached(cache, samples, ("v1_baseline",),
                                              lambda: model_for(v1_checkpoint), inference))

    if validating:
        summary = validation_report(args.output, targets, predictions,
                                    {**metadata, "duplicate_counts": duplicates})
        for row in summary:
            print(f"{row['method']}: {row['accuracy_pct']:.4f}% "
                  f"({row['delta_vs_baseline_pp']:+.4f} pp)", flush=True)
        return

    diagnostic = args.stage == "diagnostic"
    output = args.output / "diagnostic" if diagnostic else args.output
    report_settings = {**metadata, "scope": args.stage,
                       "v2_reference_sha256": file_sha256(args.v2_submission)}
    if diagnostic:
        report_settings.update({"seed": args.seed, "requested_sample_count": args.limit})
    summary = test_report(output, samples, predictions, classes, reference, report_settings)
    mismatches = summary["baseline_reference_mismatch_count"]
    print(f"V3 Baseline 對照 V2 Submission：{mismatches}/{len(samples)} 筆不同", flush=True)
    if diagnostic:
        return
    if mismatches:
        raise RuntimeError("Baseline differs from V2 submission; see test_summary.json. "
                           "Candidate submissions were not generated.")
    ids = [item for _, item in samples]
    for method in VIEWS:
        path = output / f"submission_v3_{method}.csv"
        submission(path, ids, predictions[method], classes)
        print(f"{path} ({len(ids)} 筆)", flush=True)


if __name__ == "__main__":
    main()
