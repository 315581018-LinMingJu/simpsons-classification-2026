"""用原 V5／V6 權重驗證一次低成本機率融合。"""

import argparse
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from classification_v2 import create_model, load_checkpoint, transforms_v2
from classification_v5 import EPOCHS, METHOD
from classification_v6 import ValidationImages
from compare_v1_v2 import SPLIT_SHA256, load_comparison_split
from ensemble.data import MODELS, audit_test, fresh_validation, probability_cache
from ensemble.evaluation import CONDITIONS, WEIGHTS, evaluate, fuse, guarded
from v2.data_checks import CLASS_COUNT, read_json
from v3.inference import Prediction, Views, infer_view_batches, view_loader
from v3.reports import read_reference, read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v5.challenge import ChallengeImages
from v6.composite import Targets

PROTOCOL = "v5_v6_ensemble_v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "validate", "test-probs", "predict"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--v6-run", type=Path, default=Path("runs/v6_composite/experiment"))
    parser.add_argument("--v5-submission", type=Path)
    parser.add_argument("--v6-submission", type=Path)
    parser.add_argument("--audit-run", type=Path, default=Path("runs/distribution_audit"))
    parser.add_argument("--output", type=Path, default=Path("runs/v6_ensemble"))
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--per-class", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.per_class < 4 or args.batch_size < 1 or args.workers < 0:
        parser.error("per-class >= 4, batch-size >= 1 and workers >= 0 are required")
    output = args.output.resolve()
    if (output.parent != Path("runs").resolve() or not output.name.startswith("v6_ensemble")
            or any(output.is_relative_to(path.resolve()) for path in
                   (args.data, args.v1_run, args.v2_run, args.v5_run, args.v6_run, args.audit_run))):
        parser.error("--output must be an independent runs/v6_ensemble... directory")
    args.v5_submission = args.v5_submission or args.v5_run / "submission_v5.csv"
    args.v6_submission = args.v6_submission or args.v6_run / "submission_v6.csv"
    return args


def model_settings(args):
    classes = read_json(args.v2_run / "classes.json")
    split_hash = file_sha256(args.v2_run / "split.csv")
    runs = {name: getattr(args, name + "_run") for name in MODELS}
    settings = {name: read_json(run / "settings.json") for name, run in runs.items()}
    if (len(classes) != CLASS_COUNT or len(set(classes)) != CLASS_COUNT or split_hash != SPLIT_SHA256
            or any(config["classes"] != classes or config["split_sha256"] != split_hash
                   or config["method"] != METHOD for config in settings.values())):
        raise ValueError("V5 and V6 must use the original split, class mapping and Multi-scale")
    return classes, split_hash, settings


def historical(args, classes):
    test = read_test_images(args.data / "test_images/test_images",
                            args.data / "sample_submission.csv")
    ids = [item for _, item in test]
    archived = {name: read_reference(getattr(args, name + "_submission"), ids, classes)
                for name in MODELS}
    return test, ids, archived


def inference_settings(args, classes, split_hash, sources):
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("Use a confirmed available CUDA GPU for inference")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, evaluation = transforms_v2()
    paths = {"v5": args.v5_run / f"epoch_{EPOCHS:02d}.pt", "v6": args.v6_run / "v6.pt"}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if (any(config["eval_transform"] != repr(evaluation) for config in sources.values())
            or read_json(args.v5_run / "validation.json")["checkpoint_sha256"] != hashes["v5"]
            or read_json(args.v6_run / "validation.json")["arms"]["v6"]["checkpoint_sha256"] != hashes["v6"]
            or sources["v6"]["v5_checkpoint_sha256"] != hashes["v5"]):
        raise ValueError("V5/V6 checkpoints or evaluation transform differ from their completed runs")
    return evaluation, paths, {
        "protocol": PROTOCOL, "classes": classes, "split_sha256": split_hash,
        "checkpoints": hashes, "method": METHOD, "eval_transform": repr(evaluation),
        "batch_size": args.batch_size, "workers": args.workers,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "pillow": pillow_version,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "cudnn_deterministic": True,
    }


def get_model(paths, settings, name):
    model = create_model().eval()
    load_checkpoint(model, paths[name], settings["classes"], settings["split_sha256"])
    return model


def validate(args, classes, split_hash, sources, evaluation, paths, settings):
    import torch

    train_root = args.data / "train/train/train"
    original_classes, rows, original_split = load_comparison_split(train_root, args.v1_run, args.v2_run)
    signature = image_signature(row.path for row in rows)
    if (classes != original_classes or split_hash != original_split
            or any(source["train_image_signature"] != signature for source in sources.values())):
        raise ValueError("Validation images no longer match V5/V6")
    print("準備去重且未用於 V5／V6 局部驗證的新樣本……", flush=True)
    samples, labels, folds, distractors, exclusions = fresh_validation(
        rows, train_root, args.v5_run, classes, args.per_class, args.seed,
    )
    ids = [item for _, item in samples]
    output = args.output / "validation"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "samples.csv"
    publish_csv(manifest, ("relative_path", "character", "fold", "distractor_path"),
                ((item, classes[label], fold, distractors[item].path.relative_to(train_root).as_posix())
                 for item, label, fold in zip(ids, labels, folds, strict=True)))
    composite_targets = Targets(**sources["v6"]["targets_from_unlabeled_test"])
    validation_settings = {
        "inference": settings, "conditions": CONDITIONS, "seed": args.seed,
        "v6_composite_seed": sources["v6"]["seed"],
        "targets": sources["v6"]["targets_from_unlabeled_test"],
        "train_image_signature": signature, "sample_manifest_sha256": file_sha256(manifest),
        "exclusions": exclusions,
    }
    publish(output / "settings.json", json_bytes(validation_settings))
    views = Views(METHOD, evaluation)
    arrays = {}
    for name in MODELS:
        model = None
        parts = []
        for condition in CONDITIONS:
            path = output / f"{name}_{condition}.npz"
            if not path.is_file():
                if model is None:
                    model = get_model(paths, settings, name)
                dataset = (ValidationImages(samples, condition, sources["v6"]["seed"], composite_targets, views)
                           if condition == "heldout_composite" else
                           ChallengeImages(samples, condition, views, args.seed, distractors))
                batches = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                    prefetch_factor=1 if args.workers else None,
                )
                parts.append(probability_cache(path, ids, classes, lambda: infer_view_batches(model, batches)))
            else:
                parts.append(probability_cache(path, ids, classes, None))
        arrays[name] = np.stack(parts)

    report = evaluate(arrays["v5"], arrays["v6"], labels, folds, args.seed)
    report["validation_settings_sha256"] = file_sha256(output / "settings.json")
    publish(args.output / "validation_report.json", json_bytes(report))
    print(f"{report['decision']} | selected V5 weight: {report['selected_v5_weight']}", flush=True)


def accepted_candidate(output, settings):
    report_path = output / "validation_report.json"
    report = read_json(report_path)
    if report["decision"] != "supports_ensemble":
        raise SystemExit(f"保留 V6。驗證判定 {report['decision']}，未產生集成 Submission。")
    manifest = output / "validation/settings.json"
    temperatures = report["applied_temperatures"]
    if (set(temperatures) != set(MODELS)
            or any(not 0.5 <= value <= 3 for value in temperatures.values())
            or not guarded(report["confirmation"]["ensemble_gain_vs_v6"], confirm=True)
            or report["validation_settings_sha256"] != file_sha256(manifest)
            or read_json(manifest)["inference"] != settings
            or report["selected_v5_weight"] not in WEIGHTS):
        raise ValueError("Ensemble validation does not match current inference settings")
    return report


def run_test_inference(args, classes, evaluation, paths, settings, test, ids, archived, report):
    output = args.output / "test"
    output.mkdir(parents=True, exist_ok=True)
    publish(output / "settings.json", json_bytes({
        "inference": settings, "test_image_signature": image_signature(path for path, _ in test),
        "reference_sha256": {name: file_sha256(getattr(args, name + "_submission")) for name in MODELS},
    }))
    matrices, mismatches = {}, {}
    for name in MODELS:
        path = output / f"{name}.npz"
        if path.is_file():
            matrix = probability_cache(path, ids, classes, None)
        else:
            model = get_model(paths, settings, name)
            batches = view_loader(test, METHOD, evaluation, args.batch_size, args.workers)
            matrix = probability_cache(path, ids, classes, lambda: infer_view_batches(model, batches))
            del model
        matrices[name] = matrix
        mismatches[name] = [item for item, label in zip(ids, matrix.argmax(axis=1), strict=True)
                            if label != archived[name][item]]
    publish(output / "replay_check.json", json_bytes({
        "test_images": len(ids), "count_by_model": {key: len(items) for key, items in mismatches.items()},
        "mismatch_ids": mismatches,
    }))
    if report is None:
        return

    weight = report["selected_v5_weight"]
    probabilities = fuse(matrices["v5"], matrices["v6"], weight, report["applied_temperatures"])
    predictions = {item: Prediction(int(label), float(row[label]), 0, 0)
                   for item, row, label in zip(ids, probabilities, probabilities.argmax(axis=1), strict=True)}
    predicted_path = args.output / "submission_v6_ensemble.csv"
    submission(predicted_path, ids, predictions, classes)
    publish(args.output / "candidate.json", json_bytes({
        "weights": {"v5": weight, "v6": 1 - weight}, "temperatures": report["applied_temperatures"],
        "validation_sha256": file_sha256(args.output / "validation_report.json"),
        "probability_sha256": {name: file_sha256(output / f"{name}.npz") for name in MODELS},
        "submission_sha256": file_sha256(predicted_path),
        "changed_vs_archived_v6": sum(predictions[item].label != archived["v6"][item] for item in ids),
    }))
    print(predicted_path, flush=True)


def main():
    args = arguments()
    classes, split_hash, sources = model_settings(args)

    if args.stage == "audit":
        _, ids, archived = historical(args, classes)
        args.output.mkdir(parents=True, exist_ok=True)
        audit_test(args.output, args, ids, classes, archived, split_hash)
        print(args.output / "v5_v6_test_audit.json", flush=True)
        return

    evaluation, paths, settings = inference_settings(args, classes, split_hash, sources)
    if args.stage == "validate":
        validate(args, classes, split_hash, sources, evaluation, paths, settings)
        return

    report = accepted_candidate(args.output, settings) if args.stage == "predict" else None
    test, ids, archived = historical(args, classes)
    run_test_inference(args, classes, evaluation, paths, settings, test, ids, archived, report)


if __name__ == "__main__":
    main()
