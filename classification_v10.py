"""V10 以固定 V8 ConvNeXt 權重短微調，通過配對驗證才準備候選。"""

import argparse
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from classification_v2 import transforms_v2
from classification_v8 import TestImages
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import Views, infer_loader
from v3.reports import read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v6.composite import Targets
from v8.data import CONDITIONS, benchmark, selected_signature
from v8.experiment import data_loader, init_cuda, labels_for, read_checkpoint
from v10.experiment import (DENOISE, FORMAT, METHOD, MIN_AUDIT_ACCEPTANCE, TRAIN_PER_CLASS, code_fingerprint, comparison,
                            evaluate_arm, fit, gate, gpu_environment, make_loaded_model, preview,
                            sampled_audit, subset)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "train", "screen", "confirm", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v10"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--distribution-audit", type=Path, default=Path("runs/distribution_audit/summary.json"))
    p.add_argument("--batch-size", type=int, default=16, choices=(4, 8, 16))
    p.add_argument("--eval-batch-size", type=int, default=16, choices=(4, 8, 16))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    root = Path("runs").resolve() / "v10"
    out = args.output.resolve()
    protected = (args.data, args.v8_run, args.distribution_audit.parent)
    if (root.is_symlink() or not out.is_relative_to(root)
            or any(out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out)
                   for path in protected)):
        p.error("Use an independent output under runs/v10")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    return args


@dataclass
class Experiment:
    path: Path
    parent: dict
    classes: list
    training: list
    validation: list
    summary: dict
    targets: Targets
    provenance: dict


def sources(args):
    report, train_all, validation = benchmark(SimpleNamespace(output=args.v8_run, data=args.data))
    inputs = report["inputs"]
    root = args.data / "train/train/train"
    path = args.v8_run / "models/convnext_all_layers/best.pt"
    digest = file_sha256(path)
    parent = read_checkpoint(path, inputs, args.allow_imagenet)
    config = parent["config"]
    if (parent["architecture"] != "convnext_tiny" or config["mode"] != "full"
            or config["benchmark_sha256"] != report["manifest_sha256"]
            or config["train_image_signature"] != inputs["train_image_signature"]
            or config["augmentation"] not in {"basic", "context", "robust"}
            or file_sha256(path) != digest):
        raise ValueError("Use the original convnext_all_layers/best.pt and its V8 full benchmark")
    summary = read_json(args.distribution_audit)
    targets = Targets.from_summary(summary)
    if (summary["provenance"]["split_sha256"] != inputs["split_sha256"]
            or summary["labels"]["allowed_characters"] != inputs["classes"]
            or summary["labels"]["test_ground_truth_available"] is not False
            or summary["test"]["decoded_images"] != TEST_COUNT
            or asdict(targets) != config["targets"]):
        raise ValueError("Use the original distribution audit for this split")
    seed = inputs["seed"]
    training = subset(train_all, TRAIN_PER_CLASS, seed + 1, "train")
    provenance = {
        "protocol": FORMAT, "v8_checkpoint": path.as_posix(), "v8_checkpoint_sha256": digest,
        "v8_checkpoint_origin": parent["origin"],
        "benchmark_sha256": file_sha256(args.v8_run / "benchmark/prepare.json"),
        "validation_manifest_sha256": report["manifest_sha256"]["validation"],
        "split_sha256": inputs["split_sha256"], "classes": inputs["classes"], "training_seed": seed + 1,
        "validation_seed": seed, "v10_training_signature": selected_signature(root, training),
        "distribution_audit_sha256": file_sha256(args.distribution_audit),
        "code_files_sha256": code_fingerprint(Path(__file__).resolve().parent),
    }
    return Experiment(path, parent, inputs["classes"], training, validation, summary, targets, provenance)


def audit(args, info):
    folder = args.output / "audit"
    folder.mkdir(parents=True, exist_ok=True)
    root = args.data / "train/train/train"
    report, pairs = sampled_audit(root, info.training, info.targets, info.provenance["training_seed"])
    manifest = folder / "train.csv"
    publish_csv(manifest, ("image", "label"), ((case.item, case.label) for case in info.training))
    preview(folder / "preview.png", pairs)
    keys = ("brightness", "contrast", "fine_detail")
    result = {**report, "provenance": info.provenance, "training_images": len(info.training),
              "training_manifest_sha256": file_sha256(manifest),
              "audit_test_reference_medians": {key: info.summary["test"]["feature_medians"][key] for key in keys}}
    publish(folder / "summary.json", json_bytes(result))
    print(json_bytes({key: result[key] for key in ("decision", "training_images", "train_images_audited",
                                                   "acceptance", "geometry_p10_p50_p90",
                                                   "accepted_scene_medians")}).decode(), end="", flush=True)


def checked_audit(args, info):
    folder = args.output / "audit"
    path = folder / "summary.json"
    if not path.is_file():
        raise SystemExit("先執行 V10 audit，確認增強樣本。")
    report = read_json(path)
    if (report["provenance"] != info.provenance
            or report["training_manifest_sha256"] != file_sha256(folder / "train.csv")):
        raise ValueError("Audit inputs or training manifest changed")
    if report["decision"] != "ready" or report["acceptance"] < MIN_AUDIT_ACCEPTANCE:
        raise SystemExit("Audit 可用場景不足，停止 GPU 實驗。")
    return file_sha256(path)


def trained_path(args):
    return args.output / "experiment/best.pt"


def load_adapted(path, info, audit_hash):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format") != FORMAT or checkpoint.get("architecture") != "convnext_tiny"
            or checkpoint.get("provenance") != info.provenance or checkpoint["classes"] != info.classes
            or checkpoint.get("audit_sha256") != audit_hash):
        raise ValueError("V10 checkpoint differs from the audited experiment")
    return checkpoint


def verified_report(args, phase, info, adapted_hash, audit_hash, screen_hash=None):
    path = args.output / phase / "comparison.json"
    if not path.is_file():
        raise SystemExit(f"先完成 V10 {phase} 比較。")
    report = read_json(path)
    if (report["provenance"] != info.provenance or report["v10_checkpoint_sha256"] != adapted_hash
            or report["audit_sha256"] != audit_hash or report.get("screen_sha256") != screen_hash
            or report["phase"] != phase or report["method"] != METHOD or report["denoise"] is not DENOISE):
        raise ValueError(f"Incompatible V10 {phase} report")
    expected_decision = "advance_to_confirm" if phase == "screen" else "eligible_for_candidate"
    if report["decision"] != expected_decision or gate(phase, report["hard_effect"], report["condition_gain_pp"]):
        raise SystemExit(f"{phase} 未達停損門檻，保留 V8。")
    return file_sha256(path)


def validate(args, info, adapted, adapted_hash, audit_hash, screen_hash=None):
    seed = info.provenance["validation_seed"]
    init_cuda(seed)
    cases = [case for case in info.validation if case.fold == args.stage]
    root = args.data / "train/train/train"
    signature = selected_signature(root, cases)
    destination = args.output / args.stage
    metadata = {
        "phase": args.stage, "provenance": info.provenance, "v10_checkpoint_sha256": adapted_hash,
        "audit_sha256": audit_hash, "screen_sha256": screen_hash, "validation_image_signature": signature,
        "conditions": list(CONDITIONS), "method": METHOD, "denoise": DENOISE, "precision": "fp32",
        "batch_size": args.eval_batch_size, "workers": args.workers, "environment": gpu_environment(),
    }
    arms = {}
    for name, checkpoint in (("v8", info.parent), ("v10", adapted)):
        print(f"{args.stage} | {name} | {len(cases)} images per condition", flush=True)
        arms[name] = evaluate_arm(destination / name, checkpoint, root, cases, info.classes,
                                  seed, args, {**metadata, "arm": name})
    if (selected_signature(root, cases) != signature
            or file_sha256(info.path) != info.provenance["v8_checkpoint_sha256"]
            or file_sha256(trained_path(args)) != adapted_hash):
        raise ValueError("Validation images or checkpoints changed during the comparison")
    result = comparison(args.stage, labels_for(cases), arms["v8"][0], arms["v10"][0], seed)
    report = {**metadata, **result,
              "evaluation_sha256": {name: file_sha256(destination / name / "summary.json") for name in arms}}
    publish(destination / "comparison.json", json_bytes(report))
    publish_csv(destination / "comparison.csv",
                ("condition", "v8_accuracy_pct", "v10_accuracy_pct", "gain_pp", "ci95_lower_pp",
                 "ci95_upper_pp", "wins", "losses", "prediction_changes"), (
        (name, values["v8_accuracy_pct"], values["v10_accuracy_pct"], values["effect"]["pp"],
         *values["effect"]["ci95_pp"], values["wins"], values["losses"], values["prediction_changes"])
        for name, values in result["conditions"].items()
    ))
    print(json_bytes({key: result[key] for key in ("decision", "reasons", "hard_effect", "condition_gain_pp")})
          .decode(), end="", flush=True)


def predict(args, info, adapted, adapted_hash, confirm_hash):
    template = args.data / "sample_submission.csv"
    samples = read_test_images(args.data / "test_images/test_images", template)
    test_hash = image_signature(path for path, _ in samples)
    template_hash = file_sha256(template)
    init_cuda(info.provenance["validation_seed"])
    model = make_loaded_model(adapted, info.classes)
    dataset = TestImages(samples, Views(METHOD, transforms_v2()[1]), DENOISE)
    batches = data_loader(dataset, args.eval_batch_size, args.workers)
    predictions = {}
    for batch in infer_loader(model, batches):
        predictions.update(batch)
    ids = [item for _, item in samples]
    if (list(predictions) != ids or image_signature(path for path, _ in samples) != test_hash
            or file_sha256(template) != template_hash or file_sha256(trained_path(args)) != adapted_hash):
        raise ValueError("Test order, Test images or V10 checkpoint changed during candidate inference")
    folder = args.output / "candidate"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "submission_v10.csv"
    submission(path, ids, predictions, info.classes)
    publish(folder / "settings.json", json_bytes({
        "provenance": info.provenance, "v10_checkpoint_sha256": adapted_hash, "confirm_sha256": confirm_hash,
        "template_sha256": template_hash, "test_image_signature": test_hash, "submission_sha256": file_sha256(path),
        "method": METHOD, "denoise": DENOISE, "batch_size": args.eval_batch_size, "workers": args.workers,
        "environment": gpu_environment(), "images": len(ids),
    }))
    print(f"{path} | {len(ids)} IDs", flush=True)


def run(args):
    info = sources(args)
    if args.stage == "audit":
        audit(args, info)
        return
    audit_hash = checked_audit(args, info)
    path = trained_path(args)
    if args.stage == "train":
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"V10 | {len(info.training)} images | 1 epoch | ConvNeXt final stage + head", flush=True)
        training = (load_adapted(path, info, audit_hash)["training"] if path.exists()
                    else fit(args, info, info.training, info.targets, path, audit_hash))
        publish(path.parent / "training.json", json_bytes(training))
        print(json_bytes(training).decode(), end="", flush=True)
        return
    if not path.is_file():
        raise SystemExit("先完成 V10 train。")
    adapted_hash = file_sha256(path)
    screen_hash = (verified_report(args, "screen", info, adapted_hash, audit_hash)
                   if args.stage in ("confirm", "predict") else None)
    adapted = load_adapted(path, info, audit_hash)
    if args.stage == "predict":
        confirm_hash = verified_report(args, "confirm", info, adapted_hash, audit_hash, screen_hash)
        predict(args, info, adapted, adapted_hash, confirm_hash)
    else:
        validate(args, info, adapted, adapted_hash, audit_hash, screen_hash)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("這個 V10 輸出目錄已有執行中的階段。")
        run(args)


if __name__ == "__main__":
    main()
