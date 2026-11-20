"""V9 ConvNeXt 免訓練推論篩選、Test 診斷與候選。"""

import argparse
import fcntl
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from distribution_audit.novelty import read_novelty
from v2.data_checks import read_json
from v3.inference import Prediction, SCALES
from v3.reports import read_reference, read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.geometry import seeded
from v7.impulse import MIN_TRIGGER_FRACTION
from v8.data import CONDITIONS, benchmark
from v8.experiment import init_cuda, read_checkpoint, version_info
from v8.models import make_model
from v9.reports import (CONFIRM_MIN_PP, MAX_CLEAN_DROP_PP, MAX_MAJOR_DROP_PP, SCREEN_MIN_PP,
                        frame_fraction, passes, primary_retention, test_diagnostics, validation_report)
from v9.views import Example, RECIPES, REFERENCE, describe_recipes, scores_for

PROTOCOL = "v9_convnext_tta_v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "screen", "diagnostic", "confirm", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v9"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--v8-submission", type=Path)
    p.add_argument("--audit-run", type=Path, default=Path("runs/distribution_audit"))
    p.add_argument("--batch-size", type=int, help="Defaults to the archived V8 batch size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--test-limit", type=int, default=768, help="Diagnostics rounds up to complete V8 batches")
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    out = args.output.resolve()
    root = Path("runs").resolve() / "v9"
    if not out.is_relative_to(root):
        p.error("--output must be inside runs/v9")
    if (args.batch_size is not None and args.batch_size < 1) or args.workers < 0 or args.test_limit < 1:
        p.error("Use positive batch size and Test sample size, and nonnegative workers")
    args.checkpoint = args.checkpoint or args.v8_run / "models/convnext_full/best.pt"
    protected = [args.data, args.v8_run, args.audit_run, args.checkpoint.parent]
    if args.v8_submission:
        protected.append(args.v8_submission.parent)
    if any(out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in protected):
        p.error("V9 output must be separate from existing data, checkpoints and reports")
    if root.is_symlink():
        p.error("runs/v9 must be an independent output directory")
    return args


@dataclass
class Sources:
    inputs: dict
    validation: list
    samples: list
    archived: dict
    archive_settings: dict
    provenance: dict


def sources(args):
    report, _, cases = benchmark(SimpleNamespace(output=args.v8_run, data=args.data))
    inputs = report["inputs"]
    template = args.data / "sample_submission.csv"
    samples = read_test_images(args.data / "test_images/test_images", template)
    ids = [item for _, item in samples]
    digest = file_sha256(args.checkpoint)
    if args.v8_submission is None:
        candidates = []
        for path in sorted((args.v8_run / "candidates").glob("*/submission_v8.csv")):
            settings_path = path.parent / "settings.json"
            if not settings_path.is_file():
                continue
            settings = read_json(settings_path)
            if (settings.get("checkpoint_sha256") == digest and settings.get("method") == "multiscale"
                    and settings.get("denoise") is True):
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError("Expected one archived V8 Multi-scale + Denoise CSV; pass --v8-submission")
        args.v8_submission = candidates[0]
    settings_path = args.v8_submission.parent / "settings.json"
    settings = read_json(settings_path)
    identity = {
        "checkpoint_sha256": digest,
        "template_sha256": file_sha256(template),
        "test_image_signature": image_signature(path for path, _ in samples),
        "submission_sha256": file_sha256(args.v8_submission),
    }
    if (any(settings.get(key) != value for key, value in identity.items())
            or settings.get("architecture") != "convnext_tiny" or settings.get("method") != "multiscale"
            or settings.get("denoise") is not True or settings.get("images") != len(ids)
            or type(settings.get("batch_size")) is not int or settings["batch_size"] < 1):
        raise ValueError("Use the original verified V8 ConvNeXt Multi-scale + Denoise checkpoint and CSV")
    archived = read_reference(args.v8_submission, ids, inputs["classes"])
    provenance = {**identity, "protocol": PROTOCOL, "split_sha256": inputs["split_sha256"],
                  "v8_settings_sha256": file_sha256(settings_path),
                  "benchmark_sha256": file_sha256(args.v8_run / "benchmark/prepare.json"),
                  "validation_manifest_sha256": report["manifest_sha256"]["validation"],
                  "seed": inputs["seed"], "source_origin": settings["origin"]}
    return Sources(inputs, cases, samples, archived, settings, provenance)


def audit_features(args, info):
    folder = args.audit_run
    summary = read_json(folder / "summary.json")
    if (summary["provenance"]["split_sha256"] != info.inputs["split_sha256"]
            or summary["labels"]["allowed_characters"] != info.inputs["classes"]
            or summary["labels"]["test_ground_truth_available"] is not False
            or summary["test"]["decoded_images"] != len(info.samples)):
        raise ValueError("Distribution audit and V8 data do not match")
    features = {}
    for row in read_novelty(folder / "test_novelty.csv"):
        item = row["id"]
        if item not in info.archived or item in features:
            raise ValueError(f"Unknown or repeated audit Test ID: {item}")
        if (min(int(row["width"]), int(row["height"])) < 1
                or not 0 <= float(row["black_edge_fraction"]) <= 1):
            raise ValueError(f"Invalid audit image dimensions or black-edge fraction: {item}")
        features[item] = row
    if features.keys() != info.archived.keys():
        raise ValueError("Distribution audit does not cover all Test IDs")
    return summary, features


def write_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    publish(path, payload)
    return file_sha256(path)


def audit(args, info):
    summary, features = audit_features(args, info)
    features_order = ("brightness", "contrast", "weak_color_fraction", "fine_detail")
    training_settings = args.checkpoint.parent / "settings.json"
    saved = read_json(training_settings) if training_settings.is_file() else {}
    config = saved.get("config", {})
    per_label = {}
    for case in info.validation:
        if case.fold == "screen":
            per_label.setdefault(case.label, case)
    examples = validation_examples(args, list(per_label.values()))
    result = {
        "provenance": info.provenance,
        "audit_summary_sha256": file_sha256(args.audit_run / "summary.json"),
        "audit_novelty_sha256": file_sha256(args.audit_run / "test_novelty.csv"),
        "v8_training": {key: config.get(key) for key in ("mode", "augmentation", "amp")},
        "train_and_test": {
            key: {**{column: summary[key][column] for column in
                     ("formats_by_readable_header", "sizes", "non_mostly_black_edge_pct")},
                  "feature_medians": {name: summary[key]["feature_medians"][name] for name in features_order}}
            for key in ("train", "test")
        },
        "normalization_both": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "decode_both": "Pillow RGB",
        "reference": {"resize_short_edges": list(SCALES), "crop": 224, "aggregation": "mean_softmax",
                      "force_square": False, "impulse_trigger_min_fraction": MIN_TRIGGER_FRACTION},
        "test_crop_frame_fraction_p10_p50_p90": {
            str(scale): np.percentile([frame_fraction(int(row["width"]), int(row["height"]), scale)
                                      for row in features.values()], [10, 50, 90]).tolist()
            for scale in (240, 256, 288)
        },
        "synthetic_primary_alpha_retention": primary_retention(examples, info.inputs["seed"]),
    }
    path = args.output / "preprocessing.json"
    write_report(path, result)
    print(f"{path} | Test 中位畫面保留率 "
          f"{ {key: round(value[1], 3) for key, value in result['test_crop_frame_fraction_p10_p50_p90'].items()} }",
          flush=True)


def gpu_model(args, info):
    checkpoint = read_checkpoint(args.checkpoint, info.inputs, args.allow_imagenet)
    if checkpoint["architecture"] != "convnext_tiny" or checkpoint["origin"] != info.archive_settings["origin"]:
        raise ValueError("V8 checkpoint provenance does not match the archived candidate")
    init_cuda(info.inputs["seed"])
    model = make_model("convnext_tiny", len(info.inputs["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cuda().eval(), version_info()


def validation_examples(args, cases):
    root = args.data / "train/train/train"
    return [Example(root / case.item, case.item, root / case.distractor) for case in cases]


def validation_run(args, info, model, env, plan, freeze=None):
    folder = args.output / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    cases = [case for case in info.validation if case.fold == args.stage]
    examples = validation_examples(args, cases)
    signature = image_signature(path for ex in examples for path in (ex.path, ex.distractor))
    metadata = {**info.provenance, "phase": args.stage, "environment": env, "image_signature": signature,
                "batch_size": args.batch_size, "workers": args.workers}
    scores, timings = {}, {}
    for condition in CONDITIONS:
        scores[condition], bank = scores_for(folder / f"{condition}.npz", examples, plan,
                                            {**metadata, "condition": condition}, len(info.inputs["classes"]),
                                            condition, info.inputs["seed"], model, args.batch_size, args.workers)
        timings[condition] = {"image_forwards": bank.forwards, "inference_seconds": bank.inference_seconds,
                              "wall_seconds": bank.wall_seconds}
        print(f"{args.stage} {condition} | {bank.forwards} views | {bank.wall_seconds:.1f}s", flush=True)
    result = validation_report(scores, np.array([case.label for case in cases]), plan, info.inputs["seed"],
                               confirm=args.stage == "confirm")
    path = folder / "summary.json"
    report = {**metadata, **result, "recipes": describe_recipes(plan), "timings": timings,
              "gate": {"hard_screen_min_pp": SCREEN_MIN_PP, "hard_confirm_min_pp": CONFIRM_MIN_PP,
                       "max_clean_drop_pp": MAX_CLEAN_DROP_PP, "max_major_drop_pp": MAX_MAJOR_DROP_PP,
                       "confirm_ci95_lower_must_exceed": 0}}
    if freeze:
        report["frozen_screen_sha256"] = freeze["screen_sha256"]
        report["tested_recipe"] = freeze["selected"]
    digest = write_report(path, report)
    if args.stage == "screen":
        write_report(args.output / "frozen.json", {"provenance": info.provenance,
                     "screen_sha256": digest, "selected": result["selected"],
                     "recipe": describe_recipes([result["selected"]]) if result["selected"] else None})
    chosen = result["selected"] or report.get("tested_recipe")
    print(json_bytes({"decision": result["decision"], "recipe": chosen,
                      "reference_hard": result["methods"][REFERENCE]["hard_accuracy"],
                      "paired_hard": result["methods"][chosen]["hard_effect"] if chosen else None,
                      "report": str(path)}).decode(), end="", flush=True)


def frozen(args, info):
    freeze = read_json(args.output / "frozen.json")
    selected = freeze["selected"]
    screen = args.output / "screen/summary.json"
    if (freeze["provenance"] != info.provenance or freeze["screen_sha256"] != file_sha256(screen)
            or selected != read_json(screen)["selected"]
            or selected is not None and (selected not in RECIPES or selected == REFERENCE
                                         or freeze["recipe"] != describe_recipes([selected]))):
        raise ValueError("Frozen recipe does not match the recorded Screen decision")
    return freeze


def sample_batches(samples, batch_size, limit, seed):
    # 保留歷史 V8 批次及其最後不足一批的大小。
    starts = range(0, len(samples), batch_size)
    count = min(len(starts), (limit + batch_size - 1) // batch_size)
    chosen = sorted(seeded(seed, "v9:test_batches").sample(starts, count))
    return [sample for start in chosen for sample in samples[start:start + batch_size]]


def diagnostic(args, info, model, env):
    _, features = audit_features(args, info)
    batch_size = info.archive_settings["batch_size"]
    examples = [Example(path, item) for path, item in sample_batches(
        info.samples, batch_size, args.test_limit, info.inputs["seed"]
    )]
    items = [ex.item for ex in examples]
    folder = args.output / "diagnostic"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {**info.provenance, "environment": env, "phase": "diagnostic", "workers": args.workers,
                "archive_batch_size": batch_size, "batch_size": args.batch_size}
    methods, bank = scores_for(folder / "views.npz", examples, list(RECIPES), metadata,
                               len(info.inputs["classes"]), "test", info.inputs["seed"], model,
                               args.batch_size, args.workers)
    dims, group_rows, mismatch = test_diagnostics(items, methods, bank.fractions, features, info.archived)
    classes = info.inputs["classes"]
    publish_csv(folder / "by_group.csv",
                ("method", "dimension", "group", "images", "mean_confidence", "mean_top2_margin",
                 "mean_view_disagreement", "change_vs_v8_replay", "change_vs_v8_csv"), group_rows)
    fields = ("id", "method", "archived_v8", "prediction", "confidence", "top2_margin", "view_disagreement",
              "visual_band", "combo_support", "outer_black", "impulse_trigger", "impulse_fraction")
    publish_csv(folder / "per_image.csv", fields, (
        (item, name, classes[info.archived[item]], classes[score.labels[i]], score.confidence[i], score.margin[i],
         score.disagreement[i], *dims[i].values(), bank.fractions[i])
        for name, score in methods.items() for i, item in enumerate(items)
    ))
    publish_csv(folder / "v8_replay_mismatches.csv", ("id", "archived", "replay", "confidence", "top2_margin"), (
        (items[i], classes[info.archived[items[i]]], classes[methods[REFERENCE].labels[i]],
         methods[REFERENCE].confidence[i], methods[REFERENCE].margin[i]) for i in mismatch
    ))
    result = {"provenance": info.provenance, "samples": len(items), "v8_replay_mismatches": len(mismatch),
              "replayed_complete_archive_batches": args.batch_size == batch_size,
              "group_table_sha256": file_sha256(folder / "by_group.csv"),
              "impulse_triggered": int((bank.fractions >= MIN_TRIGGER_FRACTION).sum()),
              "environment": env, "recipes": describe_recipes(RECIPES),
              "audit_novelty_sha256": file_sha256(args.audit_run / "test_novelty.csv"),
              "image_forwards": bank.forwards, "wall_seconds": bank.wall_seconds}
    write_report(folder / "summary.json", result)
    print(json_bytes({key: result[key] for key in ("samples", "impulse_triggered", "v8_replay_mismatches",
                                                    "wall_seconds")}).decode(), end="", flush=True)


def candidate_gate(args, info, freeze):
    path = args.output / "confirm/summary.json"
    diagnostic_path = args.output / "diagnostic/summary.json"
    confirm = read_json(path)
    review = read_json(diagnostic_path)
    selected = freeze["selected"]
    if (confirm["decision"] != "passes_confirmation" or confirm["selected"] != selected
            or confirm["frozen_screen_sha256"] != freeze["screen_sha256"]
            or any(confirm.get(key) != value for key, value in info.provenance.items())
            or confirm["recipes"].get(selected) != freeze["recipe"][selected]
            or not passes(confirm["methods"][selected], confirm=True)
            or review["group_table_sha256"] != file_sha256(diagnostic_path.parent / "by_group.csv")
            or review["provenance"] != info.provenance or review["v8_replay_mismatches"] != 0
            or not review["replayed_complete_archive_batches"]):
        raise ValueError("Candidate requires paired Confirm support and an exact archived V8 Test replay")
    return {"confirm_sha256": file_sha256(path), "diagnostic_sha256": file_sha256(diagnostic_path)}


def predict(args, info, model, env, freeze, gate):
    folder = args.output / "candidate"
    folder.mkdir(parents=True, exist_ok=True)
    name = freeze["selected"]
    examples = [Example(path, item) for path, item in info.samples]
    metadata = {**info.provenance, **gate, "environment": env, "phase": "predict",
                "batch_size": args.batch_size, "workers": args.workers}
    results, bank = scores_for(folder / "views.npz", examples, [name], metadata, len(info.inputs["classes"]),
                               "test", info.inputs["seed"], model, args.batch_size, args.workers)
    scores = results[name]
    if (file_sha256(args.checkpoint) != info.provenance["checkpoint_sha256"]
            or file_sha256(args.v8_submission) != info.provenance["submission_sha256"]
            or file_sha256(args.data / "sample_submission.csv") != info.provenance["template_sha256"]
            or image_signature(path for path, _ in info.samples) != info.provenance["test_image_signature"]):
        raise ValueError("V8 sources or Test changed during candidate inference")
    ids = [ex.item for ex in examples]
    predictions = {item: Prediction(int(scores.labels[i]), float(scores.confidence[i]),
                                    float(scores.disagreement[i]), 0.0) for i, item in enumerate(ids)}
    path = folder / "submission_v9.csv"
    changed = sum(predictions[item].label != info.archived[item] for item in ids)
    if changed:
        submission(path, ids, predictions, info.inputs["classes"])
    result = {"recipe": name, "recipes": freeze["recipe"], "images": len(ids),
              "changed_vs_v8": changed, "decision": "candidate_saved" if changed else "identical_to_v8",
              "submission_sha256": file_sha256(path) if changed else None, "seconds": bank.wall_seconds}
    write_report(folder / "summary.json", {**metadata, **result})
    print(json_bytes(result).decode(), end="", flush=True)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("V9 已有執行中的階段。")
        info = sources(args)
        args.batch_size = args.batch_size or info.archive_settings["batch_size"]
        if args.stage == "audit":
            audit(args, info)
            return
        freeze = frozen(args, info) if args.stage in ("confirm", "predict") else None
        if freeze and not freeze["selected"]:
            raise SystemExit("Screen 無達門檻配方，停止 V9 候選流程。")
        gate = candidate_gate(args, info, freeze) if args.stage == "predict" else None
        model, environment = gpu_model(args, info)
        if args.stage == "diagnostic":
            diagnostic(args, info, model, environment)
        elif args.stage == "predict":
            predict(args, info, model, environment, freeze, gate)
        else:
            plan = [REFERENCE, freeze["selected"]] if freeze else list(RECIPES)
            validation_run(args, info, model, environment, plan, freeze)


if __name__ == "__main__":
    main()
