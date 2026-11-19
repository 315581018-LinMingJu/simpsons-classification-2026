"""凍結 V8，在固定有標籤 Screen 上做單因素比較。"""

import argparse
import csv
import fcntl
import time
from collections import Counter
from pathlib import Path

import numpy as np

from classification_v2 import transforms_v2
from classification_v11 import sources as v8_sources
from classification_v12 import reference_checkpoint
from v2.data_checks import read_json
from v3.inference import Views, infer_view_batches
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v8.data import open_rgb, selected_signature
from v8.experiment import data_loader, init_cuda
from v8.models import make_model
from v10.experiment import subset
from v12.model import environment, permitted
from v8_controlled.conditions import ARMS, VIEW_NAMES, blind_views, controlled_frames, kept_views
from v8_controlled.report import CONTRASTS, effects, geometry_summary

PROTOCOL = "v8_factors_validation_v1"
PER_CLASS = 4
MIN_PER_CLASS = 3


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--output", type=Path, default=Path("runs/v8_controlled"))
    parser.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    parser.add_argument("--v9-run", type=Path, default=Path("runs/v9"))
    parser.add_argument("--v10-run", type=Path, default=Path("runs/v10"))
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 16, 32))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-imagenet", action="store_true")
    args = parser.parse_args(argv)
    out, root = args.output.resolve(), Path("runs").resolve() / "v8_controlled"
    protected = (args.data, args.v8_run, args.v9_run, args.v10_run)
    if (root.is_symlink() or not out.is_relative_to(root) or any(
        out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in protected
    )):
        parser.error("--output must be independent under runs/v8_controlled")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


def print_json(payload):
    print(json_bytes(payload).decode(), end="", flush=True)


def earlier_screens(args, info):
    result = {}
    reference = info.identity["checkpoint_sha256"]
    for version, path in (("v9", args.v9_run / "screen/summary.json"),
                          ("v10", args.v10_run / "screen/comparison.json")):
        if not path.is_file():
            result[version] = {"status": "unavailable", "path": str(path)}
            continue
        try:
            report = read_json(path)
            source = report if version == "v9" else report["provenance"]
            digest = source.get("checkpoint_sha256") if version == "v9" else source.get("v8_checkpoint_sha256")
            same = digest == reference and source.get("benchmark_sha256") == info.identity["benchmark_sha256"]
            result[version] = {"status": "matched_reference" if same else "different_reference",
                               "decision": report.get("decision"), "path": str(path), "sha256": file_sha256(path)}
            if same and version == "v10":
                result[version]["hard_effect"] = report.get("hard_effect")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result[version] = {"status": "unreadable", "reason": str(exc), "path": str(path)}
    return result


def sources(args):
    info = v8_sources(args)
    info.cases = subset([case for case in info.validation if case.fold == "screen"], PER_CLASS,
                        info.inputs["seed"], "v8controlled:screen")
    if Counter(case.label for case in info.cases) != Counter({i: PER_CLASS for i in range(len(info.inputs["classes"]))}):
        raise ValueError("V8 Screen must allow the fixed per-class sample")
    root = Path(__file__).resolve().parent
    code = ("compare_v8_factors.py", "v8_controlled/conditions.py", "v8_controlled/report.py",
            "v12/model.py", "classification_v12.py")
    info.identity = {**info.identity, "protocol": PROTOCOL,
                     "screen_sample_signature": selected_signature(info.root, info.cases),
                     "new_code_sha256": {name: file_sha256(root / name) for name in code}}
    args.batch_size = args.batch_size or info.settings["batch_size"]
    return info


def prepare(args, info):
    record = args.output / "prepare.json"
    if record.is_file():
        result = read_prepared(args, info)
    else:
        accepted, excluded = [], []
        seed = info.inputs["seed"]
        for case in info.cases:
            example = controlled_frames(open_rgb(info.root / case.item), seed, case.item)
            if example.valid:
                accepted.append(case)
            else:
                excluded.append({"item": case.item, "reason": example.invalid_reason})
        per_class = Counter(case.label for case in accepted)
        classes = info.inputs["classes"]
        minimum_ok = all(per_class[i] >= MIN_PER_CLASS for i in range(len(classes)))
        path = args.output / "cohort.csv"
        publish_csv(path, ("item", "label"), ((case.item, case.label) for case in accepted))
        result = {"identity": info.identity, "cohort_sha256": file_sha256(path), "planned": len(info.cases),
                  "eligible_ids": [case.item for case in accepted], "eligible": len(accepted),
                  "excluded": excluded, "counts_by_class": {name: per_class[i] for i, name in enumerate(classes)},
                  "earlier_screens": earlier_screens(args, info),
                  "decision": "ready" if minimum_ok else "stop_insufficient_protected_samples",
                  "planned_image_forwards": len(accepted) * len(ARMS) * len(VIEW_NAMES)}
        publish(record, json_bytes(result))
    print_json({key: result[key] for key in ("decision", "planned", "eligible", "excluded",
                                            "earlier_screens", "planned_image_forwards")})


def read_prepared(args, info):
    path = args.output / "prepare.json"
    if not path.is_file():
        raise SystemExit("先執行 CPU prepare。")
    result = read_json(path)
    cohort = args.output / "cohort.csv"
    if result["identity"] != info.identity or file_sha256(cohort) != result["cohort_sha256"]:
        raise ValueError("V8 comparison input differs from the fixed cohort")
    with cohort.open(encoding="utf-8", newline="") as file:
        rows = [(row["item"], int(row["label"])) for row in csv.DictReader(file)]
    known = {case.item: case.label for case in info.cases}
    if ([item for item, _ in rows] != result["eligible_ids"]
            or any(known.get(item) != label for item, label in rows)):
        raise ValueError("V8 validation labels or IDs differ from prepare")
    return result


class ArmImages:
    def __init__(self, examples, arm, views):
        self.examples, self.arm, self.views = examples, arm, views

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        case, data = self.examples[index]
        return self.views(data.frames[self.arm]), case.item


def evaluate_bank(args, model, examples, arm, views):
    loader = data_loader(ArmImages(examples, arm, views), args.batch_size, args.workers)
    started = time.perf_counter()
    batches = list(infer_view_batches(model, loader))
    ids = [item for items, _, _, _ in batches for item in items]
    if ids != [case.item for case, _ in examples]:
        raise ValueError("V8 inference changed the prepared validation order")
    probabilities = np.concatenate([batch[1] for batch in batches], axis=1)
    classes = model.classifier[-1].out_features
    if (probabilities.shape != (len(VIEW_NAMES), len(examples), classes)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-5, rtol=0)):
        raise ValueError("Unexpected or non-finite V8 view probabilities")
    cost = {"image_forwards": len(VIEW_NAMES) * len(examples), "warmup_image_forwards": 1,
            "inference_seconds": sum(batch[3] for batch in batches), "wall_seconds": time.perf_counter() - started}
    return probabilities, cost


def save_validation(args, info, prepared):
    import torch

    started = time.perf_counter()
    permitted(args.allow_imagenet)
    checkpoint = reference_checkpoint(args, info)
    cohort = {case.item: case for case in info.cases}
    ids = prepared["eligible_ids"]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= cohort.keys():
        raise ValueError("Prepared cohort contains invalid validation IDs")
    seed = info.inputs["seed"]
    rendering = time.perf_counter()
    examples = [(cohort[item], controlled_frames(open_rgb(info.root / item), seed, item)) for item in ids]
    render_seconds = time.perf_counter() - rendering
    if any(not example.valid for _, example in examples):
        raise ValueError("An image failed the frozen condition invariants on replay")
    init_cuda(seed)
    model = make_model("convnext_tiny", len(info.inputs["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    views = Views("multiscale", transforms_v2()[1])
    predictions, costs = {}, {}
    retention = np.stack([example.retention for _, example in examples])
    selected = np.stack([kept_views(row) for row in retention])
    matched = np.stack([blind_views(chosen, seed, case.item) for (case, _), chosen in zip(examples, selected, strict=True)])
    for arm in ARMS:
        probabilities, costs[arm] = evaluate_bank(args, model, examples, arm, views)
        predictions[arm] = probabilities.mean(axis=0).argmax(axis=1)
        if arm == "offcenter":
            # 同一份前向結果，依合成原圖塊的保留程度選視角。
            for name, masks in (("retained", selected), ("matched_count", matched)):
                predictions[name] = np.asarray([
                    probabilities[masks[i], i].mean(axis=0).argmax() for i in range(len(examples))
                ])
        print(f"{arm} | {costs[arm]['image_forwards']} image views | {costs[arm]['wall_seconds']:.1f}s", flush=True)
    if (file_sha256(info.checkpoint) != info.identity["checkpoint_sha256"]
            or selected_signature(info.root, info.cases) != info.identity["screen_sample_signature"]):
        raise ValueError("V8 checkpoint or validation images changed mid-run")
    labels = np.asarray([case.label for case, _ in examples])
    geometry = geometry_summary(retention, selected, matched, [example.cue_retention for _, example in examples])
    comparisons = effects(labels, predictions, seed, geometry)
    classes = info.inputs["classes"]
    methods = (*ARMS, "retained", "matched_count")
    publish_csv(args.output / "predictions.csv",
                ("item", "label", *methods, "full_views_patch_retained_pct",
                 "selected_views_patch_retained_pct", "selected_views", "matched_count_views"), (
        (case.item, classes[case.label], *(classes[predictions[method][i]] for method in methods),
         100 * float(retention[i].mean()), 100 * float(retention[i, selected[i]].mean()),
         "|".join(name for name, keep in zip(VIEW_NAMES, selected[i], strict=True) if keep),
         "|".join(name for name, keep in zip(VIEW_NAMES, matched[i], strict=True) if keep))
        for i, (case, _) in enumerate(examples)
    ))
    publish_csv(args.output / "comparison.csv",
                ("contrast", "images", "reference_accuracy_pct", "treatment_accuracy_pct", "accuracy_delta_pp",
                 "accuracy_ci95_lower_pp", "accuracy_ci95_upper_pp", "flip_pct", "flip_ci95_lower_pp",
                 "flip_ci95_upper_pp", "wins", "losses", "mcnemar_holm_p", "decision"), (
        (name, row["images"], row["reference_accuracy_pct"], row["treatment_accuracy_pct"],
         row["accuracy_effect"]["pp"], *row["accuracy_effect"]["ci95_pp"], row["flip_rate"]["pp"],
         *row["flip_rate"]["ci95_pp"], row["wins"], row["losses"], row["mcnemar_holm_p"], row["decision"])
        for name, row in comparisons.items()
    ))
    return {
        "comparisons": comparisons, "geometry": geometry, "cost_by_arm": costs,
        "total_image_forwards": sum(item["image_forwards"] for item in costs.values()),
        "render_seconds": render_seconds, "total_wall_seconds": time.perf_counter() - started,
        "marginal_image_forwards_vs_reference": {
            name: costs[treatment]["image_forwards"] if treatment != "retained" else 0
            for name, (_, treatment) in CONTRASTS.items()
        },
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "predictions_sha256": file_sha256(args.output / "predictions.csv"),
        "comparison_table_sha256": file_sha256(args.output / "comparison.csv"),
    }


def validate(args, info):
    prepared = read_prepared(args, info)
    if prepared["decision"] != "ready":
        raise SystemExit("CPU 受保護樣本不足，依預先門檻停止。")
    path = args.output / "summary.json"
    metadata = {"identity": info.identity, "prepare_sha256": file_sha256(args.output / "prepare.json"),
                "batch_size": args.batch_size, "workers": args.workers, "scope": "labeled_v8_screen_only"}
    if path.is_file():
        report = read_json(path)
        if (any(report.get(key) != value for key, value in metadata.items())
                or file_sha256(args.output / "predictions.csv") != report["predictions_sha256"]
                or file_sha256(args.output / "comparison.csv") != report["comparison_table_sha256"]):
            raise ValueError("Controlled comparison input or saved results differ")
    else:
        report = {**metadata, **save_validation(args, info, prepared), "environment": environment()}
        publish(path, json_bytes(report))
    shortlist = {
        name: {key: item[key] for key in ("images", "accuracy_effect", "flip_rate", "wins", "losses",
                                          "mcnemar_holm_p", "decision")}
        for name, item in report["comparisons"].items()
    }
    print_json({"comparisons": shortlist, **{key: report[key] for key in (
        "geometry", "total_image_forwards", "total_wall_seconds", "peak_reserved_gib")}})


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V8 對照診斷已有執行中的階段。")
        info = sources(args)
        if args.stage == "prepare":
            prepare(args, info)
        else:
            validate(args, info)


if __name__ == "__main__":
    main()
