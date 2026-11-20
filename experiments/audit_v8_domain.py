"""在 CPU 上比較 V8 影像分布與封存的分類紀錄。"""

import argparse
import fcntl
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import BadZipFile

import numpy as np

from classification_v9 import audit_features, sources, validation_examples
from classification_v12 import shifted_cohort
from v2.data_checks import read_json
from v3.inference import VIEWS
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v4.geometry import seeded
from v6.composite import Targets
from v8.data import CONDITIONS, read_cases
from v8_audit.artifacts import load_v9, load_v9_csv, load_validation_csv, validation_folders
from v8_audit.measure import (COMBOS, GROUPS, GROUP_NAMES, SCORES, SPATIAL, VISUAL, Scan, flag, group_rows,
                              novelty_record, percentile_rows, scan)
from v12.protocol import SHIFT

OPTIONAL_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, BadZipFile)
RULES_URL = "https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--output", type=Path, default=Path("runs/v8_domain_audit"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--v9-run", type=Path, default=Path("runs/v9"))
    p.add_argument("--v10-run", type=Path, default=Path("runs/v10"))
    p.add_argument("--v12-run", type=Path, default=Path("runs/v12"))
    p.add_argument("--v13-run", type=Path, default=Path("runs/v13"))
    p.add_argument("--v11-audit", type=Path, default=Path("runs/v11/audit.json"))
    p.add_argument("--audit-run", type=Path, default=Path("runs/distribution_audit"))
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    p.add_argument("--train-sample", type=int, default=1000, choices=range(1, 5001), metavar="1..5000")
    p.add_argument("--test-sample", type=int, default=768, choices=range(1, 1501), metavar="1..1500")
    args = p.parse_args(argv)
    out, root = args.output.resolve(), Path("runs").resolve() / "v8_domain_audit"
    protected = (args.data, args.v8_run, args.v9_run, args.v10_run, args.v12_run, args.v13_run, args.audit_run,
                 args.v11_audit.parent)
    if (root.is_symlink() or not out.is_relative_to(root) or any(
        out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in protected
    )):
        p.error("--output must be independent under runs/v8_domain_audit")
    args.checkpoint = args.v8_run / "models/convnext_all_layers/best.pt"
    args.v8_submission = args.v8_run / "candidates/convnext_tta_denoise/submission_v8.csv"
    return args


def existing(candidates, warnings):
    for path, reader in candidates:
        if not path.is_file():
            continue
        try:
            return reader()
        except OPTIONAL_ERRORS as exc:
            warnings.append({"path": str(path), "reason": str(exc)})
    return {}, {"source": None, "images": 0}


def test_predictions(args, info, warnings):
    folder = args.v9_run / "diagnostic"
    from v9.views import Example

    examples = [Example(path, item) for path, item in info.samples]
    rows, meta = existing(((folder / "views.npz", lambda: load_v9(
        folder / "views.npz", info, examples, "diagnostic"
    )), (folder / "per_image.csv", lambda: load_v9_csv(folder, info))), warnings)
    mismatches = [item for item, score in rows.items() if score["prediction"] != info.archived[item]]
    matched = {item: score for item, score in rows.items() if score["prediction"] == info.archived[item]}
    total = len(info.samples)
    return matched, {**meta, "total_test_images": total, "matched_archive_rows": len(matched),
                     "matched_archive_coverage": len(matched) / total, "top1_replay_mismatches": len(mismatches),
                     "replay_mismatch_ids_first_20": mismatches[:20],
                     "subset_only": len(matched) != total}


def validation_predictions(args, info, cohorts, audit_sha, warnings):
    root = args.data / "train/train/train"
    scores, meta = {}, {}
    for condition, cases in cohorts.items():
        examples = validation_examples(args, cases)
        bank = args.v9_run / "screen" / f"{condition}.npz"
        folders = list(validation_folders(args, condition))
        candidates = [(bank, lambda: load_v9(bank, info, examples, "screen", condition))] if condition != SHIFT else []
        candidates += [(folder / "summary.json", lambda folder=folder, condition=condition:
                        load_validation_csv(folder, info, cases, condition, root, audit_sha)) for folder in folders]
        scores[condition], meta[condition] = existing(candidates, warnings)
    return scores, meta


def validation_cohorts(args, info, warnings):
    cases = [case for case in info.validation if case.fold == "screen"]
    cohorts = {condition: cases for condition in CONDITIONS}
    path = args.v11_audit
    try:
        legacy = read_json(args.v8_run / "benchmark/prepare.json")
        identity = {**info.provenance, "v8_image_signature": legacy["selected_image_signature"],
                    "validation_seed": info.inputs["seed"]}
        context = SimpleNamespace(identity=identity, inputs=info.inputs, validation=info.validation)
        cohort = shifted_cohort(path if path.is_file() else None, context)
        cohorts[SHIFT] = [case for case in cohort if case.fold == "screen"]
        return cohorts, file_sha256(path) if path.is_file() else None
    except OPTIONAL_ERRORS as exc:
        warnings.append({"path": str(path), "reason": str(exc)})
        return cohorts, None


def pick(rows, count, seed, key):
    pool = sorted(rows)
    return seeded(seed, key).sample(pool, min(count, len(pool)))


def scan_scope(tasks, args, failures, label):
    rows, failed = scan(tasks, args.workers)
    if failed:
        failures[label] = failed
    return rows


def test_visuals(args, info, scores, warnings, failures):
    paths = {item: path for path, item in info.samples}
    seed = info.inputs["seed"]
    sampled = pick(paths, args.test_sample, seed, "v8domain:uniform_test")
    scored = pick(scores, args.test_sample, seed, "v8domain:cached_spatial")
    union = list(dict.fromkeys([*sampled, *scored]))
    fresh = scan_scope([Scan(item, paths[item]) for item in union], args, failures, "test_sample")
    try:
        _, legacy = audit_features(args, info)
        rows = {item: novelty_record(record) for item, record in legacy.items()}
        # 舊稽核有 ID 核對，沒有逐張影像雜湊。用本次抽樣對照其量測。
        for item in sampled[:16]:
            if item not in fresh or any(not np.isclose(rows[item][name], fresh[item][name], rtol=0, atol=1e-5)
                                       for name in ("width", "height", "brightness", "black_edge_fraction", "fine_detail")):
                raise ValueError(f"Distribution audit disagrees with current image or Pillow decode for {item}")
        source = {"mode": "verified_legacy_full_test_with_spot_check", "rows": len(rows),
                  "sha256": file_sha256(args.audit_run / "test_novelty.csv"), "spot_check_images": min(16, len(sampled))}
    except OPTIONAL_ERRORS as exc:
        warnings.append({"path": str(args.audit_run), "reason": str(exc)})
        rows = {item: fresh[item] for item in sampled if item in fresh}
        source = {"mode": "uniform_test_sample", "rows": len(rows), "sha256": None}
    population = {item: {key: value for key, value in row.items() if key in VISUAL and key not in SPATIAL}
                  for item, row in rows.items()}
    return population, {item: fresh[item] for item in sampled if item in fresh}, fresh, source


def inference_cost(test_size, test_meta, val_meta):
    views = test_size * VIEWS["multiscale"]
    source = test_meta if test_meta.get("wall_seconds") and test_meta.get("image_forwards") else val_meta.get("clean", {})
    count, time = source.get("image_forwards"), source.get("wall_seconds")
    if count and time and count > 0 and time > 0 and np.isfinite(time):
        return {"image_forwards": views, "projected_minutes": views / count * time / 60,
                "mode": "linear_projection_same_recorded_hardware", "source": source["source"],
                "gpu_device_recorded": source.get("gpu_device_recorded"), "recorded_seconds": time,
                "recorded_image_forwards": count}
    return {"image_forwards": views, "mode": "scenario_only_no_gpu_measurement",
            "minutes_at_views_per_second_excluding_io": {str(rate): views / rate / 60 for rate in (200, 500, 1000)}}


def class_rows(train, labels, classes):
    train_count = Counter(case.label for case in train)
    test_count = Counter(labels.values())
    return [{"character": name, "train_label_n": train_count[i], "test_predicted_n": test_count[i],
             "train_label_pct": 100 * train_count[i] / len(train), "test_predicted_pct": 100 * test_count[i] / len(labels),
             "predicted_minus_train_pp": 100 * (test_count[i] / len(labels) - train_count[i] / len(train))}
            for i, name in enumerate(classes)]


def score_by_visual_flag(rows):
    result = {}
    for key in GROUP_NAMES:
        if key in GROUPS and GROUPS[key][0] in SCORES:
            continue
        buckets = {"flagged": [], "rest": []}
        for row in rows:
            status = flag(row, key)
            if status is not None and "confidence" in row:
                buckets["flagged" if status else "rest"].append(row)
        if any(buckets.values()):
            result[key] = {name: {"images": len(group), "scores": percentile_rows(group, SCORES)}
                           for name, group in buckets.items()}
    return result


def save_group_csv(path, groups):
    fields = ("cohort", "group", "eligible", "flagged", "flagged_rate", "errors", "error_rate",
              "ci95_low", "ci95_high", "rest_count", "rest_errors", "rest_error_rate",
              "rest_ci95_low", "rest_ci95_high", "error_gap_pp", "exploratory_signal")
    packed = []
    for cohort, rows in groups.items():
        for row in rows:
            result = {**row, "cohort": cohort}
            for prefix in ("", "rest_"):
                low, high = result.get(prefix + "ci95") or (None, None)
                result[prefix + "ci95_low"], result[prefix + "ci95_high"] = low, high
            packed.append(result)
    publish_csv(path, fields, ((row.get(key) for key in fields) for row in packed))


def review_rows(info, root, test_rows, val_rows, truth, uniform):
    paths = dict((item, path.as_posix()) for path, item in info.samples)
    claims = []
    used = set()
    classes = info.inputs["classes"]

    def add(ids, scope, reason, limit):
        n = 0
        for item in ids:
            if (scope, item) in used:
                continue
            used.add((scope, item))
            row = test_rows[item] if scope == "test" else val_rows[scope][item]
            label = "" if scope == "test" else classes[truth[item]]
            pred = row.get("prediction")
            claims.append((scope, item, paths[item] if scope == "test" else (root / item).as_posix(),
                           reason, classes[pred] if pred is not None else "", label, row.get("confidence"),
                           "|".join(key for key in GROUP_NAMES if flag(row, key))))
            n += 1
            if n == limit:
                break

    add(uniform, "test", "uniform_sample_control", 3)
    low = sorted((item for item in test_rows if test_rows[item].get("confidence") is not None),
                 key=lambda item: test_rows[item]["confidence"])
    add(low, "test", "low_confidence_cached", 3)
    for key in ("bright", "crop_activity_loss_proxy", "offcenter_activity_proxy", "compact_activity_proxy",
                "separated_activity_peaks_proxy"):
        add((item for item in uniform if flag(test_rows[item], key)), "test", key, 2)
    for scope in ("clean", "context", "distractor"):
        if scope in val_rows:
            add((item for item, row in val_rows[scope].items()
                 if "prediction" in row and row["prediction"] != truth[item]), scope, "verified_validation_error", 2)
    return claims


def compact(summary):
    return {
        "test_size": summary["test_size"],
        "measurements": summary["measurement_populations"],
        "test_scores": {key: summary["test_score_source"][key] for key in (
            "source", "matched_archive_rows", "top1_replay_mismatches", "matched_archive_coverage")},
        "test_score_quantiles": summary["distributions"]["test_v8_matched_scores"],
        "score_coverage": summary["test_score_rows_by_metric"],
        "test_view_agreement": {key: summary["test_score_source"].get(key) for key in (
            "mean_view_disagreement", "unanimous_rate")},
        "test_most_predicted": summary["test_most_predicted"],
        "largest_visual_differences": summary["visual_median_differences"],
        "validation_errors": summary["validation_errors"],
        "exploratory_validation_signals": {
            condition: {key: rows[0][key] for key in ("group", "flagged", "error_gap_pp")}
            for condition, rows in summary["exploratory_validation_signals"].items() if rows
        },
        "inference_cost_if_needed": summary["inference_cost_if_needed"],
        "warnings": len(summary["optional_artifact_warnings"]),
    }


def run(args):
    info = sources(args)
    root = args.data / "train/train/train"
    train = read_cases(args.v8_run / "benchmark/train.csv", root, info.inputs["classes"], ("train",))
    seed = info.inputs["seed"]
    warnings, failures = [], {}
    test_scores, test_meta = test_predictions(args, info, warnings)
    cohorts, v11_sha = validation_cohorts(args, info, warnings)
    val_scores, val_meta = validation_predictions(args, info, cohorts, v11_sha, warnings)
    print("V8 domain audit | CPU image statistics and verified archived predictions", flush=True)
    population, uniform, spatial, visual_source = test_visuals(args, info, test_scores, warnings, failures)
    train_ids = pick([case.item for case in train], args.train_sample, seed, "v8domain:uniform_train")
    train_rows = scan_scope([Scan(item, root / item) for item in train_ids], args, failures, "train_sample")
    truth = {case.item: case.label for case in cohorts["clean"]}
    targets = Targets(**info.inputs["targets"])
    validation = {}
    for condition, cases in cohorts.items():
        if not val_scores[condition] and condition != "clean":
            continue
        tasks = [Scan(case.item, root / case.item, condition, seed, root / case.distractor, targets) for case in cases]
        values = scan_scope(tasks, args, failures, f"validation_{condition}")
        validation[condition] = {item: {**row, **val_scores[condition].get(item, {})}
                                 for item, row in values.items()}
    test_rows = {item: {**population.get(item, {}), **spatial.get(item, {}), **test_scores.get(item, {}),
                        "prediction": info.archived[item]} for _, item in info.samples}
    groups = {
        "test_visual": group_rows(population), "test_uniform_spatial_sample": group_rows(uniform),
        "test_v8_matched_scores": group_rows(test_scores),
        **{f"validation_{key}": group_rows(rows, truth) for key, rows in validation.items()},
    }
    test_sample_scores = {item: test_rows[item] for item in test_scores if item in spatial}
    groups["test_spatial_and_v8_scores_overlap"] = group_rows(test_sample_scores)
    distributions = {
        "train_sample": percentile_rows(list(train_rows.values()), VISUAL),
        "test_visual": percentile_rows(list(population.values()), VISUAL),
        "test_uniform_spatial_sample": percentile_rows(list(uniform.values()), VISUAL),
        "test_v8_matched_scores": percentile_rows(list(test_scores.values()), SCORES),
        **{f"validation_{key}": percentile_rows(list(rows.values()), (*VISUAL, *SCORES))
           for key, rows in validation.items()},
    }
    errors, signals = {}, {}
    for condition, scores in val_scores.items():
        if scores:
            errors[condition] = {"images": len(scores),
                                 "errors": sum(row["prediction"] != truth[item] for item, row in scores.items())}
            strongest = sorted((row for row in groups[f"validation_{condition}"] if row["exploratory_signal"]),
                              key=lambda row: row["error_gap_pp"], reverse=True)
            signals[condition] = strongest[:3]
    differences = {}
    for key in ("black_edge_fraction", "brightness", "fine_detail", "outer_activity", "aspect_ratio",
                "activity_offcenter", "center_crop_activity_loss"):
        dists = {name: distributions[name].get(key) for name in ("train_sample", "validation_clean",
                                                                 "test_uniform_spatial_sample")}
        differences[key] = {name: None if stats is None else stats["p10_p50_p90"][1] for name, stats in dists.items()}
    classes = class_rows(train, info.archived, info.inputs["classes"])
    fields = tuple(classes[0])
    publish_csv(args.output / "prediction_classes.csv", fields, ((row[key] for key in fields) for row in classes))
    save_group_csv(args.output / "risk_groups.csv", groups)
    test_fields = ("id", "archived_v8", *VISUAL, *SCORES, "risk_flags")
    publish_csv(args.output / "test_risk.csv", test_fields, (
        (item, info.inputs["classes"][row["prediction"]], *(row.get(key) for key in (*VISUAL, *SCORES)),
         "|".join(key for key in GROUP_NAMES if flag(row, key))) for item, row in test_rows.items()
    ))
    publish_csv(args.output / "review.csv", ("cohort_or_condition", "id", "image_path", "reason", "v8_prediction",
                                            "validation_label", "confidence", "risk_flags"),
                review_rows(info, root, test_rows, validation, truth, uniform))
    summary = {
        "provenance": info.provenance, "rules": {"url": RULES_URL, "status": "not_verified",
            "scope": "general_image_statistics_and_existing_verified_predictions"},
        "test_size": len(info.samples), "visual_source": visual_source, "test_score_source": test_meta,
        "test_score_rows_by_metric": {key: sum(key in row for row in test_scores.values()) for key in SCORES},
        "validation_score_sources": val_meta, "distributions": distributions,
        "measurement_populations": {"train_sample": len(train_rows), "validation_clean": len(validation["clean"]),
            "test_visual": len(population), "test_uniform_spatial_sample": len(uniform),
            "test_spatial_score_overlap": len(test_sample_scores)},
        "visual_median_differences": differences, "groups": groups, "thresholds": GROUPS, "combinations": COMBOS,
        "v11_audit_sha256": v11_sha,
        "test_score_by_visual_group": score_by_visual_flag(test_rows.values()),
        "test_most_predicted": [{"character": row["character"], "count": row["test_predicted_n"]}
                                for row in sorted(classes, key=lambda row: -row["test_predicted_n"])[:5]],
        "validation_errors": errors, "exploratory_validation_signals": signals,
        "prediction_total_variation_vs_train_label_share": sum(abs(row["predicted_minus_train_pp"]) for row in classes) / 200,
        "inference_cost_if_needed": inference_cost(len(info.samples), test_meta, val_meta),
        "optional_artifact_warnings": warnings, "image_read_failures": failures,
    }
    publish(args.output / "summary.json", json_bytes(summary))
    print(json_bytes(compact(summary)).decode(), end="", flush=True)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "audit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V8 domain audit 輸出正在使用。")
        run(args)


if __name__ == "__main__":
    main()
