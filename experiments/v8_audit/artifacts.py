"""唯讀驗證 V9 機率與較新版本的 V8 驗證預測。"""

import csv
import json
from pathlib import Path

import numpy as np

from v2.data_checks import read_json
from v3.storage import file_sha256, image_signature
from v8.data import selected_signature
from v8.experiment import load_evaluation
from v9.views import Example, RECIPES, REFERENCE, cache_bank


def verify_identity(saved, expected):
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("prediction provenance does not match the deployed V8 archive")


def probability_metrics(probabilities, view_labels):
    mean = probabilities.mean(axis=0)
    top2 = np.sort(np.partition(mean, -2, axis=1)[:, -2:], axis=1)
    choice = mean.argmax(axis=1)
    logarithm = np.zeros_like(mean)
    np.log(mean, out=logarithm, where=mean > 0)
    entropy = -(mean * logarithm).sum(axis=1)
    disagreement = [1 - np.unique(row, return_counts=True)[1].max() / len(row) for row in view_labels.T]
    return [{
        "prediction": int(choice[i]), "confidence": float(top2[i, 1]),
        "margin": float(top2[i, 1] - top2[i, 0]), "entropy_nats": float(entropy[i]),
        "view_disagreement": float(disagreement[i]),
    } for i in range(len(choice))]


def load_v9(path, info, examples, phase, condition=None):
    # Checkpoints are fingerprinted in V9; this always reads cached scores on CPU.
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["signature"].item())
        ids = data["ids"].tolist()
    verify_identity(metadata, info.provenance)
    if metadata.get("phase") != phase or metadata.get("class_count") != len(info.inputs["classes"]):
        raise ValueError("V9 cache phase or class count does not match")
    expected_ids = [ex.item for ex in examples]
    if (not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= set(expected_ids)):
        raise ValueError("V9 cache has invalid or unexpected IDs")
    if phase == "screen":
        signature = image_signature(p for ex in examples for p in (ex.path, ex.distractor))
        if ids != expected_ids or metadata.get("image_signature") != signature or metadata.get("condition") != condition:
            raise ValueError("V9 validation cache does not match the original Screen images")
    raw_plan = metadata.pop("views")
    metadata.pop("class_count")
    if (not isinstance(raw_plan, list) or not all(isinstance(pair, list) and len(pair) == 2
                                                  and type(pair[0]) is bool and isinstance(pair[1], str)
                                                  for pair in raw_plan)):
        raise ValueError("invalid V9 view definitions")
    plan = [tuple(pair) for pair in raw_plan]
    reference = [(True, view) for view in RECIPES[REFERENCE].views]
    if len(set(plan)) != len(plan) or not set(reference) <= set(plan):
        raise ValueError("missing or duplicate V8 view records")

    def no_recompute():
        raise FileNotFoundError(path)

    by_id = {ex.item: ex for ex in examples}
    bank = cache_bank(path, [by_id[item] for item in ids], plan, metadata, len(info.inputs["classes"]), no_recompute)
    indices = [plan.index(view) for view in reference]
    views = bank.labels[indices]
    rows = dict(zip(ids, probability_metrics(bank.probabilities[indices], views), strict=True))
    return rows, {
        "source": str(path), "sha256": file_sha256(path), "images": len(ids),
        "mean_view_disagreement": float(np.mean([row["view_disagreement"] for row in rows.values()])),
        "unanimous_rate": float((views == views[:1]).all(axis=0).mean()),
        "pairwise_agreement": {
            name: {other: float((views[i] == views[j]).mean()) for j, (_, other) in enumerate(reference)}
            for i, (_, name) in enumerate(reference)
        },
        "wall_seconds": bank.wall_seconds, "image_forwards": bank.forwards,
        "gpu_device_recorded": metadata.get("environment", {}).get("gpu"),
    }


def load_v9_csv(folder, info):
    summary = read_json(folder / "summary.json")
    verify_identity(summary["provenance"], info.provenance)
    if file_sha256(folder / "by_group.csv") != summary["group_table_sha256"]:
        raise ValueError("V9 group table fingerprint mismatch")
    classes = {name: i for i, name in enumerate(info.inputs["classes"])}
    path = folder / "per_image.csv"
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["method"] != REFERENCE:
                continue
            item = row["id"]
            if (item in rows or item not in info.archived or classes[row["archived_v8"]] != info.archived[item]):
                raise ValueError("invalid V9 per-image archive ID or V8 prediction")
            values = {key: float(row[field]) for key, field in
                      (("confidence", "confidence"), ("margin", "top2_margin"),
                       ("view_disagreement", "view_disagreement"))}
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in values.values()):
                raise ValueError("invalid V9 per-image score")
            rows[item] = {"prediction": classes[row["prediction"]], **values}
    if not rows or len(rows) != summary["samples"]:
        raise ValueError("V9 per-image rows do not match the recorded sample")
    if sum(score["prediction"] != info.archived[item] for item, score in rows.items()) != summary["v8_replay_mismatches"]:
        raise ValueError("V9 replay mismatch count differs")
    with (folder / "by_group.csv").open(encoding="utf-8-sig", newline="") as file:
        aggregate = [row for row in csv.DictReader(file) if row["method"] == REFERENCE
                     and row["dimension"] == "all" and row["group"] == "all"]
    if len(aggregate) != 1 or int(aggregate[0]["images"]) != len(rows):
        raise ValueError("V9 per-image rows do not match the aggregate")
    for metric, key in (("confidence", "mean_confidence"), ("margin", "mean_top2_margin"),
                        ("view_disagreement", "mean_view_disagreement")):
        if not np.isclose(np.mean([row[metric] for row in rows.values()]), float(aggregate[0][key]), rtol=0, atol=1e-7):
            raise ValueError("V9 per-image summary mismatch")
    return rows, {"source": str(path), "sha256": file_sha256(path), "images": len(rows),
                  "scope": "per_image_csv_aggregate_checked", "wall_seconds": summary["wall_seconds"],
                  "image_forwards": summary["image_forwards"],
                  "gpu_device_recorded": summary.get("environment", {}).get("gpu")}


def load_validation_csv(folder, info, cases, condition, root, audit_sha=None):
    path = folder / "summary.json"
    settings = read_json(path)
    v10 = settings.get("provenance", {})
    signature = settings.get("image_signature") or settings.get("validation_image_signature")
    digest = settings.get("checkpoint_sha256")
    if digest is None and settings.get("arm") == "v8":
        digest = v10.get("v8_checkpoint_sha256")
    fold = settings.get("fold") or settings.get("phase")
    if (fold != "screen" or settings.get("method") != "multiscale"
            or settings.get("denoise") is not True or settings.get("precision") != "fp32"
            or condition not in settings.get("conditions", [])
            or signature != selected_signature(root, cases)
            or digest != info.provenance["checkpoint_sha256"]):
        raise ValueError("V8 validation report does not match the archive model, fold or images")
    source_identity = settings.get("identity") or settings.get("context", {}).get("identity")
    if source_identity:
        verify_identity(source_identity, {"checkpoint_sha256": info.provenance["checkpoint_sha256"],
                                          "benchmark_sha256": info.provenance["benchmark_sha256"]})
        if condition == "heldout_composite" and source_identity.get("v11_audit_sha256") != audit_sha:
            raise ValueError("Heldout Composite uses a different V11 audit")
    elif (settings.get("manifest_sha256") or v10.get("validation_manifest_sha256")) != info.provenance["validation_manifest_sha256"]:
        raise ValueError("V8 validation manifest does not match")
    if settings.get("seed", v10.get("validation_seed", info.inputs["seed"])) != info.inputs["seed"]:
        raise ValueError("V8 validation stress seed does not match")
    values, _ = load_evaluation(folder, cases, settings, info.inputs["classes"])
    scores = {case.item: {"prediction": int(value)} for case, value in zip(cases, values[condition], strict=True)}
    return scores, {"source": str(folder / "predictions.csv"), "images": len(scores),
                    "wall_seconds": settings["timings"].get(condition, {}).get("seconds"),
                    "image_forwards": len(scores) * len(RECIPES[REFERENCE].views),
                    "gpu_device_recorded": (settings.get("environment") or
                                            settings.get("context", {}).get("environment", {})).get("gpu")}


def validation_folders(args, condition):
    yield args.v13_run / "scores/screen/reference/v8" / condition
    yield args.v12_run / "scores/screen/v8" / condition
    yield args.v10_run / "screen/v8"
    report_dir = args.v8_run / "reports/screen"
    yield from sorted(path.parent for path in report_dir.glob("*/summary.json"))
