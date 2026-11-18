"""只驗證已確認的 ID 155 歷史差異，原始快取全程唯讀。"""

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from v2.data_checks import read_json
from v3.baseline_probe import locate_batch
from v3.storage import json_bytes, read_baseline_snapshot

ALLOWED_ID = "155"
HISTORICAL_LABEL = "kent_brockman"
CURRENT_LABEL = "timothy_lovejoy"


def baseline_result(ids, predictions, reference, settings):
    mismatches = [item for item in ids if predictions[item].label != reference[item]]
    return {
        "settings": settings, "samples": len(ids),
        "mismatch_count": len(mismatches), "mismatch_ids": mismatches,
    }


def require(condition, reason):
    if not condition:
        raise ValueError(f"Historical Baseline exception rejected: {reason}")


def validate_probe(report, ids, metadata, reference_hash, source_path, live_audit):
    require(isinstance(report, dict), "expected a Probe JSON object")
    require(report.get("id") == ALLOWED_ID and report.get("historical_v2_label") == HISTORICAL_LABEL,
            "Probe targets a different ID or historical label")
    expected = json.loads(json_bytes(metadata))
    require(report.get("expected_cache_metadata") == expected
            and report.get("v2_reference_sha256") == reference_hash,
            "Probe checkpoint, inputs, settings, or V2 CSV do not match")

    audit = report.get("cache")
    require(isinstance(audit, dict) and audit.get("status") == "inspected"
            and audit.get("prior_baseline_check_reference_matches") is True
            and isinstance(audit.get("path"), str)
            and Path(audit["path"]).resolve() == source_path.resolve(),
            "Probe did not inspect the matching historical Baseline cache")
    for key in ("integrity_check", "saved_cache_version", "metadata_matches", "changed_metadata_keys",
                "baseline_count", "preview_count", "baseline_structure_issues",
                "baseline_target", "preview_target", "compatible"):
        require(key in audit and audit[key] == live_audit[key], f"Probe cache field differs: {key}")
    require(audit["compatible"] is True and audit["metadata_matches"] is True,
            "Probe cache compatibility is not established")

    size = metadata["reference_baseline_batch_size"]
    start, stop, offset = locate_batch(ids, ALLOWED_ID, size)
    settings = report.get("replay_settings")
    require(isinstance(settings, dict) and settings.get("batch_ids") == ids[start:stop]
            and settings.get("batch_size") == size and settings.get("actual_batch_size") == stop - start
            and settings.get("batch_start_index_zero_based") == start
            and settings.get("position_in_batch_zero_based") == offset
            and settings.get("cudnn_benchmark") is True and settings.get("cudnn_deterministic") is False,
            "Probe did not replay the original batch with the Baseline settings")

    comparison = report.get("replay_comparison")
    require(isinstance(comparison, dict) and comparison.get("batch_input_bitwise_equal") is True
            and comparison.get("batch_logits_bitwise_equal") is True
            and all(type(comparison.get(key)) in (int, float) and comparison[key] == 0
                    for key in ("batch_logit_max_abs_diff", "target_logit_max_abs_diff")),
            "Probe input and logits are not proven bitwise equal")
    replays = report.get("replays")
    require(isinstance(replays, dict) and set(replays) == {"v2", "v3"}, "both replays are required")
    input_hashes, top_scores = [], []
    for replay in replays.values():
        require(isinstance(replay, dict) and replay.get("top1") == CURRENT_LABEL
                and replay.get("target_matches_cached_baseline") is True
                and replay.get("batch_reference_mismatch_ids") == [ALLOWED_ID]
                and replay.get("logits_dtype") == "float32",
                "replay does not reproduce the current Baseline and its sole historical mismatch")
        digest = replay.get("input_tensor_sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "replay input fingerprint is missing or invalid")
        input_hashes.append(digest)
        top = replay.get("top5")
        require(isinstance(top, list) and len(top) == 5 and all(
            isinstance(row, dict) and row.get("character") in metadata["classes"]
            and type(row.get("logit")) in (float, int) and math.isfinite(row["logit"])
            and type(row.get("probability")) in (float, int) and 0 <= row["probability"] <= 1
            for row in top
        ), "replay top-five scores are incomplete or invalid")
        require([row["character"] for row in top[:2]] == [CURRENT_LABEL, HISTORICAL_LABEL],
                "replay top-two labels differ from the confirmed case")
        top_scores.append([(row["character"], row["logit"]) for row in top])
    require(input_hashes[0] == input_hashes[1], "replay input fingerprints differ")
    require(top_scores[0] == top_scores[1], "replay top-five logits differ")


def authorize_exception(source, probe_path, allowed_id, ids, reference, metadata, reference_hash):
    require(allowed_id == ALLOWED_ID, f"only ID {ALLOWED_ID} is supported")
    source_path = source / "test.sqlite3"
    classes = metadata["classes"]
    audit, predictions = read_baseline_snapshot(
        source_path, metadata, ids, metadata["reference_baseline_batch_size"], classes, ALLOWED_ID,
    )
    require(audit["compatible"] and len(predictions) == len(ids),
            "source cache is incomplete or incompatible")
    settings = {**metadata, "v2_reference_sha256": reference_hash}
    current = baseline_result(ids, predictions, reference, settings)
    require(current["mismatch_ids"] == [ALLOWED_ID],
            "the sole mismatch must be ID 155 across the full test set")
    require(classes[reference[ALLOWED_ID]] == HISTORICAL_LABEL
            and classes[predictions[ALLOWED_ID].label] == CURRENT_LABEL,
            "historical or current label is different from the confirmed case")
    require(read_json(source / "baseline_check.json") == json.loads(json_bytes(current)),
            "original baseline_check.json does not match the live cache and V2 CSV")

    raw_probe = probe_path.read_bytes()
    report = json.loads(raw_probe)
    validate_probe(report, ids, metadata, reference_hash, source_path, audit)
    snapshot = json_bytes([[item, asdict(predictions[item])] for item in ids])
    return predictions, {
        "allowed_id": ALLOWED_ID, "historical_label": HISTORICAL_LABEL, "current_label": CURRENT_LABEL,
        "sample_count": len(ids), "historical_mismatch_ids": current["mismatch_ids"],
        "v2_checkpoint_sha256": metadata["v2_checkpoint_sha256"],
        "v2_reference_sha256": reference_hash,
        "source_cache": str(source_path.resolve()),
        "source_baseline_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "probe_report": str(probe_path.resolve()),
        "probe_report_sha256": hashlib.sha256(raw_probe).hexdigest(),
        "probe": report,
    }
