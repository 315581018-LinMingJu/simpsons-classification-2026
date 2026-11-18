"""依全量稽核挑選真實 Test 複核圖，並附上可取得的 V6 訊號。"""

import csv
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from ensemble.data import checked_probs
from v2.data_checks import read_json
from v3.storage import file_sha256, publish_csv
from v4.analysis import save_sheet
from v4.geometry import seeded
from v7.impulse import MIN_TRIGGER_FRACTION, clean_impulses, impulse_mask

# 只標記附件中看得到的線索。沒有 Test 標籤或目標角色框。
OBSERVED = {
    "visible_color_speckle": ("6171", "7126", "9107", "2842", "10399", "5402"),
    "visible_background_texture": ("1369", "7182", "9816", "9736", "2056"),
    "small_visible_patch": ("6949", "8022", "131"),
    "off_center_visible_patch": ("6392", "7949", "2682"),
    "partial_crop_or_possible_occlusion": ("810", "6392", "9734", "1737"),
    "multiple_visible_characters": ("7812", "131", "7051"),
    "possible_negative_colors": ("6042", "5830", "2555"),
    "rotation_or_perspective": ("3186", "9736", "8022"),
}


def inspect_test(path):
    with Image.open(path) as image:
        return float(impulse_mask(image.convert("RGB")).mean())


def audit_rows(path, ids, decoded_count):
    allowed, results = set(ids), {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            item = row["id"]
            if item not in allowed or item in results:
                raise ValueError(f"Unexpected or duplicate Test ID in distribution audit: {item}")
            detail = float(row["fine_detail"])
            if not np.isfinite(detail) or detail < 0:
                raise ValueError(f"Invalid fine detail score: {item}")
            results[item] = row
    if len(results) != decoded_count:
        raise ValueError("Distribution audit decoded count differs from the novelty table")
    return results


def review_groups(signals, rows, seed):
    activated = [item for item, fraction in signals.items() if fraction >= MIN_TRIGGER_FRACTION]
    high_proxy = sorted(activated, key=lambda item: (-signals[item], item))[:6]
    high_detail = sorted((item for item, fraction in signals.items() if fraction < MIN_TRIGGER_FRACTION
                          and item in rows), key=lambda item: (-float(rows[item]["fine_detail"]), item))[:6]
    random_ids = seeded(seed, "v7:test-review").sample(sorted(signals), min(6, len(signals)))
    groups = [
        ("high_impulse_proxy", "manual_confirmation_required", high_proxy),
        ("high_detail_low_impulse_proxy", "manual_confirmation_required", high_detail),
        ("random_test", "manual_confirmation_required", random_ids),
    ]
    uncertain = {"partial_crop_or_possible_occlusion", "possible_negative_colors", "rotation_or_perspective"}
    groups.extend((label, "manual_confirmation_required" if label in uncertain else "visually_reviewed_examples",
                   [item for item in items if item in signals]) for label, items in OBSERVED.items())
    return [(label, status, items) for label, status, items in groups if items]


def cached_v6(path, ids, classes, checkpoint_hash, test_signature, archived_path):
    if not path.is_file():
        return None, "unavailable"
    try:
        settings = read_json(path.parent / "settings.json")
        inference = settings["inference"]
        matches = (inference["classes"] == classes and inference["method"] == "multiscale"
                   and inference["checkpoints"]["v6"] == checkpoint_hash
                   and settings["test_image_signature"] == test_signature
                   and settings["reference_sha256"]["v6"] == file_sha256(archived_path))
        if not matches:
            return None, "incompatible_provenance"
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids:
                return None, "incompatible_ids"
            matrix = checked_probs(data["probabilities"], (len(ids), len(classes)))
    except (BadZipFile, OSError, ValueError, KeyError) as exc:
        return None, f"unusable: {type(exc).__name__}"
    return matrix, "reused_verified_v6_probabilities"


def summarize_groups(ids, signals, audit, probabilities, archived):
    categories = {}
    for index, item in enumerate(ids):
        row = audit.get(item, {})
        for axis, value in (
            ("visual_band", row.get("visual_band", "unknown")),
            ("combo_support", row.get("combo_support", "unknown")),
            ("all", "all"),
        ):
            key = f"{axis}:{value}"
            values = categories.setdefault(key, {"images": 0, "activated": 0, "confidence": [],
                                                  "activated_confidence": [], "others_confidence": []})
            values["images"] += 1
            active = item in signals and signals[item] >= MIN_TRIGGER_FRACTION
            values["activated"] += active
            if probabilities is not None:
                confidence = float(probabilities[index].max())
                values["confidence"].append(confidence)
                values["activated_confidence" if active else "others_confidence"].append(confidence)
    for values in categories.values():
        for field in ("confidence", "activated_confidence", "others_confidence"):
            recorded = values[field]
            values[field] = float(np.mean(recorded)) if recorded else None
        values["activation_pct"] = 100 * values["activated"] / values["images"]
    mismatch = None if probabilities is None else sum(
        int(value) != archived[item] for item, value in zip(ids, probabilities.argmax(axis=1), strict=True)
    )
    return categories, mismatch


def write_montages(output, groups, files, signals, audit, archived, classes, probabilities, ids):
    indices = {item: index for index, item in enumerate(ids)}
    cases, sheet = [], []
    for group, status, members in groups:
        sheet.append([])
        for item in members:
            confidence, margin = None, None
            replay = ""
            if probabilities is not None:
                row = probabilities[indices[item]]
                high = np.sort(row)[-2:]
                confidence, margin = float(high[-1]), float(high[-1] - high[-2])
                replay = classes[int(row.argmax())]
            caption = f"{group}\nID {item} | {signals[item]:.2%}"
            if status == "manual_confirmation_required":
                caption += "\nmanual review"
            sheet[-1].append((caption, files[item]))
            cases.append((group, status, item, signals[item], signals[item] >= MIN_TRIGGER_FRACTION,
                          audit.get(item, {}).get("visual_band", "unknown"),
                          audit.get(item, {}).get("fine_detail", ""), "" if archived is None else classes[archived[item]],
                          replay, confidence, margin, ""))
    save_sheet(output / "test_categories.png", sheet)
    high = next((items for group, _, items in groups if group == "high_impulse_proxy"), [])
    pairs = []
    for item in high:
        with Image.open(files[item]) as image:
            original = image.convert("RGB")
        proposed, fraction = clean_impulses(original)
        pairs.append([(f"ID {item} original", original), (f"Selective replacement {fraction:.2%}", proposed)])
    if pairs:
        save_sheet(output / "filter_examples.png", pairs)
    publish_csv(output / "test_review.csv",
                ("selection", "status", "id", "impulse_fraction", "filter_active", "visual_band",
                 "fine_detail", "archived_v6_prediction", "probability_v6_prediction",
                 "v6_confidence", "v6_margin", "notes"), cases)
