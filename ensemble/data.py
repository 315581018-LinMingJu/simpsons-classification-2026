"""去重驗證切分、測試改判稽核，以及可續用的機率陣列。"""

import csv
import hashlib
import io
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from distribution_audit.compare import grouped_changes
from v2.data_checks import read_json
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v4.geometry import seeded

MODELS = ("v5", "v6")
FOLDS = ("calibration", "selection", "confirmation")


def fresh_validation(rows, train_root, v5_run, classes, per_class, seed):
    val = {row.path.relative_to(train_root).as_posix(): row for row in rows if row.part == "val"}
    old_settings = read_json(v5_run / "settings.json")
    exclusions = read_json(v5_run / "validation_subset.json")
    if exclusions["image_signature"] != old_settings["train_image_signature"]:
        raise ValueError("V5 exact-duplicate exclusions do not match the original training images")
    cross = set(exclusions["excluded_ids"])
    if not cross <= val.keys():
        raise ValueError("V5 exclusions refer to unknown validation IDs")
    used = set()
    with (v5_run / "validation_samples.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            item = row["relative_path"]
            if (item not in val or item in used or item in cross
                    or row["label"] != classes[val[item].label]):
                raise ValueError(f"Invalid earlier V5 validation entry: {item}")
            used.add(item)
    if len(used) != old_settings["validation_samples"]:
        raise ValueError("Earlier V5 validation manifest has a different sample count")

    groups = defaultdict(list)
    for item, row in val.items():
        if item in cross:
            continue
        with Image.open(row.path) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
            digest.update(rgb.tobytes())
        groups[digest.hexdigest()].append((item, row))

    candidates = defaultdict(list)
    removed_seen = removed_conflicts = removed_repeats = 0
    for group in groups.values():
        if any(item in used for item, _ in group):
            removed_seen += len(group)
        elif len({row.label for _, row in group}) > 1:
            removed_conflicts += len(group)
        else:
            item, row = min(group, key=lambda pair: pair[0])
            candidates[row.label].append((row.path, item))
            removed_repeats += len(group) - 1

    train_by_label = defaultdict(list)
    for row in rows:
        if row.part == "train":
            train_by_label[row.label].append(row)
    samples, labels, folds, distractors = [], [], [], {}
    for label in range(len(classes)):
        pool = sorted(candidates[label])
        sampled = seeded(seed, f"ensemble:label:{label}").sample(pool, min(len(pool), per_class))
        if len(sampled) < 4:
            raise ValueError(f"Fewer than four fresh validation images for {classes[label]}")
        first = len(sampled) // 4
        for i, (path, item) in enumerate(sampled):
            samples.append((path, item))
            labels.append(label)
            folds.append(FOLDS[min(i // first, 2)])
            rng = seeded(seed, f"ensemble:distractor:{item}")
            choice = rng.choice(sorted(key for key in train_by_label if key != label))
            distractors[item] = rng.choice(train_by_label[choice])
    return samples, np.asarray(labels, dtype=int), folds, distractors, {
        "removed_exact_train_matches": len(cross),
        "removed_earlier_v5_validation_and_exact_copies": removed_seen,
        "removed_conflicting_pixel_groups_images": removed_conflicts,
        "removed_other_duplicate_images": removed_repeats,
    }


def checked_probs(probs, expected_shape):
    if (probs.shape != expected_shape or probs.dtype != np.float32
            or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid or incompatible Multi-scale probability array")
    return probs


def probability_cache(path, expected_ids, classes, get_batches):
    expected_shape = (len(expected_ids), len(classes))
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            if cached["ids"].tolist() != expected_ids:
                raise ValueError(f"Cached IDs differ from the manifest: {path}")
            result = checked_probs(cached["probabilities"], expected_shape)
        print(f"Reuse probabilities: {path.name}", flush=True)
        return result
    ids, parts = [], []
    for batch_ids, probs, _, _ in get_batches():
        ids.extend(batch_ids)
        parts.append(probs.mean(axis=0))
    if ids != expected_ids:
        raise ValueError("Multi-scale inference did not preserve the requested IDs")
    result = checked_probs(np.concatenate(parts, axis=0), expected_shape)
    payload = io.BytesIO()
    np.savez_compressed(payload, ids=np.asarray(ids), probabilities=result)
    publish(path, payload.getvalue())
    print(f"Saved probabilities: {path.name}", flush=True)
    return result


def audit_test(output, args, expected_ids, classes, archived, split_hash):
    run = args.audit_run
    offset = read_json(run / "summary.json")
    if (offset["provenance"]["split_sha256"] != split_hash
            or offset["provenance"]["v5_submission_sha256"] != file_sha256(args.v5_submission)
            or offset["labels"]["allowed_characters"] != classes
            or offset["test"]["files"] != len(expected_ids)):
        raise ValueError("Distribution audit does not match the current V5 predictions and dataset")
    known_ids = set(expected_ids)
    with (run / "test_novelty.csv").open(encoding="utf-8-sig", newline="") as file:
        by_id = {}
        for row in csv.DictReader(file):
            item = row["id"]
            if item not in known_ids or item in by_id:
                raise ValueError(f"Unexpected or duplicate ID in distribution audit: {item}")
            by_id[item] = row
    if len(by_id) != offset["test"]["decoded_images"]:
        raise ValueError("Distribution audit novelty table has a different decoded-image count")

    v5, v6 = (archived[key] for key in MODELS)
    changes = {item: v5[item] != v6[item] for item in expected_ids}
    transitions = Counter((classes[v5[item]], classes[v6[item]]) for item in expected_ids if changes[item])
    groups = []
    for item in expected_ids:
        row = by_id.get(item, {})
        values = {key: row.get(key) or "unknown" for key in ("visual_band", "combo_support", "black_edge")}
        values["appearance_combination"] = "|".join(row.get(key) or "unknown" for key in (
            "black_edge", "weak_color", "brightness", "contrast", "fine_detail"
        ))
        groups.append((item, values))
    grouped = grouped_changes(groups, changes)
    fields = ("dimension", "group", "images", "changed", "changed_pct",
              "share_of_all_changes_pct", "rate_ratio")
    roles = [(name, sum(n for (src, _), n in transitions.items() if src == name),
              sum(n for (_, dst), n in transitions.items() if dst == name)) for name in classes]
    publish_csv(output / "v5_v6_role_changes.csv", ("role", "predictions_leaving", "predictions_entering"), roles)
    publish_csv(output / "v5_v6_transitions.csv", ("v5_prediction", "v6_prediction", "test_images"),
                ((src, dst, n) for (src, dst), n in transitions.most_common()))
    publish_csv(output / "v5_v6_by_group.csv", fields, grouped)
    publish(output / "v5_v6_test_audit.json", json_bytes({
        "test_count": len(expected_ids), "changed": sum(changes.values()),
        "changed_pct": 100 * sum(changes.values()) / len(expected_ids),
        "without_visual_group": len(expected_ids) - len(by_id),
        "prediction_sha256": {name: file_sha256(getattr(args, name + "_submission")) for name in MODELS},
        "distribution_novelty_sha256": file_sha256(run / "test_novelty.csv"),
        "top_transitions": [{"v5": src, "v6": dst, "count": n}
                            for (src, dst), n in transitions.most_common(12)],
        "visual_bands": [dict(zip(fields, row, strict=True)) for row in grouped if row[0] == "visual_band"],
    }))
