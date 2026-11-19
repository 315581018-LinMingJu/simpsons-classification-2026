"""沿用 V8 切分，建立近重複隔離與已知同源的診斷組。"""

import hashlib
import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from v4.geometry import seeded
from v5.challenge import panel
from v8.data import open_rgb, stress

REFERENCES_PER_CLASS = 40
POSITIVE_PER_CLASS = 2
VAL_PER_CLASS = 4
MIN_VAL_PER_CLASS = 3
HASH_LIMIT = 12
FOLDS = ("screen", "confirm")
CONDITIONS = ("context", "distractor")
POPCOUNT = np.array([n.bit_count() for n in range(256)], dtype=np.uint8)


def difference_hash(image):
    # 兩個方向共 128 位元。先去外圍黑底，再降低 JPEG 紋理的影響。
    cutout = panel(image)
    matte = Image.new("L", cutout.size, 128)
    matte.paste(cutout.convert("L"), mask=cutout.getchannel("A"))

    def fingerprint(gray):
        horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        vertical = np.asarray(gray.resize((8, 9), Image.Resampling.LANCZOS))
        return np.packbits(np.concatenate(((horizontal[:, 1:] > horizontal[:, :-1]).ravel(),
                                           (vertical[1:, :] > vertical[:-1, :]).ravel())))

    return np.stack([fingerprint(gray) for gray in (matte, matte.transpose(Image.Transpose.FLIP_LEFT_RIGHT))])


def fingerprint_file(path):
    image = open_rgb(path)
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    image.thumbnail((96, 96), Image.Resampling.LANCZOS)
    return digest, difference_hash(image)


def fingerprint_files(paths, workers):
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            return list(pool.imap(fingerprint_file, paths, chunksize=64))
    return list(map(fingerprint_file, paths))


def near_hash(probe, bank):
    if not len(bank):
        return False
    return any(np.any(POPCOUNT[bank ^ code].sum(axis=1, dtype=np.uint16) <= HASH_LIMIT) for code in probe)


def make_plan(training, validation, root, seed, workers):
    # 比對 V8 manifest 的所有訓練圖，避免只比對小索引而宣稱來源隔離。
    fingerprints = fingerprint_files([root / case.item for case in training + validation], workers)
    train_fp = {case.item: fp for case, fp in zip(training, fingerprints[:len(training)], strict=True)}
    train_hash = np.stack([fp[1][0] for fp in train_fp.values()])
    train_exact = {fp[0] for fp in train_fp.values()}
    val_fp = {case.item: fp for case, fp in zip(validation, fingerprints[len(training):], strict=True)}
    by_label = defaultdict(list)
    for case in training:
        by_label[case.label].append(case)

    indexed, queries, accepted, accepted_exact = [], [], [], set()
    counts = defaultdict(int)
    for label, group in sorted(by_label.items()):
        order = seeded(seed, f"v11:index:{label}").sample(sorted(group, key=lambda c: c.item), len(group))
        used, unique = set(), []
        for case in order:
            signature, _ = train_fp[case.item]
            if signature in used:
                continue
            used.add(signature)
            unique.append(case)
            if len(unique) == REFERENCES_PER_CLASS:
                break
        for case in unique:
            indexed.append({"item": case.item, "label": label})
        positives, positive_hashes = [], []
        for case in unique:
            _, twins = train_fp[case.item]
            if not near_hash(twins, np.asarray(positive_hashes)):
                positives.append(case)
                positive_hashes.append(twins[0])
            if len(positives) == 2 * POSITIVE_PER_CLASS:
                break
        for index, case in enumerate(positives):
            fold = FOLDS[index // POSITIVE_PER_CLASS]
            condition = CONDITIONS[index % len(CONDITIONS)]
            others = sorted(key for key in by_label if key != label)
            rng = seeded(seed, f"v11:distractor:{case.item}")
            other = rng.choice(by_label[rng.choice(others)]).item
            queries.append({"item": case.item, "label": label, "fold": fold, "role": "positive",
                            "condition": condition, "other": other})

    for fold in FOLDS:
        for label in sorted(by_label):
            pool = sorted((c for c in validation if c.fold == fold and c.label == label), key=lambda c: c.item)
            rng = seeded(seed, f"v11:val:{fold}:{label}")
            rng.shuffle(pool)
            taken = 0
            for case in pool:
                digest, twins = val_fp[case.item]
                if digest in train_exact or digest in accepted_exact:
                    counts["exact_excluded"] += 1
                    continue
                if near_hash(twins, train_hash) or near_hash(twins, np.asarray(accepted)):
                    counts["near_hash_excluded"] += 1
                    continue
                accepted.append(twins[0])
                accepted_exact.add(digest)
                queries.append({"item": case.item, "label": label, "fold": fold, "role": "validation",
                                "condition": CONDITIONS[taken % len(CONDITIONS)], "other": case.distractor})
                taken += 1
                if taken == VAL_PER_CLASS:
                    break
    return indexed, queries, dict(counts)


def render_query(root, row, seed):
    image = open_rgb(root / row["item"])
    other = open_rgb(root / row["other"]) if row["condition"] == "distractor" else None
    return stress(image, row["condition"], seed, row["item"], other)


def plan_counts(indexed, queries, nclasses):
    by_role = {f"{fold}_{role}": [sum(row["fold"] == fold and row["role"] == role and row["label"] == label
                                     for row in queries) for label in range(nclasses)]
               for fold in FOLDS for role in ("positive", "validation")}
    index_per_class = np.bincount([row["label"] for row in indexed], minlength=nclasses)
    ready = index_per_class.min() >= 4 and all(
        min(counts) >= (MIN_VAL_PER_CLASS if name.endswith("validation") else POSITIVE_PER_CLASS)
        for name, counts in by_role.items()
    )
    return ready, {key: sum(value) for key, value in by_role.items()}, by_role


def validate_plan(indexed, queries, training, validation):
    train = {case.item: case.label for case in training}
    val = {case.item: (case.label, case.fold, case.distractor) for case in validation}
    refs = {row["item"]: row["label"] for row in indexed}
    if len(refs) != len(indexed) or any(train.get(item) != label for item, label in refs.items()):
        raise ValueError("V11 reference index contains an invalid Train item")
    seen = set()
    for row in queries:
        item, label, fold, other = (row[key] for key in ("item", "label", "fold", "other"))
        if (
            item in seen or fold not in FOLDS or row["condition"] not in CONDITIONS
            or other not in train or train[other] == label
            or row["role"] not in ("positive", "validation")
            or (row["role"] == "positive" and refs.get(item) != label)
            or (row["role"] == "validation" and val.get(item) != (label, fold, other))
        ):
            raise ValueError("V11 query does not match its original Train or Validation manifest")
        seen.add(item)
