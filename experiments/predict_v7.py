"""僅重算 Review 已驗證的 1,728 張觸發圖，其餘沿用封存 V6。"""

import argparse
import csv
import math
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from classification_v2 import create_model, load_checkpoint, transforms_v2
from classification_v5 import METHOD
from diagnose_v7 import base_image, scan, score, sources
from ensemble.data import probability_cache
from v2.data_checks import read_json
from v3.inference import Prediction, Views, infer_view_batches
from v3.reports import read_reference, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v7.impulse import MIN_TRIGGER_FRACTION, clean_impulses
from v7.review import cached_v6

EXPECTED_ACTIVE = 1728
LEDGER_COLUMNS = ("id", "impulse_fraction", "filter_active", "visual_band", "combo_support")
PROTOCOL = "v7_selective_impulse_candidate_v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v6-run", type=Path, default=Path("runs/v6_composite/experiment"))
    parser.add_argument("--v6-submission", type=Path)
    parser.add_argument("--v7-run", type=Path, default=Path("runs/v7_diagnostics"))
    parser.add_argument("--v6-probabilities", type=Path, default=Path("runs/v6_ensemble/test/v6.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/v7_candidate"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size >= 1 and workers >= 0 are required")
    out = args.output.resolve()
    protected = (args.data, args.v2_run, args.v6_run, args.v7_run)
    if (out.parent != Path("runs").resolve() or not out.name.startswith("v7_candidate")
            or any(out.is_relative_to(path.resolve()) for path in protected)):
        parser.error("--output must be an independent runs/v7_candidate... directory")
    args.v6_submission = args.v6_submission or args.v6_run / "submission_v6.csv"
    return args


def trigger_ledger(path, ids, expected_active):
    active = {}
    saved_ids = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        if tuple(next(reader, ())) != LEDGER_COLUMNS:
            raise ValueError("Unexpected or duplicated fields in V7 trigger ledger")
        for row in reader:
            if len(row) != len(LEDGER_COLUMNS):
                raise ValueError("Invalid field count in V7 trigger ledger")
            item, value, flag, band, support = row
            fraction = float(value)
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError(f"Invalid trigger fraction: {item}")
            if flag != str(fraction >= MIN_TRIGGER_FRACTION):
                raise ValueError(f"Trigger flag disagrees with the fixed threshold: {item}")
            saved_ids.append(item)
            if flag == "True":
                active[item] = (fraction, band, support)
    if not all(ids) or len(ids) != len(set(ids)) or saved_ids != ids:
        raise ValueError("V7 trigger ledger must cover each Test ID in original template order")
    if len(active) != expected_active:
        raise ValueError(f"Expected exactly {expected_active} triggered images; found {len(active)}")
    return active


def verified_review(run, ids, info, expected_active):
    review_path = run / "review/review.json"
    validation_path = run / "validation/validation_report.json"
    ledger_path = run / "review/test_impulse_signals.csv"
    review, validation = read_json(review_path), read_json(validation_path)
    ledger_sha = file_sha256(ledger_path)
    if any(review["provenance"].get(key) != item or validation["provenance"].get(key) != item
           for key, item in info.items()):
        raise ValueError("V7 Review/Validation do not match current Test, checkpoint and V6 CSV")
    reviewed_sha = review.get("trigger_ledger_sha256")
    if (review["total_test"] != len(ids) or review["inspected"] != len(ids)
            or review["groups"]["all:all"]["activated"] != expected_active
            or (reviewed_sha is not None and reviewed_sha != ledger_sha)
            or review["trigger_fraction"] != MIN_TRIGGER_FRACTION
            or validation["trigger_fraction"] != MIN_TRIGGER_FRACTION):
        raise ValueError("V7 Review coverage, ledger or trigger threshold is inconsistent")
    conditions = validation["conditions"]
    if (validation["decision"] != "supports_labeled_impulse_followup"
            or validation["method"] != METHOD
            or not conditions["textured_impulse"]["macro_gain"]["ci95_pp"][0] > 0
            or not all(conditions[key]["macro_gain"]["pp"] >= -0.5
                       for key in ("clean", "texture_control"))):
        raise ValueError("Completed V7 labeled validation does not support a candidate")
    active = trigger_ledger(ledger_path, ids, expected_active)
    return active, {
        "review_sha256": file_sha256(review_path), "validation_sha256": file_sha256(validation_path),
        "trigger_ledger_sha256": ledger_sha, "review_has_ledger_hash": reviewed_sha is not None,
    }


class TriggeredImages:
    def __init__(self, samples, active, transform, filtered):
        self.samples, self.active, self.transform, self.filtered = samples, active, transform, filtered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        image = base_image(path, item, "test", None, None)
        if self.filtered:
            image, fraction = clean_impulses(image)
            if fraction != self.active[item][0] or fraction < MIN_TRIGGER_FRACTION:
                raise ValueError(f"Active image no longer matches the verified mask fraction: {item}")
        return self.transform(image), item


def paired_prediction_arrays(args, classes, checkpoint, split, evaluation, active, test, cached):
    import torch

    samples = [(path, item) for path, item in test if item in active]
    ids = [item for _, item in samples]
    transform = Views(METHOD, evaluation)
    model = None

    def get_batches(filtered):
        nonlocal model
        if not filtered and cached is not None:
            yield ids, cached[None, ...], None, 0.0
            return
        if model is None:
            model = create_model().eval()
            load_checkpoint(model, checkpoint, classes, split)
        dataset = TriggeredImages(samples, active, transform, filtered)
        batches = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
            prefetch_factor=1 if args.workers else None,
        )
        yield from infer_view_batches(model, batches)

    raw = probability_cache(args.output / "raw_v6_active.npz", ids, classes,
                            partial(get_batches, False))
    filtered = probability_cache(args.output / "filtered_v7_active.npz", ids, classes,
                                 partial(get_batches, True))
    return raw, filtered


def confidence_summary(raw, filtered, indices):
    if not indices:
        return {"images": 0}
    result = {"images": len(indices)}
    for i, name in ((1, "top1_confidence"), (2, "top1_top2_margin")):
        before = np.array([raw[j][i] for j in indices])
        after = np.array([filtered[j][i] for j in indices])
        result[name] = {
            "raw_mean": float(before.mean()), "filtered_mean": float(after.mean()),
            "raw_median": float(np.median(before)), "filtered_median": float(np.median(after)),
            "mean_change": float((after - before).mean()),
        }
    return result


def write_candidate(output, ids, classes, archived, active, raw_probs, filtered_probs, raw_source):
    raw, filtered = ([score(row) for row in probabilities]
                     for probabilities in (raw_probs, filtered_probs))
    active_ids = list(active)
    by_id = dict(zip(active_ids, filtered, strict=True))
    ordered = {item: by_id[item][0] if item in by_id else archived[item] for item in ids}
    changes = [item for item in ids if ordered[item] != archived[item]]
    if not set(changes) <= active.keys():
        raise ValueError("A non-triggered image changed from the original V6 CSV")
    transitions = Counter((classes[archived[item]], classes[ordered[item]]) for item in changes)
    same, different, replay_drift, filter_changes = [], [], [], []
    details = []
    for i, (item, before, after) in enumerate(zip(active_ids, raw, filtered, strict=True)):
        fraction, band, support = active[item]
        changed = archived[item] != after[0]
        (different if changed else same).append(i)
        if archived[item] != before[0]:
            replay_drift.append(item)
        if before[0] != after[0]:
            filter_changes.append(item)
        details.append((item, fraction, band, support, classes[archived[item]], raw_source,
                        classes[before[0]], before[1], before[2], classes[after[0]], after[1], after[2],
                        changed, before[0] != after[0], archived[item] != before[0]))

    publish_csv(output / "active_predictions.csv",
                ("id", "impulse_fraction", "visual_band", "combo_support", "archived_v6_label",
                 "raw_probability_source", "raw_v6_label", "raw_v6_confidence", "raw_v6_margin",
                 "filtered_v7_label", "filtered_v7_confidence", "filtered_v7_margin",
                 "changed_vs_archived_v6", "changed_vs_raw_v6", "raw_v6_differs_from_archive"), details)
    publish_csv(output / "role_transitions.csv", ("archived_v6_label", "filtered_v7_label", "images"),
                ((src, dst, n) for (src, dst), n in transitions.most_common()))
    destination = output / "submission_v7.csv"
    predictions = {item: Prediction(label, 0, 0, 0) for item, label in ordered.items()}
    submission(destination, ids, predictions, classes)
    if read_reference(destination, ids, classes) != ordered:
        raise ValueError("Candidate CSV did not preserve the validated ID order and labels")
    summary = {
        "images": len(ids), "triggered": len(active), "copied_untriggered_from_v6": len(ids) - len(active),
        "changed_vs_archived_v6": len(changes), "changed_pct": 100 * len(changes) / len(ids),
        "changed_pct_of_triggered": 100 * len(changes) / len(active),
        "raw_probability_source": raw_source,
        "raw_v6_differs_from_archive": len(replay_drift), "raw_v6_archive_mismatch_ids": replay_drift,
        "filter_changed_vs_raw_v6": len(filter_changes),
        "confidence_and_margin": {
            "all_triggered": confidence_summary(raw, filtered, list(range(len(active)))),
            "changed_vs_archive": confidence_summary(raw, filtered, different),
            "unchanged_vs_archive": confidence_summary(raw, filtered, same),
        },
        "role_transitions": [{"archived_v6": src, "filtered_v7": dst, "images": n}
                             for (src, dst), n in transitions.most_common()],
        "submission_sha256": file_sha256(destination),
    }
    publish(output / "candidate_summary.json", json_bytes(summary))
    return summary


def main():
    args = arguments()
    classes, checkpoint, v6, test, ids, archived, info = sources(args)
    active, verified = verified_review(args.v7_run, ids, info, EXPECTED_ACTIVE)
    selected_paths = [path for path, item in test if item in active]
    print(f"重新核對 {len(active)} 張觸發圖的實際遮罩比例……", flush=True)
    values = scan(selected_paths, args.workers)
    if any(fraction != active[item][0] for item, fraction in zip(active, values, strict=True)):
        raise ValueError("At least one triggered image no longer matches the reviewed mask")
    all_cached, cache_state = cached_v6(
        args.v6_probabilities, ids, classes, info["v6_checkpoint_sha256"],
        info["test_image_signature"], args.v6_submission,
    )
    cached = None if all_cached is None else all_cached[[i for i, item in enumerate(ids) if item in active]]
    raw_source = "active_raw_replay" if cached is None else "verified_full_v6_probability_cache"

    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("Use one confirmed available CUDA GPU for inference")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, evaluation = transforms_v2()
    if v6["eval_transform"] != repr(evaluation):
        raise ValueError("Current evaluation transform differs from V6")
    args.output.mkdir(parents=True, exist_ok=True)
    publish(args.output / "settings.json", json_bytes({
        **info, **verified, "protocol": PROTOCOL, "classes": classes, "method": METHOD,
        "filter_code_sha256": file_sha256(Path(__file__).resolve().parent / "v7/impulse.py"),
        "inference_code_sha256": file_sha256(Path(__file__).resolve().parent / "v3/inference.py"),
        "expected_active": EXPECTED_ACTIVE, "trigger_fraction": MIN_TRIGGER_FRACTION,
        "raw_probability_source": raw_source, "full_v6_probability_cache_status": cache_state,
        "v6_probability_cache_sha256": None if cached is None else file_sha256(args.v6_probabilities),
        "batch_size": args.batch_size, "workers": args.workers, "eval_transform": repr(evaluation),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "pillow": pillow_version, "numpy": np.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }))
    raw, filtered = paired_prediction_arrays(
        args, classes, checkpoint, info["split_sha256"], evaluation, active, test, cached,
    )
    if (image_signature(path for path, _ in test) != info["test_image_signature"]
            or file_sha256(args.v6_submission) != info["v6_submission_sha256"]
            or file_sha256(checkpoint) != info["v6_checkpoint_sha256"]):
        raise ValueError("Test images, archived predictions or checkpoint changed during inference")
    summary = write_candidate(args.output, ids, classes, archived, active, raw, filtered, raw_source)
    print(f"{args.output / 'submission_v7.csv'} | 對原 V6 改判 "
          f"{summary['changed_vs_archived_v6']}/{summary['images']} ({summary['changed_pct']:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
