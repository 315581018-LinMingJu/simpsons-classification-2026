"""唯讀檢查 Baseline 快取，並重播指定 ID 的原始批次。"""

import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np

from classification_v2 import loader as v2_loader
from v2.data_checks import read_json
from v3.inference import view_loader
from v3.storage import inspect_baseline_cache


def locate_batch(ids, item, batch_size):
    try:
        index = ids.index(item)
    except ValueError as exc:
        raise ValueError(f"Unknown test ID: {item!r}") from exc
    start = index // batch_size * batch_size
    return start, min(start + batch_size, len(ids)), index - start


def top_scores(logits, probabilities, classes, reference_index):
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Probe produced non-finite scores")
    ranked = sorted(range(len(classes)), key=lambda rank: (-float(logits[rank]), rank))

    def score(index):
        return {"character": classes[index], "logit": float(logits[index]),
                "probability": float(probabilities[index])}

    top = [score(index) for index in ranked[:5]]
    return {
        "top5": top,
        "top1_top2_logit_margin": top[0]["logit"] - top[1]["logit"],
        "top1_top2_probability_margin": top[0]["probability"] - top[1]["probability"],
        "historical_label_current_score": score(reference_index),
    }


def read_cache(output, metadata, ids, batch_size, classes, item, reference_hash):
    path = output / "test.sqlite3"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            with conn:
                conn.execute("BEGIN")
                audit = inspect_baseline_cache(conn, metadata, ids, batch_size, classes, item)
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "audit_error", "path": str(path), "error": str(exc)}
    report = output / "baseline_check.json"
    if report.is_file():
        try:
            previous_hash = read_json(report)["settings"]["v2_reference_sha256"]
            audit["prior_baseline_check_reference_matches"] = previous_hash == reference_hash
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audit["prior_baseline_check_error"] = str(exc)
    return {"status": "inspected", "path": str(path), **audit}


def replay(model, batches, path, batch_number, expected_ids):
    import torch

    with torch.inference_mode():
        for number, (images, ids) in enumerate(batches):
            if path == "v3":
                images, = images
                torch.cuda.synchronize()
            logits = model(images.cuda(non_blocking=True))
            if number != batch_number:
                logits.argmax(dim=1).cpu()
                continue
            if list(ids) != expected_ids:
                raise ValueError(f"{path} replay batch order differs from template")
            return (
                hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
                logits.cpu().numpy(), logits.softmax(dim=1).cpu().numpy(),
            )
    raise ValueError(f"{path} did not return the target batch")


def probe(args, samples, classes, reference, config, evaluation, metadata, reference_hash, make_model):
    import torch

    ids = [item for _, item in samples]
    batch_size = config["batch_size"]
    start, stop, offset = locate_batch(ids, args.probe_id, batch_size)
    audit = read_cache(args.output, metadata, ids, batch_size, classes, args.probe_id, reference_hash)
    replay_samples = samples[:stop] if args.replay_prefix else samples[start:stop]
    batch_number = start // batch_size if args.replay_prefix else 0
    batch_ids = ids[start:stop]

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = make_model()

    settings = {
        "batch_size": batch_size, "batch_number": start // batch_size + 1,
        "batch_start_index_zero_based": start, "position_in_batch_zero_based": offset,
        "batch_ids": batch_ids, "actual_batch_size": stop - start,
        "dataset_last_batch_size": len(ids) % batch_size or batch_size,
        "replayed_prefix": args.replay_prefix, "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "v2_workers": config.get("workers", 4), "v3_workers": args.workers,
    }
    result = {
        "id": args.probe_id, "historical_v2_label": classes[reference[args.probe_id]],
        "historical_v2_scores_available": False, "v2_reference_sha256": reference_hash,
        "expected_cache_metadata": metadata, "cache": audit, "replay_settings": settings,
        "replays": {},
    }
    measured = {}
    for path in ("v2", "v3") if args.probe_path == "both" else (args.probe_path,):
        if path == "v2":
            batches = v2_loader(replay_samples, evaluation, config, settings["v2_workers"])
        else:
            batches = view_loader(replay_samples, "baseline", evaluation, batch_size, args.workers)
        input_hash, logits, probabilities = replay(model, batches, path, batch_number, batch_ids)
        measured[path] = (input_hash, logits)
        labels = logits.argmax(axis=1).tolist()
        cached = audit.get("baseline_target")
        target = top_scores(logits[offset], probabilities[offset], classes, reference[args.probe_id])
        result["replays"][path] = {
            **target, "input_tensor_sha256": input_hash, "logits_dtype": str(logits.dtype),
            "top1": classes[labels[offset]],
            "batch_reference_mismatch_ids": [
                item for item, label in zip(batch_ids, labels, strict=True) if label != reference[item]
            ],
            "target_matches_cached_baseline": None if cached is None else labels[offset] == cached["label"],
        }

    if len(measured) == 2:
        v2_hash, v2_logits = measured["v2"]
        v3_hash, v3_logits = measured["v3"]
        result["replay_comparison"] = {
            "batch_input_bitwise_equal": v2_hash == v3_hash,
            "batch_logits_bitwise_equal": v2_logits.tobytes() == v3_logits.tobytes(),
            "batch_logit_max_abs_diff": float(np.abs(v2_logits - v3_logits).max()),
            "target_logit_max_abs_diff": float(np.abs(v2_logits[offset] - v3_logits[offset]).max()),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
