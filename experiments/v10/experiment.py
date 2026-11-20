"""固定一次短微調及 V8 原條件的配對判定。"""

import io
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from classification_v2 import save_checkpoint, transforms_v2
from distribution_audit.features import FEATURES, quantiles, visual_features
from v3.storage import file_sha256, json_bytes, publish
from v4.analysis import paired_effect
from v4.geometry import seeded
from v8.data import (CONDITIONS, HARD_WEIGHTS, TrainingImages as V8TrainingImages,
                     open_rgb, selected_signature)
from v8.experiment import evaluate, init_cuda, load_evaluation, save_evaluation, train_epoch
from v8.models import configure, make_model
from v10.augment import (MIN_OFFSET_FRACTION, MIN_VISIBLE_EDGE, MIN_VISIBLE_FRACTION,
                         SCENE_PROBABILITY, SPAN_RANGE, TrainingImages, render_scene)

FORMAT = "simpsons-v10-v1"
LEARNING_RATE = 3e-5
TRAIN_PER_CLASS = 240
AUDIT_PER_CLASS = 5
MIN_AUDIT_ACCEPTANCE = 0.50
METHOD = "multiscale"
DENOISE = True
MAX_CLEAN_DROP_PP = 0.5
MAX_OTHER_DROP_PP = 4.0
HARD_MIN_PP = {"screen": 0.75, "confirm": 1.0}


def subset(cases, per_class, seed, purpose):
    groups = defaultdict(list)
    for case in cases:
        groups[case.label].append(case)
    return [case for label, group in sorted(groups.items())
            for case in seeded(seed, f"v10:{purpose}:{label}").sample(
                sorted(group, key=lambda case: case.item), min(len(group), per_class))]


def sampled_audit(root, cases, targets, seed):
    inspected = subset(cases, AUDIT_PER_CLASS, seed, "audit")
    preview = set(seeded(seed, "v10:preview").sample(range(len(inspected)), min(12, len(inspected))))
    pair_rows, rendered, originals = [], [], []
    reasons = Counter()
    groups = defaultdict(Counter)
    for i, case in enumerate(inspected):
        image = open_rgb(root / case.item)
        result = render_scene(image, targets, seed, case.item)
        reasons[result.reason] += 1
        groups[case.label][result.reason] += 1
        if result.reason == "accepted":
            rendered.append(result)
            originals.append(visual_features(image))
        if i in preview:
            pair_rows.append((image, result.image, result.reason))
    features = ("black_edge_fraction", "brightness", "contrast", "fine_detail")

    def medians(matrix):
        return {key: float(np.median(np.asarray(matrix)[:, FEATURES.index(key)])) for key in features}

    acceptance = reasons["accepted"] / len(inspected)
    report = {
        "train_images_audited": len(inspected), "reasons": dict(reasons),
        "accepted_by_class_index": {str(label): group["accepted"] for label, group in sorted(groups.items())},
        "acceptance": acceptance, "estimated_augmented_fraction": SCENE_PROBABILITY * acceptance,
        "minimum_acceptance": MIN_AUDIT_ACCEPTANCE,
        "decision": "ready" if acceptance >= MIN_AUDIT_ACCEPTANCE else "stop_low_usable_scene_rate",
        "geometry_p10_p50_p90": {
            field: quantiles([getattr(item, field) for item in rendered], (0.1, 0.5, 0.9))
            for field in ("span_fraction", "visible_fraction", "offset_fraction")
        } if rendered else {},
        "accepted_original_medians": medians(originals) if rendered else {},
        "accepted_scene_medians": medians([visual_features(item.image) for item in rendered]) if rendered else {},
    }
    return report, pair_rows


def preview(path, pairs):
    size, pad = 112, 6
    rows = (len(pairs) + 3) // 4
    result = Image.new("RGB", (pad + 4 * (2 * size + 2 * pad), pad + rows * (size + 25)), "white")
    draw = ImageDraw.Draw(result)
    for i, (before, after, reason) in enumerate(pairs):
        x, y = pad + i % 4 * (2 * size + 2 * pad), pad + i // 4 * (size + 25)
        for offset, image in ((0, before), (size + pad, after)):
            result.paste(image.resize((size, size), Image.Resampling.BILINEAR), (x + offset, y))
        draw.text((x, y + size + 2), reason, fill="black")
    out = io.BytesIO()
    result.save(out, format="PNG")
    publish(path, out.getvalue())


def gate(phase, hard, gains):
    failures = []
    if hard["pp"] + 1e-9 < HARD_MIN_PP[phase]:
        failures.append("hard_gain_below_minimum")
    if phase == "confirm" and hard["ci95_pp"][0] <= 0:
        failures.append("hard_ci_includes_no_improvement")
    if gains["clean"] < -MAX_CLEAN_DROP_PP - 1e-9:
        failures.append("clean_regression")
    if gains["context"] < -1e-9:
        failures.append("context_regression")
    if min(gains[key] for key in CONDITIONS if key not in ("clean", "context")) < -MAX_OTHER_DROP_PP - 1e-9:
        failures.append("other_condition_regression")
    return failures


def comparison(phase, labels, reference, adapted, seed):
    metrics, effects = {}, {}
    for name in CONDITIONS:
        before = reference[name] == labels
        after = adapted[name] == labels
        delta = after.astype(np.int8) - before.astype(np.int8)
        effects[name] = delta
        paired = paired_effect(delta, labels, seed)
        metrics[name] = {
            "v8_accuracy_pct": 100 * float(before.mean()),
            "v10_accuracy_pct": 100 * float(after.mean()),
            "effect": paired,
            "wins": int((delta == 1).sum()),
            "losses": int((delta == -1).sum()),
            "prediction_changes": int((reference[name] != adapted[name]).sum()),
        }
    hard = paired_effect(sum(weight * effects[key] for key, weight in HARD_WEIGHTS.items()), labels, seed)
    gains = {name: metric["effect"]["pp"] for name, metric in metrics.items()}
    failures = gate(phase, hard, gains)
    return {
        "decision": "stop" if failures else "advance_to_confirm" if phase == "screen" else "eligible_for_candidate",
        "reasons": failures, "samples_per_condition": len(labels),
        "hard_effect": hard, "condition_gain_pp": gains, "conditions": metrics,
        "v8_hard_pct": sum(weight * metrics[key]["v8_accuracy_pct"] for key, weight in HARD_WEIGHTS.items()),
        "v10_hard_pct": sum(weight * metrics[key]["v10_accuracy_pct"] for key, weight in HARD_WEIGHTS.items()),
        "thresholds": {"hard_gain_min_pp": HARD_MIN_PP[phase], "max_clean_drop_pp": MAX_CLEAN_DROP_PP,
                       "min_context_gain_pp": 0, "max_other_drop_pp": MAX_OTHER_DROP_PP,
                       "require_positive_hard_ci_lower": phase == "confirm"},
    }


def gpu_environment():
    import torch
    import torchvision

    return {
        "python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "numpy": np.__version__, "pillow": pillow_version, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


def make_loaded_model(checkpoint, classes):
    model = make_model("convnext_tiny", len(classes))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cuda().eval()


def fit(args, info, training, targets, destination, audit_hash):
    import torch
    from torchvision import transforms as T

    seed = info.provenance["training_seed"]
    init_cuda(seed)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V10 短跑需要 RTX 5090 的 BF16 支援。")
    model = make_loaded_model(info.parent, info.classes)
    modules, optimizer = configure(model, "convnext_tiny", "partial", LEARNING_RATE, 1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    basic, evaluation = transforms_v2()
    root = args.data / "train/train/train"
    base = V8TrainingImages(root, training, basic, targets, info.parent["config"]["augmentation"], seed)
    dataset = TrainingImages(root, training, base, T.Compose(evaluation.transforms[2:]), targets, seed)
    args.accum, args.amp = 64 // args.batch_size, "bf16"
    configuration = {
        "epochs": 1, "mode": "convnext_last_stage_and_head", "head_lr": LEARNING_RATE,
        "last_stage_lr": LEARNING_RATE / 10, "optimizer": "AdamW", "weight_decay": 1e-4,
        "batch_size": args.batch_size, "accumulation": args.accum, "amp": args.amp, "workers": args.workers,
        "reference_augmentation": info.parent["config"]["augmentation"], "scene_attempt_rate": SCENE_PROBABILITY,
        "span_range": list(SPAN_RANGE), "min_visible_fraction": MIN_VISIBLE_FRACTION,
        "min_visible_edge_pixels": MIN_VISIBLE_EDGE, "min_center_offset_fraction": MIN_OFFSET_FRACTION,
    }
    settings = {"provenance": info.provenance, "configuration": configuration, "environment": gpu_environment(),
                "audit_sha256": audit_hash,
                "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad)}
    publish(destination.parent / "settings.json", json_bytes(settings))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = train_epoch(model, modules, optimizer, scaler, dataset, args, seed)
    torch.cuda.synchronize()
    training_info = {"loss": loss, "seconds": time.perf_counter() - started, "images": len(training),
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    if file_sha256(info.path) != info.provenance["v8_checkpoint_sha256"]:
        raise ValueError("V8 checkpoint changed during V10 training")
    if selected_signature(root, training) != info.provenance["v10_training_signature"]:
        raise ValueError("V10 training images changed during the experiment")
    save_checkpoint(destination, {**settings, "format": FORMAT, "architecture": "convnext_tiny",
                                  "classes": info.classes, "model": model.state_dict(), "training": training_info})
    return training_info


def evaluate_arm(output, checkpoint, root, cases, classes, seed, args, metadata):
    path = output / "summary.json"
    if path.is_file():
        return load_evaluation(output, cases, metadata, classes)
    model = make_loaded_model(checkpoint, classes)
    predictions, timings = evaluate(model, root, cases, seed, args.eval_batch_size, args.workers, METHOD, DENOISE)
    save_evaluation(output, cases, classes, predictions, timings, metadata)
    return load_evaluation(output, cases, metadata, classes)


def code_fingerprint(root):
    dependencies = (
        "classification_v10.py", "v10/augment.py", "v10/experiment.py",
        "classification_v2.py", "classification_v8.py", "compare_v1_v2.py", "v2/data_checks.py",
        "v3/inference.py", "v3/reports.py", "v3/storage.py", "v4/analysis.py", "v4/geometry.py",
        "v5/background.py", "v5/challenge.py", "v6/composite.py", "v7/impulse.py",
        "v8/data.py", "v8/experiment.py", "v8/models.py", "ensemble/data.py",
        "distribution_audit/features.py",
    )
    return {name: file_sha256(Path(root) / name) for name in dependencies}
