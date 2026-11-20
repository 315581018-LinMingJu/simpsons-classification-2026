"""單卡訓練、配對評估與可恢復的 V8 紀錄。"""

import random
import subprocess
import sys
import time

import numpy as np
from PIL import __version__ as pillow_version

from classification_v2 import save_checkpoint, transforms_v2, write_json
from v3.inference import Views, infer_loader
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v4.analysis import paired_effect
from v6.composite import Targets
from v8.data import CONDITIONS, HARD_WEIGHTS, EvaluationImages, TrainingImages
from v8.models import SOURCES, configure, make_model, official_weight, permit_weights

FORMAT = "simpsons-v8-v1"


def version_info():
    import torch
    import torchvision

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "commit": git.stdout.strip(), "python": sys.version, "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__, "pillow": pillow_version,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


def init_cuda(seed):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("請先確認 GPU 可用，再用 CUDA_VISIBLE_DEVICES 指定一張 RTX 5090。")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_loader(dataset, batch_size, workers, shuffle_seed=None):
    import torch

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
        prefetch_factor=1 if workers else None, shuffle=shuffle_seed is not None,
        generator=torch.Generator().manual_seed(shuffle_seed if shuffle_seed is not None else 0),
    )


def evaluate(model, root, cases, seed, batch_size, workers, method, denoise, conditions=CONDITIONS):
    import torch

    view = Views(method, transforms_v2()[1])
    model.eval()
    predictions, timings = {}, {}
    expected = [case.item for case in cases]
    for condition in conditions:
        batches = data_loader(EvaluationImages(root, cases, condition, view, seed, denoise),
                              batch_size, workers)
        torch.cuda.synchronize()
        start = time.perf_counter()
        labels, ids, confidences = [], [], []
        for batch in infer_loader(model, batches):
            for item, pred in batch:
                ids.append(item)
                labels.append(pred.label)
                confidences.append(pred.confidence)
        if ids != expected:
            raise ValueError("Validation IDs changed during inference")
        predictions[condition] = np.asarray(labels)
        timings[condition] = {"seconds": time.perf_counter() - start,
                              "mean_confidence": float(np.mean(confidences))}
    return predictions, timings


def report_metrics(classes, labels, predictions):
    hits = {condition: values == labels for condition, values in predictions.items()}
    correct = {condition: float(hit.mean()) for condition, hit in hits.items()}
    return {
        "accuracy": correct,
        "hard_accuracy": (sum(correct[key] * weight for key, weight in HARD_WEIGHTS.items())
                          if HARD_WEIGHTS.keys() <= correct.keys() else None),
        "per_class": {
            name: {"samples": int((labels == index).sum()),
                   **{condition: float(hit[labels == index].mean()) for condition, hit in hits.items()}}
            for index, name in enumerate(classes)
        },
    }


def comparison(labels, candidate, reference, seed):
    effects = {condition: ((candidate[condition] == labels).astype(int) -
                           (reference[condition] == labels).astype(int)) for condition in CONDITIONS}
    hard = sum(effects[key] * weight for key, weight in HARD_WEIGHTS.items())
    effect = paired_effect(hard, labels, seed)
    clean = float(100 * effects["clean"].mean())
    by_condition = {key: float(100 * delta.mean()) for key, delta in effects.items()}
    worst = min(by_condition[key] for key in HARD_WEIGHTS if key != "negative")
    return {
        "hard_effect": effect, "accuracy_gain_pp": by_condition,
        "worth_partial_on_screen": effect["pp"] >= -3 and clean >= -2,
        "passes_confirmation_threshold": (
            effect["pp"] >= 2 and effect["ci95_pp"][0] > 0 and clean >= -1 and worst >= -8
        ),
    }


def labels_for(cases):
    return np.asarray([case.label for case in cases])


def read_checkpoint(path, inputs, allow_imagenet):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("format") != FORMAT or ckpt.get("architecture") not in SOURCES
            or ckpt["classes"] != inputs["classes"] or ckpt["split_sha256"] != inputs["split_sha256"]):
        raise ValueError(f"V8 checkpoint has incompatible architecture, classes or split: {path}")
    origin = ckpt["origin"]
    if origin != {"kind": "scratch"} and origin != imagenet_origin(ckpt["architecture"]):
        raise ValueError(f"V8 checkpoint origin is unrecognized: {path}")
    permit_weights(origin, allow_imagenet)
    return ckpt


def imagenet_origin(architecture):
    source = SOURCES[architecture]
    return {"kind": "imagenet1k", "weight": source.enum, "url": source.url, "sha256": source.sha256}


def accumulation_size(left, batch_size, accum):
    return min(left, batch_size * accum)


def train_epoch(model, modules, optimizer, scaler, dataset, args, epoch_seed):
    import torch

    model.eval()
    for module in modules:
        module.train()
    batches = data_loader(dataset, args.batch_size, args.workers, epoch_seed)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer.zero_grad(set_to_none=True)
    samples = len(dataset)
    count, total = 0, 0.0
    for step, (images, labels) in enumerate(batches):
        if step % args.accum == 0:
            group_size = accumulation_size(samples - count, args.batch_size, args.accum)
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16 if args.amp == "fp16" else torch.bfloat16,
                            enabled=args.amp != "off"):
            raw_loss = criterion(model(images), labels)
        scaler.scale(raw_loss / group_size).backward()
        count += len(labels)
        total += raw_loss.detach().item()
        if not np.isfinite(total):
            raise ValueError("Non-finite loss. Try --amp bf16 or --amp off and reduce --lr.")
        if (step + 1) % args.accum == 0 or count == samples:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return total / samples


def run_training(args, report, training, validation):
    import torch

    inputs = report["inputs"]
    seed = inputs["seed"]
    root = args.data / "train/train/train"
    directory = args.output / "models" / args.tag
    last, best = directory / "last.pt", directory / "best.pt"
    config = {
        "benchmark_sha256": report["manifest_sha256"], "train_image_signature": inputs["train_image_signature"],
        "targets": inputs["targets"], "seed": seed, "mode": args.mode,
        "augmentation": args.augmentation, "lr": args.lr, "weight_decay": 0.0001, "amp": args.amp,
        "batch_size": args.batch_size, "accum": args.accum, "workers": args.workers,
    }
    parent_path = last if args.resume else args.from_checkpoint
    parent = read_checkpoint(parent_path, inputs, args.allow_imagenet) if parent_path else None
    if parent and parent["architecture"] != args.architecture:
        raise ValueError("Parent checkpoint has a different architecture")
    if args.resume and parent["config"] != config:
        raise ValueError("Resume requires matching settings. Use a new --tag and --from-checkpoint to change them.")
    origin = parent["origin"] if parent else (
        imagenet_origin(args.architecture) if args.init == "imagenet" else {"kind": "scratch"}
    )
    permit_weights(origin, args.allow_imagenet)
    if not parent and args.init == "scratch" and args.mode != "full":
        raise ValueError("Randomly initialized models require --mode full")
    init_cuda(seed)
    env = version_info()
    if args.resume and any(parent["environment"][key] != env[key]
                           for key in ("commit", "torch", "torchvision", "cuda", "cudnn", "gpu", "pillow", "numpy")):
        raise ValueError("Resume requires the original code and runtime. Use --from-checkpoint for a new run.")
    pretrained = None
    if not parent and origin["kind"] == "imagenet1k":
        pretrained = official_weight(args.architecture, args.output / "weights", args.weights_file)
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    model = make_model(args.architecture, len(inputs["classes"]), pretrained)
    if parent is not None:
        model.load_state_dict(parent["model"], strict=True)
    model.cuda()
    modules, optimizer = configure(model, args.architecture, args.mode, args.lr, config["weight_decay"])
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16. Choose --amp fp16 or --amp off.")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    history, best_epoch = [], 0
    if args.resume:
        optimizer.load_state_dict(parent["optimizer"])
        scaler.load_state_dict(parent["scaler"])
        history, best_epoch = parent["history"], parent["best_epoch"]
        if best_epoch == len(history):
            save_checkpoint(best, {key: value for key, value in parent.items() if key not in ("optimizer", "scaler")})
    lineage = parent["parent_sha256"] if args.resume else (file_sha256(parent_path) if parent_path else None)
    settings = {
        "format": FORMAT, "config": config, "origin": origin, "parent_sha256": lineage,
        "architecture": args.architecture, "classes": inputs["classes"], "split_sha256": inputs["split_sha256"],
        "environment": env, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "evaluation": {"method": "full_resize", "conditions": CONDITIONS, "precision": "fp32"},
    }
    if not args.resume:
        publish(directory / "settings.json", json_bytes(settings))
    if len(history) > args.epochs:
        raise ValueError("--epochs is the total target and cannot be below the saved epoch")
    if history:
        write_json(directory / "history.json", history)
    training_tf = transforms_v2()[0]
    targets = Targets(**inputs["targets"])
    screen = [case for case in validation if case.fold == "screen"]
    for epoch in range(len(history) + 1, args.epochs + 1):
        epoch_seed = seed + epoch
        init_cuda(epoch_seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        dataset = TrainingImages(root, training, training_tf, targets, args.augmentation, epoch_seed)
        loss = train_epoch(model, modules, optimizer, scaler, dataset, args, epoch_seed)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        predictions, timings = evaluate(model, root, screen, seed, args.batch_size, args.workers,
                                        "full_resize", False)
        metrics = report_metrics(inputs["classes"], labels_for(screen), predictions)
        current = {
            "epoch": epoch, "loss": loss, "train_seconds": train_seconds, "evaluation": timings,
            "clean_accuracy": metrics["accuracy"]["clean"], "hard_accuracy": metrics["hard_accuracy"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        if not history or current["hard_accuracy"] > history[best_epoch - 1]["hard_accuracy"]:
            best_epoch = epoch
        history.append(current)
        checkpoint = {
            **settings, "epoch": epoch, "model": model.state_dict(), "metrics": metrics,
            "history": history, "best_epoch": best_epoch,
        }
        save_checkpoint(last, {**checkpoint, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
        if best_epoch == epoch:
            save_checkpoint(best, checkpoint)
        write_json(directory / "history.json", history)
        print(f"Epoch {epoch} | loss {loss:.4f} | Clean {current['clean_accuracy']:.2%} | "
              f"Hard {current['hard_accuracy']:.2%} | {train_seconds:.1f}s | "
              f"peak {current['peak_reserved_gib']:.2f} GiB", flush=True)
    print(f"最佳權重 {best}（epoch {best_epoch}）", flush=True)


def result_paths(directory):
    return directory / "summary.json", directory / "predictions.csv"


def save_evaluation(directory, cases, classes, predictions, timings, settings):
    labels = labels_for(cases)
    scores = report_metrics(classes, labels, predictions)
    path, csv_path = result_paths(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = settings["conditions"]
    publish_csv(csv_path, ("image", "label", *conditions), (
        (case.item, case.label, *(predictions[key][i] for key in conditions))
        for i, case in enumerate(cases)
    ))
    publish(path, json_bytes({**settings, **scores, "samples": len(cases), "timings": timings,
                              "predictions_sha256": file_sha256(csv_path)}))
    return scores


def load_evaluation(directory, cases, expected, classes):
    import csv

    from v2.data_checks import read_json

    path, csv_path = result_paths(directory)
    result = read_json(path)
    if (any(result.get(key) != value for key, value in expected.items())
            or result["predictions_sha256"] != file_sha256(csv_path)):
        raise ValueError(f"Validation results have incompatible inputs: {directory}")
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        conditions = expected["conditions"]
        if next(reader) != ["image", "label", *conditions]:
            raise ValueError("Validation prediction columns changed")
        rows = list(reader)
    if [(row[0], int(row[1])) for row in rows] != [(c.item, c.label) for c in cases]:
        raise ValueError("Validation prediction IDs or labels changed")
    predictions = np.asarray([row[2:] for row in rows], dtype=int)
    if predictions.shape != (len(cases), len(conditions)) or np.any(
        (predictions < 0) | (predictions >= len(classes))
    ):
        raise ValueError("Validation predictions use invalid classes")
    return dict(zip(conditions, predictions.T, strict=True)), result
