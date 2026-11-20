"""V8 單卡強骨幹篩選、獨立驗證與候選推論。"""

import argparse
import fcntl
import re
from pathlib import Path

from classification_v2 import create_model, load_checkpoint, transforms_v2
from compare_v1_v2 import load_comparison_split
from v2.data_checks import read_json
from v3.inference import Views, infer_loader
from v3.reports import read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish
from v7.impulse import clean_impulses
from v8.data import CONDITIONS, Case, benchmark, open_rgb, prepare, selected_signature
from v8.experiment import (comparison, data_loader, evaluate, init_cuda, labels_for,
                           load_evaluation, read_checkpoint, run_training, save_evaluation, version_info)
from v8.models import SOURCES, make_model


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "train", "validate", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v8_strong_model"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    p.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    p.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    p.add_argument("--v6-run", type=Path, default=Path("runs/v6_composite/experiment"))
    p.add_argument("--audit", type=Path, default=Path("runs/distribution_audit/summary.json"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-per-class", type=int, default=160, help="0 uses all original Train images")
    p.add_argument("--per-class", type=int, default=16)
    p.add_argument("--tag", help="New V8 model or report name")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--architecture", choices=tuple(SOURCES), default="convnext_tiny")
    p.add_argument("--init", choices=("imagenet", "scratch"), default="scratch")
    p.add_argument("--allow-imagenet", action="store_true",
                   help="Confirm assignment and competition permission for official ImageNet-1K weights")
    p.add_argument("--weights-file", type=Path, help="Optional official local weight with the exact SHA-256")
    p.add_argument("--from-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mode", choices=("head", "partial", "full"), default="head")
    p.add_argument("--augmentation", choices=("basic", "context", "robust"), default="robust")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4, help="Head LR; trainable backbone LR is 0.1x")
    p.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--fold", choices=("screen", "confirm", "original"), default="screen")
    p.add_argument("--method", choices=("full_resize", "multiscale"), default="full_resize")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--compare-tag", help="Compare an existing V8 validation report on the same fold")
    args = p.parse_args(argv)
    out = args.output.resolve()
    protected = (args.data, args.v1_run, args.v2_run, args.v5_run, args.v6_run)
    if (out.parent != Path("runs").resolve() or not out.name.startswith("v8_")
            or any(out.is_relative_to(path.resolve()) for path in protected)):
        p.error("--output must be an independent runs/v8_... directory")
    if not 0 <= args.seed < 2**32 - 1000 or args.train_per_class < 0 or args.per_class < 4 or args.per_class % 4:
        p.error("Use a nonnegative seed and train limit, and --per-class a multiple of 4")
    if args.stage == "train":
        if args.resume and (args.from_checkpoint or args.weights_file or args.init == "imagenet"):
            p.error("--resume uses the latest checkpoint from --tag")
        if args.weights_file and (args.init != "imagenet" or args.from_checkpoint):
            p.error("--weights-file requires a fresh --init imagenet run")
        if args.batch_size < 1 or args.accum < 1 or not 1 <= args.epochs <= 999 or not 0 < args.lr < float("inf"):
            p.error("Use positive batch size, accumulation, epoch count and learning rate")
    if args.batch_size < 1 or args.workers < 0:
        p.error("Use positive batch size and nonnegative workers")
    if args.stage != "prepare" and (not args.tag or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag)
                                    or args.tag == "v7_reference"):
        p.error("Provide a V8 --tag with letters, digits, hyphens or underscores")
    if args.compare_tag and (args.stage != "validate" or args.fold == "original"
                             or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.compare_tag)
                             or args.compare_tag in (args.tag, "v7_reference")):
        p.error("--compare-tag requires another V8 report on screen or confirm")
    if args.stage in ("validate", "predict") and args.checkpoint is None:
        p.error("Provide --checkpoint for the V8 model")
    return args


def verified_reference(args, inputs):
    settings = read_json(args.v6_run / "settings.json")
    validation = read_json(args.v6_run / "validation.json")
    path = args.v6_run / "v6.pt"
    digest = file_sha256(path)
    if (settings["classes"] != inputs["classes"] or settings["split_sha256"] != inputs["split_sha256"]
            or settings["method"] != "multiscale"
            or settings["eval_transform"] != repr(transforms_v2()[1])
            or validation["arms"]["v6"]["checkpoint_sha256"] != digest):
        raise ValueError("V6 reference does not match its recorded checkpoint and V7 preprocessing")
    return path, digest


def validation_run(args, report, validation):
    import gc

    import torch

    ckpt_hash = file_sha256(args.checkpoint)
    inputs = report["inputs"]
    seed, classes = inputs["seed"], inputs["classes"]
    checkpoint = read_checkpoint(args.checkpoint, inputs, args.allow_imagenet)
    reference, ref_hash = verified_reference(args, inputs)
    init_cuda(seed)
    env = version_info()
    root = args.data / "train/train/train"
    conditions = list(CONDITIONS)
    if args.fold == "original":
        mapped, rows, split = load_comparison_split(root, args.v1_run, args.v2_run)
        if mapped != classes or split != inputs["split_sha256"]:
            raise ValueError("Original validation split differs from the V8 benchmark")
        cases = [Case(row.path.relative_to(root).as_posix(), row.label, args.fold) for row in rows if row.part == "val"]
        conditions = ["clean"]
        manifest_hash = split
    else:
        cases = [case for case in validation if case.fold == args.fold]
        manifest_hash = report["manifest_sha256"]["validation"]
    common = {"fold": args.fold, "seed": seed, "manifest_sha256": manifest_hash, "conditions": conditions,
              "image_signature": selected_signature(root, cases),
              "environment": env, "batch_size": args.batch_size, "workers": args.workers, "precision": "fp32"}
    parent = args.output / "reports" / args.fold
    reference_dir = parent / "v7_reference"
    reference_settings = {**common, "checkpoint_sha256": ref_hash, "method": "multiscale", "denoise": True}
    if not (reference_dir / "summary.json").exists():
        model = create_model()
        load_checkpoint(model, reference, classes, inputs["split_sha256"])
        torch.cuda.reset_peak_memory_stats()
        predictions, timings = evaluate(model, root, cases, seed, args.batch_size, args.workers,
                                        "multiscale", True, conditions)
        save_evaluation(reference_dir, cases, classes, predictions, timings, {
            **reference_settings, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": reference.stat().st_size,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    ref_predictions, ref_result = load_evaluation(reference_dir, cases, reference_settings, classes)
    directory = parent / args.tag
    settings = {**common, "checkpoint_sha256": ckpt_hash, "method": args.method, "denoise": args.denoise}
    if not (directory / "summary.json").exists():
        model = make_model(checkpoint["architecture"], len(classes))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.cuda()
        torch.cuda.reset_peak_memory_stats()
        predictions, timings = evaluate(model, root, cases, seed, args.batch_size, args.workers,
                                        args.method, args.denoise, conditions)
        settings_extra = {"architecture": checkpoint["architecture"], "origin": checkpoint["origin"],
                          "parameters": checkpoint["parameters"], "checkpoint_bytes": args.checkpoint.stat().st_size,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        save_evaluation(directory, cases, classes, predictions, timings, {**settings, **settings_extra})
    predictions, result = load_evaluation(directory, cases, settings, classes)
    decision = (comparison(labels_for(cases), predictions, ref_predictions, seed) if args.fold != "original"
                else {"clean_gain_pp": 100 * (result["accuracy"]["clean"] - ref_result["accuracy"]["clean"])})
    summary = {
        **decision, "fold": args.fold, "tag": args.tag, "checkpoint_sha256": ckpt_hash,
        "v7_checkpoint_sha256": ref_hash, "v8_clean": result["accuracy"]["clean"],
        "v7_clean": ref_result["accuracy"]["clean"], "v8_hard": result["hard_accuracy"],
        "v7_hard": ref_result["hard_accuracy"], "peak_reserved_gib": result["peak_reserved_gib"],
        "inference_seconds": sum(x["seconds"] for x in result["timings"].values()),
    }
    publish(directory / "comparison.json", json_bytes(summary))
    print(json_bytes(summary).decode(), end="", flush=True)
    if args.compare_tag:
        other, other_result = load_evaluation(parent / args.compare_tag, cases, common, classes)
        ablation = {**comparison(labels_for(cases), predictions, other, seed),
                    "tag": args.tag, "compare_tag": args.compare_tag, "fold": args.fold,
                    "checkpoint_sha256": ckpt_hash, "other_checkpoint_sha256": other_result["checkpoint_sha256"]}
        publish(directory / f"versus_{args.compare_tag}.json", json_bytes(ablation))
        print(json_bytes(ablation).decode(), end="", flush=True)


class TestImages:
    def __init__(self, samples, views, denoise):
        self.samples, self.views, self.denoise = samples, views, denoise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        image = open_rgb(path)
        if self.denoise:
            image, _ = clean_impulses(image)
        return self.views(image), item


def predict(args, report):
    inputs = report["inputs"]
    checkpoint = read_checkpoint(args.checkpoint, inputs, args.allow_imagenet)
    destination = args.output / "candidates" / args.tag
    if destination.exists():
        raise FileExistsError(destination)
    template = args.data / "sample_submission.csv"
    samples = read_test_images(args.data / "test_images/test_images", template)
    init_cuda(inputs["seed"])
    device = version_info()
    checkpoint_hash = file_sha256(args.checkpoint)
    test_signature = image_signature(path for path, _ in samples)
    template_hash = file_sha256(template)
    model = make_model(checkpoint["architecture"], len(inputs["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    views = Views(args.method, transforms_v2()[1])
    batches = data_loader(TestImages(samples, views, args.denoise), args.batch_size, args.workers)
    predictions = {}
    for batch in infer_loader(model, batches):
        predictions.update(batch)
    ids = [item for _, item in samples]
    if list(predictions) != ids or test_signature != image_signature(path for path, _ in samples):
        raise ValueError("Test order or image snapshot changed during inference")
    if file_sha256(template) != template_hash or file_sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Template or checkpoint changed during prediction")
    destination.mkdir(parents=True, exist_ok=False)
    csv = destination / "submission_v8.csv"
    submission(csv, ids, predictions, inputs["classes"])
    publish(destination / "settings.json", json_bytes({
        "architecture": checkpoint["architecture"], "origin": checkpoint["origin"], "environment": device,
        "checkpoint_sha256": checkpoint_hash, "method": args.method, "denoise": args.denoise,
        "template_sha256": template_hash, "test_image_signature": test_signature,
        "submission_sha256": file_sha256(csv), "images": len(ids),
        "batch_size": args.batch_size, "workers": args.workers,
    }))
    print(f"候選檔 {csv}", flush=True)


def main():
    args = arguments()
    if args.stage == "prepare":
        prepare(args)
        return
    report, training, validation = benchmark(args)
    if args.stage == "train":
        models = args.output / "models"
        models.mkdir(parents=True, exist_ok=True)
        with (models / (args.tag + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("這個 V8 tag 已有訓練程序。")
            run_training(args, report, training, validation)
    elif args.stage == "validate":
        validation_run(args, report, validation)
    else:
        predict(args, report)


if __name__ == "__main__":
    main()
