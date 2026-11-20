"""V12 ConvNeXt-Base 單卡短跑，V8 等條件配對停損。"""

import argparse
import fcntl
import gc
import time
from pathlib import Path

from classification_v2 import save_checkpoint, transforms_v2
from classification_v8 import TestImages
from classification_v11 import sources as v8_sources
from v2.data_checks import read_json
from v3.inference import Views, infer_loader
from v3.reports import read_reference, read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v6.composite import Targets
from v8.data import TrainingImages, selected_signature
from v8.experiment import data_loader, init_cuda, read_checkpoint, train_epoch
from v8.models import head_status, make_model, verify_weight
from v10.experiment import subset
from v12.model import ORIGIN, RATES, SOURCE, configure, environment, gpu_probe, make_base, permitted, weights_path
from v12.protocol import ALL_CONDITIONS, SHIFT, comparison, evaluate_condition

FORMAT = "simpsons-v12-v1"
PER_CLASS = 240
METHOD = "multiscale"
DENOISE = True


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("probe", "prepare", "train", "confirm", "predict"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--output", type=Path, default=Path("runs/v12"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--v11-audit", type=Path)
    p.add_argument("--weights-file", type=Path, help="Local copy of the exact official ConvNeXt-Base checkpoint")
    p.add_argument("--batch-size", type=int, default=8, choices=(4, 8, 16))
    p.add_argument("--eval-batch-size", type=int, choices=(4, 8, 16, 32))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    out, root = args.output.resolve(), Path("runs").resolve() / "v12"
    protected = [args.data, args.v8_run]
    if args.v11_audit:
        protected.append(args.v11_audit.parent)
    if (root.is_symlink() or not out.is_relative_to(root) or
            any(out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in protected)):
        p.error("--output must be independent under runs/v12")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    return args


def print_json(value):
    print(json_bytes(value).decode(), end="", flush=True)


def shifted_cohort(path, info):
    if path is None:
        return info.validation
    audit = read_json(path)
    keys = ("checkpoint_sha256", "benchmark_sha256", "v8_image_signature", "validation_seed")
    if audit["decision"] != "ready" or any(audit["identity"].get(key) != info.identity[key] for key in keys):
        raise ValueError("V11 source-exclusion audit differs from the V8 inputs")
    cases = {case.item: case for case in info.validation}
    chosen = set()
    for row in audit["queries"]:
        if row["role"] != "validation":
            continue
        case = cases.get(row["item"])
        if (case is None or case.label != row["label"] or case.fold != row["fold"] or
                case.distractor != row["other"] or case.item in chosen):
            raise ValueError("V11 validation query differs from the V8 manifest")
        chosen.add(case.item)
    selected = [case for case in info.validation if case.item in chosen]
    for fold in ("screen", "confirm"):
        if {case.label for case in selected if case.fold == fold} != set(range(len(info.inputs["classes"]))):
            raise ValueError(f"V11 audit has incomplete coverage for {fold}")
    return selected


def sources(args):
    info = v8_sources(args)
    seed = info.inputs["seed"]
    info.training = subset(info.training, PER_CLASS, seed, "v12:train")
    default = Path("runs/v11/audit.json")
    audit = args.v11_audit or (default if default.is_file() else None)
    info.shift = shifted_cohort(audit, info)
    info.targets = Targets(**info.inputs["targets"])
    root = Path(__file__).resolve().parent
    source_files = ("classification_v12.py", "v12/model.py", "v12/protocol.py", "v6/composite.py",
                    "v5/challenge.py", "v5/background.py", "v10/experiment.py")
    info.identity = {
        **info.identity, "protocol": FORMAT, "weight_origin": ORIGIN, "train_per_class": PER_CLASS,
        "v12_train_signature": selected_signature(info.root, info.training),
        "shift_cohort": "v11_near_duplicate_screened" if audit else "original_v8_validation",
        "v11_audit_sha256": file_sha256(audit) if audit else None,
        "v12_code_sha256": {name: file_sha256(root / name) for name in source_files},
    }
    args.eval_batch_size = args.eval_batch_size or info.settings["batch_size"]
    return info


def prepare(args, info):
    path = args.output / "prepare.json"
    if path.is_file():
        result = read_json(path)
        if result["identity"] != info.identity or file_sha256(args.output / "train.csv") != result["train_sha256"]:
            raise ValueError("Prepared sources and manifests do not match")
    else:
        train_path = args.output / "train.csv"
        publish_csv(train_path, ("image", "label"), ((case.item, case.label) for case in info.training))
        result = {
            "identity": info.identity, "train_sha256": file_sha256(train_path),
            "weights": {"source": SOURCE.url, "sha256": SOURCE.sha256, "mib": SOURCE.mib,
                        "available": {"local_verified": bool(verify_weight(args.weights_file, SOURCE))}
                        if args.weights_file else head_status(SOURCE.url)},
            "competition_permission": "requires_operator_confirmation",
            "train_images": len(info.training),
            "shift_cohort": info.identity["shift_cohort"],
            "counts": {fold: {"original": sum(case.fold == fold for case in info.validation),
                             "heldout": sum(case.fold == fold for case in info.shift)}
                       for fold in ("screen", "confirm")},
        }
        publish(path, json_bytes(result))
    print_json({key: result[key] for key in ("train_images", "shift_cohort", "counts", "weights")})


def prepared(args, info):
    path = args.output / "prepare.json"
    if not path.is_file():
        raise SystemExit("先執行 V12 prepare。")
    record = read_json(path)
    if record["identity"] != info.identity or record["train_sha256"] != file_sha256(args.output / "train.csv"):
        raise ValueError("V12 training or validation sources changed")
    return file_sha256(path)


def reference_checkpoint(args, info):
    parent = read_checkpoint(info.checkpoint, info.inputs, args.allow_imagenet)
    config = parent["config"]
    if (parent["architecture"] != "convnext_tiny" or config["mode"] != "full"
            or parent["origin"] != info.settings["origin"]
            or config["benchmark_sha256"] != info.identity["manifest_sha256"]
            or config["targets"] != info.inputs["targets"] or config["augmentation"] not in {"basic", "context", "robust"}):
        raise ValueError("V8 archive, model configuration or preprocessing differs from the benchmark")
    return parent


def read_base(path, settings):
    import torch

    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint.get("settings") != settings:
        raise ValueError(f"V12 checkpoint settings changed: {path}")
    return checkpoint


def stage_settings(args, info, parent, phase, prepare_hash, load_hash=None):
    return {
        "identity": info.identity, "prepare_sha256": prepare_hash, "phase": phase, "parent_sha256": load_hash,
        "seed": info.inputs["seed"] + (1 if phase == "head" else 2), "epochs": 1,
        "amp": "bf16", "batch_size": args.batch_size, "accum": 64 // args.batch_size, "workers": args.workers,
        "lr": list(RATES[phase]), "augmentation": parent["config"]["augmentation"], "weight_decay": 1e-4,
        "environment": environment(),
    }


def check_snapshot(info):
    if (file_sha256(info.checkpoint) != info.identity["checkpoint_sha256"]
            or selected_signature(info.root, info.training) != info.identity["v12_train_signature"]):
        raise ValueError("The V8 weights or V12 training images changed mid-run")


def train_phase(args, info, settings, load=None):
    import torch

    folder = args.output / "checkpoints"
    folder.mkdir(exist_ok=True)
    phase = settings["phase"]
    path = folder / f"{phase}.pt"
    if path.exists():
        old = read_base(path, settings)
        return path, old["training"]
    init_cuda(settings["seed"])
    model = make_base(len(info.inputs["classes"]), weights_path(args.output / "weights", args.weights_file)
                      if load is None else None)
    if load is not None:
        model.load_state_dict(load["model"], strict=True)
    model.cuda()
    modules, optimizer = configure(model, phase)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    dataset = TrainingImages(info.root, info.training, transforms_v2()[0], info.targets,
                             settings["augmentation"], settings["seed"])
    args.amp, args.accum = settings["amp"], settings["accum"]
    print(f"V12 {phase} | 1 epoch | {len(dataset)} images", flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = train_epoch(model, modules, optimizer, scaler, dataset, args, settings["seed"])
    torch.cuda.synchronize()
    training = {"loss": loss, "images": len(dataset), "seconds": time.perf_counter() - start,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    check_snapshot(info)
    save_checkpoint(path, {"settings": settings, "model": model.state_dict(), "training": training})
    return path, training


def evaluate_arm(args, info, cases, settings, arm, checkpoint, digest):
    import torch

    phase = settings["phase"]
    # Head 已評估的 V8 Screen Clean／heldout 可供最後 Screen 續用。
    fold = "confirm" if phase == "confirm" else "screen"
    result = {}
    folder = args.output / "scores" / fold / arm
    views = Views(METHOD, transforms_v2()[1])
    model = None
    for condition, cohort in cases.items():
        path = folder / condition
        metadata = {
            "identity": info.identity, "fold": fold, "arm": arm, "arm_checkpoint_sha256": digest,
            "conditions": [condition], "method": METHOD, "denoise": DENOISE, "precision": "fp32",
            "batch_size": args.eval_batch_size, "workers": args.workers, "environment": settings["environment"],
            "image_signature": selected_signature(info.root, cohort),
        }
        if model is None and not (path / "summary.json").is_file():
            model = (make_model("convnext_tiny", len(info.inputs["classes"])) if arm == "v8"
                     else make_base(len(info.inputs["classes"])))
            model.load_state_dict(checkpoint["model"], strict=True)
            model.cuda().eval()
        result[condition] = evaluate_condition(path, model, info, cohort, condition, views, args, metadata)
        if metadata["image_signature"] != selected_signature(info.root, cohort):
            raise ValueError("Validation images changed during V12 evaluation")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate(args, info, parent, current, path, phase, prepare_hash, previous_sha=None):
    digest = file_sha256(path)
    meta = {"identity": info.identity, "prepare_sha256": prepare_hash, "phase": phase,
            "v12_checkpoint_sha256": digest, "previous_report_sha256": previous_sha,
            "environment": environment(), "eval_batch_size": args.eval_batch_size, "workers": args.workers,
            "method": METHOD, "denoise": DENOISE}
    location = args.output / f"{phase}.json"
    if location.is_file():
        result = read_json(location)
        if any(result.get(key) != value for key, value in meta.items()):
            raise ValueError("V12 evaluation sources changed")
        return result
    fold = "confirm" if phase == "confirm" else "screen"
    conditions = ("clean", SHIFT) if phase == "head" else ALL_CONDITIONS
    cohorts = {name: [case for case in (info.shift if name == SHIFT else info.validation) if case.fold == fold]
               for name in conditions}
    init_cuda(info.inputs["seed"])
    before = evaluate_arm(args, info, cohorts, meta, "v8", parent, info.identity["checkpoint_sha256"])
    after = evaluate_arm(args, info, cohorts, meta, phase, current, digest)
    check_snapshot(info)
    if digest != file_sha256(path):
        raise ValueError("V12 checkpoint changed during evaluation")
    result = {**meta, **comparison(phase, before, after, cohorts, info.inputs["seed"]),
              "training": current["training"]}
    publish(location, json_bytes(result))
    return result


def compare_and_report(result):
    print_json({key: result[key] for key in ("phase", "decision", "reasons", "training",
                                           "conditions", "hard_effect", "heldout_effect") if key in result})


def expected_decision(args, phase, info, prepare_hash, path, decision):
    report_path = args.output / f"{phase}.json"
    if not report_path.is_file():
        raise SystemExit(f"先完成 V12 {phase}。")
    current = read_json(report_path)
    if (current["identity"] != info.identity or current["prepare_sha256"] != prepare_hash
            or current["v12_checkpoint_sha256"] != file_sha256(path)
            or current["phase"] != phase or current["method"] != METHOD or current["denoise"] is not DENOISE):
        raise ValueError(f"V12 {phase} provenance does not match")
    if current["decision"] != decision:
        raise SystemExit(f"V12 {phase} 未達停損門檻，保留 V8。")
    return file_sha256(report_path)


def predict(args, info, model_checkpoint, path, confirm_hash):
    template = args.data / "sample_submission.csv"
    samples = read_test_images(args.data / "test_images/test_images", template)
    snapshots = {"template_sha256": file_sha256(template), "checkpoint_sha256": file_sha256(path),
                 "test_signature": image_signature(image for image, _ in samples)}
    init_cuda(info.inputs["seed"])
    model = make_base(len(info.inputs["classes"]))
    model.load_state_dict(model_checkpoint["model"], strict=True)
    model.cuda().eval()
    batches = data_loader(TestImages(samples, Views(METHOD, transforms_v2()[1]), DENOISE),
                          args.eval_batch_size, args.workers)
    ids = [item for _, item in samples]
    pairs = [pair for batch in infer_loader(model, batches) for pair in batch]
    v8_submission = args.v8_run / "candidates/convnext_tta_denoise/submission_v8.csv"
    if (list(item for item, _ in pairs) != ids or snapshots["checkpoint_sha256"] != file_sha256(path)
            or snapshots["template_sha256"] != file_sha256(template)
            or file_sha256(v8_submission) != info.identity["v8_submission_sha256"]
            or snapshots["test_signature"] != image_signature(image for image, _ in samples)):
        raise ValueError("Test IDs, files, template or checkpoint changed during inference")
    reference = read_reference(v8_submission, ids, info.inputs["classes"])
    changed = sum(pred.label != reference[item] for item, pred in pairs)
    folder = args.output / "candidate"
    folder.mkdir(exist_ok=True)
    target = folder / "submission_v12.csv"
    if changed:
        submission(target, ids, dict(pairs), info.inputs["classes"])
    publish(folder / "settings.json", json_bytes({
        "identity": info.identity, **snapshots, "confirm_sha256": confirm_hash,
        "submission_sha256": file_sha256(target) if changed else None, "method": METHOD, "denoise": DENOISE,
        "changed_vs_v8": changed, "decision": "candidate_saved" if changed else "identical_to_v8",
        "images": len(ids), "batch_size": args.eval_batch_size, "workers": args.workers,
        "environment": environment(),
    }))
    print_json({"candidate": str(target) if changed else None, "images": len(ids), "changed_vs_v8": changed})


def run(args):
    info = sources(args)
    if args.stage == "prepare":
        prepare(args, info)
        return
    permitted(args.allow_imagenet)
    prepare_hash = prepared(args, info)
    env = gpu_probe()
    if env["status"] != "ready":
        print_json(env)
        raise SystemExit("V12 GPU 預檢未通過。")
    parent = reference_checkpoint(args, info)
    head_settings = stage_settings(args, info, parent, "head", prepare_hash)
    if args.stage == "train":
        head, _ = train_phase(args, info, head_settings)
        head_result = evaluate(args, info, parent, read_base(head, head_settings), head, "head", prepare_hash)
        compare_and_report(head_result)
        if head_result["decision"] != "advance_to_tune":
            return
    else:
        head = args.output / "checkpoints/head.pt"
    head_review = expected_decision(args, "head", info, prepare_hash, head, "advance_to_tune")
    tune_settings = stage_settings(args, info, parent, "tune", prepare_hash, file_sha256(head))
    if args.stage == "train":
        path, _ = train_phase(args, info, tune_settings, read_base(head, head_settings))
        compare_and_report(evaluate(args, info, parent, read_base(path, tune_settings), path, "screen",
                                    prepare_hash, head_review))
        return
    path = args.output / "checkpoints/tune.pt"
    screen_review = expected_decision(args, "screen", info, prepare_hash, path, "advance_to_confirm")
    checkpoint = read_base(path, tune_settings)
    if args.stage == "confirm":
        compare_and_report(evaluate(args, info, parent, checkpoint, path, "confirm", prepare_hash, screen_review))
    else:
        confirm_review = expected_decision(args, "confirm", info, prepare_hash, path, "eligible_for_candidate")
        if read_json(args.output / "confirm.json")["previous_report_sha256"] != screen_review:
            raise ValueError("Confirm does not match the accepted Screen result")
        predict(args, info, checkpoint, path, confirm_review)


def main():
    args = arguments()
    if args.stage == "probe":
        result = gpu_probe()
        print_json(result)
        if result["status"] != "ready":
            raise SystemExit(1)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V12 輸出已有執行中的階段。")
        run(args)


if __name__ == "__main__":
    main()
