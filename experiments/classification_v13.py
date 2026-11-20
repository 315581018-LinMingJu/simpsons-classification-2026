"""V13 以同資料和更新預算比較官方 ConvNeXt Tiny／Base。"""

import argparse
import fcntl
import gc
from pathlib import Path
from types import SimpleNamespace

from classification_v2 import save_checkpoint, transforms_v2
from classification_v11 import sources as v8_sources
from classification_v12 import reference_checkpoint, shifted_cohort
from v2.data_checks import read_json
from v3.inference import Views
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v6.composite import Targets
from v8.data import HARD_WEIGHTS, selected_signature
from v8.experiment import init_cuda
from v8.models import head_status, make_model, verify_weight
from v12.model import environment, gpu_probe, permitted
from v12.protocol import ALL_CONDITIONS, SHIFT, evaluate_condition
from v13.protocol import base_advantage, paired_report, screen_decision, wins_deployment
from v13.training import (ARMS, BUDGET, EFFECTIVE_BATCH, HEAD_LR, STAGE_LR, WEIGHT_DECAY, WEIGHTS,
                          BudgetExceeded, checkpoint_stats, configure_last_stage, make_arm, manifests)

FORMAT = "simpsons-v13-v1"
PT_STAGES = ("pilot", "extend")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("probe", "prepare", "smoke", *PT_STAGES, "confirm"))
    p.add_argument("--output", type=Path, default=Path("runs/v13"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--v11-audit", type=Path)
    p.add_argument("--tiny-weights", type=Path)
    p.add_argument("--base-weights", type=Path)
    p.add_argument("--batch-size", type=int, default=8, choices=(4, 8, 16))
    p.add_argument("--eval-batch-size", type=int, choices=(4, 8, 16, 32))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    out, root = args.output.resolve(), Path("runs").resolve() / "v13"
    protected = [args.data, args.v8_run]
    if args.v11_audit:
        protected.append(args.v11_audit.parent)
    if (root.is_symlink() or not out.is_relative_to(root) or any(
        out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in protected
    )):
        p.error("--output must be independent under runs/v13")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    original_weights = {
        "tiny": (args.v8_run / "weights", Path("runs/v8_strong_model/weights")),
        "base": (Path("runs/v12/weights"),),
    }
    for arm, folders in original_weights.items():
        flag = f"{arm}_weights"
        if getattr(args, flag) is None:
            files = (folder / WEIGHTS[arm].filename for folder in folders)
            setattr(args, flag, next((path for path in files if path.is_file()), None))
    return args


def print_json(payload):
    print(json_bytes(payload).decode(), end="", flush=True)


def source_info(args):
    info = v8_sources(args)
    seed = info.inputs["seed"]
    info.parts = manifests(info.training, info.inputs["classes"], seed)
    info.targets = Targets(**info.inputs["targets"])
    default = Path("runs/v11/audit.json")
    audit = args.v11_audit or (default if default.is_file() else None)
    info.shift = shifted_cohort(audit, info)
    root = Path(__file__).resolve().parent
    sources = ("classification_v13.py", "v13/training.py", "v13/protocol.py", "classification_v12.py",
               "v12/model.py", "v12/protocol.py", "v6/composite.py")
    info.identity = {
        **info.identity, "protocol": FORMAT,
        "weights": {arm: {"sha256": value.sha256, "url": value.url} for arm, value in WEIGHTS.items()},
        "training_signatures": {phase: selected_signature(info.root, cases) for phase, cases in info.parts.items()},
        "shift_cohort": "v11_near_duplicate_screened" if audit else "original_v8_validation",
        "v11_audit_sha256": file_sha256(audit) if audit else None,
        "v13_code_sha256": {name: file_sha256(root / name) for name in sources},
    }
    args.eval_batch_size = args.eval_batch_size or info.settings["batch_size"]
    return info


def prepare(args, info):
    path = args.output / "prepare.json"
    records = args.output / "train.csv"
    if path.is_file():
        result = read_json(path)
        if result["identity"] != info.identity or result["train_sha256"] != file_sha256(records):
            raise ValueError("V13 data or input artifacts changed")
    else:
        publish_csv(records, ("phase", "image", "label"),
                    ((stage, case.item, case.label) for stage, cases in info.parts.items() for case in cases))
        weights = {}
        for arm, source in WEIGHTS.items():
            local = getattr(args, f"{arm}_weights")
            weights[arm] = {"local_verified": bool(verify_weight(local, source))} if local else head_status(source.url)
        result = {
            "identity": info.identity, "train_sha256": file_sha256(records), "weights": weights,
            "budgets": {key: {"updates": pair[0], "cap_seconds_per_arm": pair[1], "images": len(info.parts[key])}
                        for key, pair in BUDGET.items()},
            "shift_cohort": info.identity["shift_cohort"],
        }
        publish(path, json_bytes(result))
    print_json({key: result[key] for key in ("weights", "budgets", "shift_cohort")})


def run_context(args, info, parent):
    path = args.output / "prepare.json"
    if not path.is_file():
        raise SystemExit("先執行 V13 prepare。")
    prep = read_json(path)
    if prep["identity"] != info.identity or prep["train_sha256"] != file_sha256(args.output / "train.csv"):
        raise ValueError("V13 input differs from prepare")
    return {
        "identity": info.identity, "prepare_sha256": file_sha256(path), "environment": environment(),
        "augmentation": parent["config"]["augmentation"], "workers": args.workers, "amp": "bf16",
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "effective_batch": EFFECTIVE_BATCH, "head_lr": HEAD_LR, "stage_lr": STAGE_LR,
        "weight_decay": WEIGHT_DECAY, "stochastic_depth": 0,
    }


def verify_snapshot(info, phase):
    if (file_sha256(info.checkpoint) != info.identity["checkpoint_sha256"] or selected_signature(
        info.root, info.parts[phase]
    ) != info.identity["training_signatures"][phase]):
        raise ValueError("V13 source checkpoint or training images changed")


def load_pt(args, phase, arm, context):
    import torch

    path = args.output / "checkpoints" / f"{phase}_{arm}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    prior_steps = BUDGET["pilot"][0] if phase == "extend" else 0
    if (checkpoint.get("context") != context or checkpoint.get("phase") != phase or checkpoint.get("arm") != arm
            or training["optimizer_steps"] != BUDGET[phase][0]
            or training["total_optimizer_steps"] != prior_steps + BUDGET[phase][0]):
        raise ValueError(f"V13 incompatible checkpoint: {path}")
    return path, checkpoint


def train_arm(args, info, context, phase, arm, previous_hash=None):
    import torch

    folder = args.output / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / f"{phase}_{arm}.pt"
    smoke_path = args.output / f"smoke_{arm}.json"
    if phase == "smoke" and smoke_path.is_file():
        result = read_json(smoke_path)
        if result["context"] != context or result["arm"] != arm or result["training"]["optimizer_steps"] != BUDGET["smoke"][0]:
            raise ValueError("V13 smoke uses different sources")
        return result["training"]
    if path.is_file():
        _, old = load_pt(args, phase, arm, context)
        if old["prior_sha256"] != previous_hash:
            raise ValueError("V13 extension lineage changed")
        return old["training"]

    seed = info.inputs["seed"] + {"smoke": 1, "pilot": 2, "extend": 3}[phase]
    init_cuda(seed)
    model = make_arm(arm, len(info.inputs["classes"]), args.output / "weights", getattr(args, f"{arm}_weights"),
                     from_pretrained=phase != "extend").cuda()
    modules, optimizer = configure_last_stage(model)
    if phase == "extend":
        parent_path, parent = load_pt(args, "pilot", arm, context)
        if file_sha256(parent_path) != previous_hash:
            raise ValueError("V13 pilot parent changed before extension")
        model.load_state_dict(parent["model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        del parent
    print(f"V13 {phase} | {arm} | {BUDGET[phase][0]} optimizer updates", flush=True)
    result = checkpoint_stats(arm, model, modules, optimizer, info.parts[phase], info, seed, args, phase)
    result["total_optimizer_steps"] = result["optimizer_steps"] + (BUDGET["pilot"][0] if phase == "extend" else 0)
    verify_snapshot(info, phase)
    if phase == "smoke":
        publish(smoke_path, json_bytes({"context": context, "arm": arm, "training": result}))
    else:
        save_checkpoint(path, {"context": context, "phase": phase, "arm": arm, "training": result,
                               "prior_sha256": previous_hash, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()})
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify_report(args, context, phase):
    path = args.output / f"{phase}.json"
    if not path.is_file():
        raise SystemExit(f"先完成 V13 {phase}。")
    result = read_json(path)
    if result["context"] != context or result["phase"] != phase:
        raise ValueError("V13 report context changed")
    checkpoint_phase = result.get("checkpoint_phase", phase)
    checksums = result.get("checkpoint_sha256", result.get("confirmed_checkpoint_sha256", {}))
    for arm, digest in checksums.items():
        ckpt = args.output / "checkpoints" / f"{checkpoint_phase}_{arm}.pt"
        if file_sha256(ckpt) != digest:
            raise ValueError("V13 report checkpoint changed")
    if phase == "confirm" and result["screen_sha256"] != file_sha256(args.output / f"{checkpoint_phase}.json"):
        raise ValueError("V13 Confirm Screen lock changed")
    return result


def report_eligible(result, expected):
    if result["decision"] != expected:
        raise SystemExit(f"V13 {result['phase']} 結果為 {result['decision']}，停止本階段。")


def train_pair(args, info, context, phase, previous=None):
    result = {}
    for arm in ARMS:
        try:
            result[arm] = train_arm(args, info, context, phase, arm,
                                    previous["checkpoint_sha256"][arm] if previous else None)
        except BudgetExceeded as exc:
            return {"decision": "stop_budget", "candidate": None, "failed_arm": arm, "seconds": exc.seconds,
                    "optimizer_steps_before_stop": exc.count, "training": result}
    return {"decision": "training_complete", "training": result}


def evaluate_scores(args, info, phase, parent, context, arms, checkpoint_phase):
    import torch

    seed = info.inputs["seed"]
    init_cuda(seed)
    fold = "confirm" if phase == "confirm" else "screen"
    cohorts = {condition: [case for case in (info.shift if condition == SHIFT else info.validation)
                           if case.fold == fold] for condition in ALL_CONDITIONS}
    views = Views("multiscale", transforms_v2()[1])
    results, digests = {}, {}
    for arm in ("v8", *arms):
        model = None
        if arm == "v8":
            digest = info.identity["checkpoint_sha256"]
        else:
            path = args.output / "checkpoints" / f"{checkpoint_phase}_{arm}.pt"
            digest = file_sha256(path)
            digests[arm] = digest
        scores = {}
        scope = "reference" if arm == "v8" else checkpoint_phase
        base = args.output / "scores" / fold / scope / arm
        for condition, cases in cohorts.items():
            dest = base / condition
            image_signature = selected_signature(info.root, cases)
            settings = {"context": context, "checkpoint_sha256": digest, "conditions": [condition],
                        "fold": fold, "method": "multiscale", "denoise": True, "precision": "fp32",
                        "image_signature": image_signature}
            if model is None and not (dest / "summary.json").is_file():
                if arm == "v8":
                    model = make_model("convnext_tiny", len(info.inputs["classes"]))
                    state = parent["model"]
                else:
                    model = make_arm(arm, len(info.inputs["classes"]), args.output / "weights")
                    _, current = load_pt(args, checkpoint_phase, arm, context)
                    state = current["model"]
                    del current
                model.load_state_dict(state, strict=True)
                del state
                model.cuda().eval()
            scores[condition] = evaluate_condition(dest, model, info, cases, condition, views, args, settings)
            if selected_signature(info.root, cases) != image_signature:
                raise ValueError("V13 validation images changed")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        results[arm] = scores
    verify_snapshot(info, checkpoint_phase)
    for arm, digest in digests.items():
        if file_sha256(args.output / "checkpoints" / f"{checkpoint_phase}_{arm}.pt") != digest:
            raise ValueError("V13 evaluation checkpoint changed")
    return results, digests, cohorts


def decisions(args, info, parent, context, training):
    phase = args.stage
    scores, digests, cohorts = evaluate_scores(args, info, phase, parent, context, ARMS, phase)
    seed = info.inputs["seed"]
    tiny = paired_report("screen", scores["v8"], scores["tiny"], cohorts, seed, "v8", "tiny")
    base = paired_report("screen", scores["v8"], scores["base"], cohorts, seed, "v8", "base")
    relative = paired_report("screen", scores["tiny"], scores["base"], cohorts, seed, "tiny", "base")
    return {**screen_decision(phase, tiny, base, relative), "training": training, "checkpoint_sha256": digests,
            "comparisons": {"tiny_vs_v8": tiny, "base_vs_v8": base, "base_vs_tiny": relative}}


def confirm(args, info, parent, context):
    previous_phase = "extend" if (args.output / "extend.json").is_file() else "pilot"
    previous = verify_report(args, context, previous_phase)
    report_eligible(previous, "advance_to_confirm")
    candidate = previous["candidate"]
    if candidate not in ARMS:
        raise ValueError("V13 candidate not recorded")
    arms = ARMS if candidate == "base" else ("tiny",)
    scores, digests, cohorts = evaluate_scores(args, info, "confirm", parent, context, arms, previous_phase)
    seed = info.inputs["seed"]
    report = paired_report("confirm", scores["v8"], scores[candidate], cohorts, seed, "v8", candidate)
    comparisons = {f"{candidate}_vs_v8": report}
    reasons = list(report["reasons"])
    if candidate == "base":
        relative = paired_report("confirm", scores["tiny"], scores["base"], cohorts, seed, "tiny", "base")
        comparisons["base_vs_tiny"] = relative
        if not base_advantage(relative):
            reasons.append("base_advantage_not_confirmed")
    if not wins_deployment(report) and not reasons:
        reasons.append("deployment_advantage_not_confirmed")
    return {"decision": "confirmed_advantage" if not reasons else "stop_confirm", "candidate": candidate,
            "reasons": reasons, "screen_sha256": file_sha256(args.output / f"{previous_phase}.json"),
            "checkpoint_phase": previous_phase, "confirmed_checkpoint_sha256": digests,
            "training": previous["training"], "comparisons": comparisons}


def short_report(result):
    report = {key: value for key, value in result.items() if key not in ("context", "comparisons")}
    if "comparisons" in result:
        report["comparisons"] = {
            key: {**{metric: value[metric] for metric in ("hard_effect", "heldout_effect")},
                  "clean_accuracy_pct": {arm: pct["clean"] for arm, pct in value["accuracy_pct"].items()},
                  "hard_accuracy_pct": {arm: sum(pct[name] * weight for name, weight in HARD_WEIGHTS.items())
                                        for arm, pct in value["accuracy_pct"].items()},
                  "heldout_accuracy_pct": {arm: pct[SHIFT] for arm, pct in value["accuracy_pct"].items()}}
            for key, value in result["comparisons"].items()
        }
    return report


def run(args):
    info = source_info(args)
    if args.stage == "prepare":
        prepare(args, info)
        return
    permitted(args.allow_imagenet)
    env = gpu_probe()
    if env["status"] != "ready":
        print_json(env)
        raise SystemExit("V13 GPU 預檢未通過。")
    parent = reference_checkpoint(args, info)
    info.augmentation = parent["config"]["augmentation"]
    context = run_context(args, info, parent)
    phase = args.stage
    path = args.output / f"{phase}.json"
    if path.is_file():
        print_json(short_report(verify_report(args, context, phase)))
        return
    if phase == "confirm":
        result = confirm(args, info, parent, context)
    else:
        previous = None
        if phase != "smoke":
            report_eligible(verify_report(args, context, "smoke"), "smoke_pass")
        if phase == "extend":
            previous = verify_report(args, context, "pilot")
            report_eligible(previous, "advance_to_extend")
        trained = train_pair(args, info, context, phase, previous)
        if trained["decision"] == "stop_budget":
            result = trained
        elif phase == "smoke":
            result = {**trained, "decision": "smoke_pass"}
        else:
            result = decisions(args, info, parent, context, trained["training"])
    saved = {"context": context, "phase": phase, **result}
    publish(path, json_bytes(saved))
    print_json(short_report(saved))


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
            raise SystemExit("此 V13 輸出已有執行中的階段。")
        run(args)


if __name__ == "__main__":
    main()
