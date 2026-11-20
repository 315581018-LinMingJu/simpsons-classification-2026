"""V14 成對全量延續訓練與手動提交候選。"""

import argparse
import fcntl
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from classification_v2 import save_checkpoint, transforms_v2, write_json
from classification_v8 import TestImages
from classification_v9 import sources as test_sources
from classification_v11 import sources as v8_sources
from classification_v12 import reference_checkpoint
from v2.data_checks import read_json
from v3.inference import Views, infer_loader
from v3.reports import read_reference, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v6.composite import Targets
from v8.data import CONDITIONS, EvaluationImages, selected_signature
from v8.experiment import data_loader, init_cuda, labels_for, report_metrics
from v8.models import configure, make_model
from v10.experiment import subset
from v12.model import environment, gpu_probe, permitted
from v14.training import Budget, BudgetExhausted, ContinuationImages, choose_epoch, early_stop, learning_rate, selection, train_one

FORMAT = "simpsons-v14-continuation-v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "train", "predict"))
    p.add_argument("--arm", choices=("a", "b"))
    p.add_argument("--output", type=Path, default=Path("runs/v14_continuation"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--epochs", type=int, choices=(3, 4, 5), default=4)
    p.add_argument("--max-hours", type=float, default=4)
    p.add_argument("--batch-size", type=int, choices=(4, 8, 16), default=16)
    p.add_argument("--eval-batch-size", type=int, choices=(4, 8, 16, 32), default=8)
    p.add_argument("--predict-batch-size", type=int, choices=(4, 8, 16, 32))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--reuse-v8-optimizer", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epoch", type=int, choices=range(1, 6))
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    out, root = args.output.resolve(), Path("runs").resolve() / "v14_continuation"
    if (root.is_symlink() or not out.is_relative_to(root) or any(
        out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out) for path in (args.data, args.v8_run)
    )):
        p.error("--output must be independent under runs/v14_continuation")
    if args.workers < 0 or not 0 < args.max_hours <= 8:
        p.error("Use nonnegative workers and a positive budget of at most 8 hours per arm")
    if args.stage != "prepare" and args.arm is None:
        p.error("--arm a or --arm b is required")
    if args.resume and args.stage != "train" or args.epoch is not None and args.stage != "predict":
        p.error("--resume belongs to train and --epoch belongs to predict")
    return args


def print_json(value):
    print(json_bytes(value).decode(), end="", flush=True)


def inspect_history(best, last):
    import torch

    epoch, history = best["epoch"], best["history"]
    if (type(epoch) is not int or epoch < 1 or epoch != best["best_epoch"]
            or [row["epoch"] for row in history] != list(range(1, epoch + 1))):
        raise ValueError("V8 best checkpoint history is inconsistent")
    info = {"best_epoch": epoch, "history": history,
            "best_has_optimizer": bool(best.get("optimizer", {}).get("state")), "last_epoch": None,
            "last_has_optimizer": False, "last_matches_best": False}
    if last is not None:
        info["last_epoch"] = last["epoch"]
        info["last_has_optimizer"] = bool(last.get("optimizer", {}).get("state"))
        info["last_matches_best"] = (
            all(last.get(key) == best[key] for key in ("epoch", "config", "architecture", "classes", "split_sha256",
                                                        "origin", "history", "parent_sha256"))
            and last["model"].keys() == best["model"].keys()
            and all(value.dtype == last["model"][name].dtype and torch.equal(value, last["model"][name])
                    for name, value in best["model"].items())
        )
    return info


def v8_audit(info, parent):
    import torch

    path = info.checkpoint.with_name("last.pt")
    last = torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None
    report = inspect_history(parent, last)
    history_path = path.with_name("history.json")
    archived = read_json(history_path) if history_path.is_file() else None
    report["history_json_matches_best_prefix"] = (
        archived[:parent["epoch"]] == parent["history"] if archived is not None else None
    )
    if archived is not None:
        report["history_json_sha256"] = file_sha256(history_path)
    report["v8_config"] = parent["config"]
    report["v8_recorded_evaluation"] = parent["evaluation"]
    report["last_sha256"] = file_sha256(path) if last is not None else None
    return report, last


def context(args):
    info = v8_sources(args)
    info.screen = subset([c for c in info.validation if c.fold == "screen"], 4,
                         info.inputs["seed"], "v14:validation")
    if len(info.screen) != len(info.inputs["classes"]) * 4:
        raise ValueError("V14 requires four fixed V8 Screen images per class")
    root = Path(__file__).resolve().parent
    info.identity = {
        **info.identity, "protocol": FORMAT, "screen_signature": selected_signature(info.root, info.screen),
        "continuation_code_sha256": {name: file_sha256(root / name) for name in (
            "classification_v14.py", "v14/training.py", "v10/augment.py", "classification_v12.py")},
    }
    return info


def prepare(args, info, parent):
    path = args.output / "prepare.json"
    if path.is_file():
        previous = prepared(args, info)
        print_json({key: previous[key] for key in ("v8", "optimizer_policy", "epochs", "train_images", "screen_images")})
        return
    audit, _ = v8_audit(info, parent)
    reuse = args.reuse_v8_optimizer
    possible = audit["best_has_optimizer"] or audit["last_matches_best"] and audit["last_has_optimizer"]
    if reuse and not possible:
        raise ValueError("V8 optimizer state cannot be proven to match best.pt. Run prepare with the fresh optimizer policy.")
    record = {
        "identity": info.identity, "v8": audit, "epochs": args.epochs, "max_hours_per_arm": args.max_hours,
        "eval_batch_size": args.eval_batch_size, "optimizer_policy": (
            "reuse_best" if audit["best_has_optimizer"] else "reuse_exact_last"
        ) if reuse else "fresh_adamw_from_v8_model",
        "train_images": len(info.training), "screen_images": len(info.screen),
        "screen_ids": [case.item for case in info.screen], "test_template_used_for_labels": False,
    }
    publish(path, json_bytes(record))
    print_json({key: record[key] for key in ("v8", "optimizer_policy", "epochs", "train_images", "screen_images")})


def prepared(args, info):
    path = args.output / "prepare.json"
    if not path.is_file():
        raise SystemExit("先執行 V14 prepare。")
    record = read_json(path)
    if record["identity"] != info.identity:
        raise ValueError("V14 preparation no longer matches V8 sources")
    return record


def read_epoch(path, identity, index):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("identity") != identity or checkpoint.get("epoch") != index
            or checkpoint["training"].get("epoch") != index):
        raise ValueError(f"V14 incompatible continuation checkpoint: {path}")
    return checkpoint


def read_history(folder, identity):
    paths = sorted(folder.glob("epoch_*.pt"))
    if [p.name for p in paths] != [f"epoch_{i:02d}.pt" for i in range(1, len(paths) + 1)]:
        raise ValueError("V14 epoch checkpoints are not contiguous")
    rows = []
    for index, path in enumerate(paths, 1):
        saved = read_epoch(path, identity, index)
        rows.append({**saved["training"], "checkpoint_sha256": file_sha256(path)})
    journal = folder / "history.json"
    if journal.is_file():
        recorded = read_json(journal)
        if recorded != rows[:len(recorded)]:
            raise ValueError("V14 recorded epoch checkpoint or history changed")
    return rows


def evaluate(model, info, record, args, budget):
    import torch

    model.eval()
    views = Views("multiscale", transforms_v2()[1])
    result = {}
    expected = [case.item for case in info.screen]
    for condition in CONDITIONS:
        dataset = EvaluationImages(info.root, info.screen, condition, views, info.inputs["seed"], True)
        loader = data_loader(dataset, record["eval_batch_size"], args.workers)
        rows = []
        for batch in infer_loader(model, loader, warmup=False):
            budget.check()
            rows.extend(batch)
        if [item for item, _ in rows] != expected:
            raise ValueError("V14 labeled validation order changed")
        result[condition] = np.array([pred.label for _, pred in rows])
    torch.cuda.synchronize()
    return result


def reference(args, folder, model, info, record, budget):
    path = folder / "v8_screen.json"
    key = {"v8_checkpoint_sha256": info.identity["checkpoint_sha256"],
           "prepare_sha256": file_sha256(args.output / "prepare.json")}
    if path.is_file():
        saved = read_json(path)
        if saved["identity"] != key:
            raise ValueError("V14 cached V8 reference differs")
        return {name: np.asarray(values) for name, values in saved["predictions"].items()}
    pred = evaluate(model, info, record, args, budget)
    publish(path, json_bytes({"identity": key, "predictions": {name: values.tolist() for name, values in pred.items()}}))
    return pred


def train(args, info, parent, record, folder, identity):
    import torch

    settings = folder / "settings.json"
    if (settings.is_file()) != args.resume:
        raise SystemExit("首次執行使用 train，已開始的訓練使用 train --resume。")
    if args.resume and read_json(settings) != identity:
        raise ValueError("Runtime or source changed since training began")
    if not args.resume:
        publish(settings, json_bytes(identity))
    history = read_history(folder, identity)
    budget = Budget(folder / "budget.json", record["max_hours_per_arm"] * 3600)
    reason = early_stop(history)
    try:
        if not reason and len(history) < record["epochs"]:
            budget.check()
            seed = info.inputs["seed"]
            init_cuda(seed)
            model = make_model("convnext_tiny", len(info.inputs["classes"]))
            model.load_state_dict(parent["model"], strict=True)
            model.cuda()
            _, optimizer = configure(model, "convnext_tiny", "full", learning_rate(1, record["epochs"]), 1e-4)
            baseline = reference(args, folder, model, info, record, budget)
            labels = labels_for(info.screen)
            classes = info.inputs["classes"]
            ref_metrics = report_metrics(classes, labels, baseline)
            if history:
                previous = read_epoch(folder / f"epoch_{len(history):02d}.pt", identity, len(history))
                if file_sha256(folder / f"epoch_{len(history):02d}.pt") != history[-1]["checkpoint_sha256"]:
                    raise ValueError("V14 resume checkpoint changed")
                model.load_state_dict(previous["model"], strict=True)
                optimizer.load_state_dict(previous["optimizer"])
                del previous
            elif record["optimizer_policy"] == "reuse_best":
                optimizer.load_state_dict(parent["optimizer"])
            elif record["optimizer_policy"] == "reuse_exact_last":
                v8_last = info.checkpoint.with_name("last.pt")
                if file_sha256(v8_last) != record["v8"]["last_sha256"]:
                    raise ValueError("Original V8 last.pt changed after prepare")
                last = torch.load(v8_last, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(last["optimizer"])
                del last

            basic, evaluation = transforms_v2()
            targets = Targets(**info.inputs["targets"])
            for epoch in range(len(history) + 1, record["epochs"] + 1):
                budget.check()
                epoch_seed = seed + epoch
                init_cuda(epoch_seed)
                lr = learning_rate(epoch, record["epochs"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                dataset = ContinuationImages(info.root, info.training, basic, evaluation, targets,
                                             parent["config"]["augmentation"], epoch_seed, args.arm, epoch)
                print(f"V14 {args.arm.upper()} epoch {epoch} | {len(dataset)} images | LR {lr:.2g}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                stats = train_one(model, optimizer, dataset, args, epoch_seed, budget)
                torch.cuda.synchronize()
                train_seconds = time.perf_counter() - started
                predictions = evaluate(model, info, record, args, budget)
                metrics = report_metrics(classes, labels, predictions)
                row = {
                    **stats, "epoch": epoch, "lr": lr, "batch_size": args.batch_size, "workers": args.workers,
                    "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - started,
                    "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30, "metrics": metrics,
                    "selection": selection(metrics, ref_metrics),
                    "changed_vs_v8_by_condition": {name: int((predictions[name] != baseline[name]).sum())
                                                  for name in CONDITIONS},
                }
                path = folder / f"epoch_{epoch:02d}.pt"
                if file_sha256(info.checkpoint) != info.identity["checkpoint_sha256"]:
                    raise ValueError("Original V8 checkpoint changed")
                save_checkpoint(path, {"identity": identity, "epoch": epoch, "training": row,
                                       "model": model.state_dict(), "optimizer": optimizer.state_dict()})
                history.append({**row, "checkpoint_sha256": file_sha256(path)})
                write_json(folder / "history.json", history)
                budget.save()
                print_json({key: row[key] for key in ("epoch", "loss", "selection", "epoch_seconds", "peak_vram_gib")})
                reason = early_stop(history)
                if reason:
                    break
    except BudgetExhausted:
        reason = "budget_exhausted_at_last_completed_epoch"
    finally:
        budget.save()
    write_json(folder / "history.json", history)
    best = choose_epoch(history)
    summary = {"decision": reason or "epoch_budget_complete", "arm": args.arm, "epochs_completed": len(history),
               "active_seconds": budget.elapsed, "selected": best}
    write_json(folder / "train_summary.json", summary)
    selected = None if best is None else {
        "epoch": best["epoch"], "loss": best["loss"], "clean": best["metrics"]["accuracy"]["clean"],
        "hard": best["metrics"]["hard_accuracy"], "epoch_seconds": best["epoch_seconds"],
        "peak_vram_gib": best["peak_vram_gib"], "checkpoint_sha256": best["checkpoint_sha256"],
    }
    print_json({**summary, "selected": selected})


def test_comparison(ids, old, new, classes):
    transitions = Counter((old[item], new[item]) for item in ids if old[item] != new[item])
    before, after = Counter(old.values()), Counter(new.values())
    rows = []
    for i, name in enumerate(classes):
        removed = sum(n for (source, _), n in transitions.items() if source == i)
        added = sum(n for (_, target), n in transitions.items() if target == i)
        rows.append((name, before[i], after[i], removed, added, after[i] - before[i]))
    return rows, [(classes[i], classes[j], n) for (i, j), n in sorted(transitions.items())], sum(transitions.values())


def predict(args, info, folder, identity):
    history = read_history(folder, identity)
    chosen = choose_epoch(history, args.epoch)
    if chosen is None:
        raise SystemExit("指定的 epoch 未通過嚴重退步護欄，或尚無合格 checkpoint。")
    epoch = chosen["epoch"]
    path, digest = folder / f"epoch_{epoch:02d}.pt", chosen["checkpoint_sha256"]
    destination = folder / f"candidate_epoch_{epoch:02d}"
    if destination.exists():
        raise FileExistsError(f"Candidate already exists: {destination}")
    args.checkpoint = info.checkpoint
    args.v8_submission = args.v8_run / "candidates/convnext_tta_denoise/submission_v8.csv"
    test = test_sources(args)
    if test.provenance["checkpoint_sha256"] != info.identity["checkpoint_sha256"]:
        raise ValueError("Deployment reference changed")
    checkpoint = read_epoch(path, identity, epoch)
    init_cuda(info.inputs["seed"])
    model = make_model("convnext_tiny", len(info.inputs["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    batch_size = args.predict_batch_size or info.settings["batch_size"]
    samples = test.samples
    views = Views("multiscale", transforms_v2()[1])
    batches = data_loader(TestImages(samples, views, True), batch_size, args.workers)
    predictions = {}
    started = time.monotonic()
    for batch in infer_loader(model, batches):
        if time.monotonic() - started >= 45 * 60:
            raise SystemExit("候選推論已達 45 分鐘上限。")
        predictions.update(batch)
    ids = [item for _, item in samples]
    if (list(predictions) != ids or file_sha256(path) != digest
            or image_signature(p for p, _ in samples) != test.provenance["test_image_signature"]
            or file_sha256(args.data / "sample_submission.csv") != test.provenance["template_sha256"]
            or file_sha256(args.v8_submission) != info.identity["v8_submission_sha256"]):
        raise ValueError("Candidate inputs, order or checkpoint changed during inference")
    classes = info.inputs["classes"]
    with TemporaryDirectory(prefix=".candidate-", dir=folder) as temporary:
        staged = Path(temporary)
        out = staged / "submission.csv"
        submission(out, ids, predictions, classes)
        saved = read_reference(out, ids, classes)
        counts, moves, changed = test_comparison(ids, test.archived, saved, classes)
        publish_csv(staged / "class_changes.csv",
                    ("character", "v8_count", "candidate_count", "removed", "added", "net"), counts)
        publish_csv(staged / "transitions.csv", ("v8_label", "candidate_label", "count"), moves)
        result = {"arm": args.arm, "epoch": epoch, "checkpoint_sha256": digest, "submission_sha256": file_sha256(out),
                  "images": len(ids), "changed_vs_v8": changed, "changed_pct": 100 * changed / len(ids),
                  "seconds": time.monotonic() - started, "submission": str(destination / "submission.csv"),
                  "method": "multiscale", "denoise": True, "batch_size": batch_size,
                  "test_provenance": test.provenance}
        publish(staged / "summary.json", json_bytes(result))
        staged.rename(destination)
    print_json({key: result[key] for key in ("arm", "epoch", "images", "changed_vs_v8", "changed_pct",
                                           "checkpoint_sha256", "submission_sha256", "submission")})


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output if args.stage == "prepare" else args.output / args.arm
    if folder.is_symlink():
        raise ValueError("V14 arm directory must be independent")
    folder.mkdir(exist_ok=True)
    with (folder / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V14 輸出已有階段正在使用。")
        permitted(args.allow_imagenet)
        info = context(args)
        parent = reference_checkpoint(args, info)
        if args.stage == "prepare":
            prepare(args, info, parent)
            return
        record = prepared(args, info)
        env = gpu_probe()
        if env["status"] != "ready":
            print_json(env)
            raise SystemExit("V14 GPU 預檢未通過。")
        if args.stage == "train":
            identity = {"arm": args.arm, "prepare_sha256": file_sha256(args.output / "prepare.json"),
                        "environment": environment(), "protocol": FORMAT}
            train(args, info, parent, record, folder, identity)
        else:
            identity = read_json(folder / "settings.json")
            if (identity["prepare_sha256"] != file_sha256(args.output / "prepare.json")
                    or identity["arm"] != args.arm):
                raise ValueError("V14 candidate settings do not match preparation or arm")
            predict(args, info, folder, identity)


if __name__ == "__main__":
    main()
