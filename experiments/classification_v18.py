"""V18：以 V16 高信心測試集偽標籤，從 V14-A Epoch 3 延續微調 ConvNeXt-Tiny。

stage
  prepare  CPU：核對 A3／V16 來源，從 V16 已封存的 A、B 測試機率重建 A80/B20 融合，
           先確認融合 Top-1 與 submission_v16.csv 逐筆一致，再選出高信心偽標籤。
  train    GPU：Train 全量（V8 原配方）＋偽標籤測試圖，從 A3 權重延續訓練。
           加上 --smoke 只跑少量批次確認流程，不寫入任何 checkpoint。
  predict  GPU：指定 epoch 的十視角推論，輸出 submission_v18.csv 與改判統計。
"""

import argparse
import fcntl
import math
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

V16_DEFAULT = Path("runs/v16/archive-fix-49dc35247fa67babe7ccac7c1ce12136dd780ade/candidate")
FORMAT = "simpsons-v18-pseudo-label-v1"
WEIGHT_A = .8
MIN_LR = 5e-6


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "train", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v18"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--v14-run", type=Path, default=Path("runs/v14_continuation"))
    p.add_argument("--archive-a3", type=Path)
    p.add_argument("--v16-candidate", type=Path, default=V16_DEFAULT)
    p.add_argument("--threshold", type=float, default=.90, help="V16 融合機率門檻（prepare）")
    p.add_argument("--allow-disagree", action="store_true", help="不要求 A3 與 B4 Top-1 一致（prepare）")
    p.add_argument("--pseudo-repeat", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--epochs", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, choices=(4, 8, 16), default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epoch", type=int, choices=(1, 2, 3), help="predict 使用的 epoch，預設最後完成的 epoch")
    p.add_argument("--smoke", action="store_true", help="train 只跑少量批次檢查流程，不保存")
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    root = Path("runs").resolve() / "v18"
    out = args.output.resolve()
    if root.is_symlink() or not out.is_relative_to(root):
        p.error("--output must be under runs/v18")
    if not .5 <= args.threshold < 1 or args.workers < 0 or not 0 < args.lr <= 5e-5:
        p.error("threshold 須在 [0.5, 1)，workers 非負，lr 在 (0, 5e-5]")
    if args.smoke and args.stage != "train":
        p.error("--smoke 只用於 train")
    args.output = out
    return args


def print_json(value):
    from v3.storage import json_bytes

    print(json_bytes(value).decode(), end="", flush=True)


# ---------- 偽標籤（純 numpy，可在 CPU 測試） ----------

def select_pseudo(pa, pb, threshold, require_agree, weight_a=WEIGHT_A):
    from v16.probabilities import fuse

    fused = fuse(pa, pb, weight_a)
    labels = fused.argmax(axis=1)
    confidence = fused.max(axis=1)
    keep = confidence >= np.float32(threshold)
    if require_agree:
        keep &= pa.argmax(axis=1) == pb.argmax(axis=1)
    return fused, labels, confidence, keep


def load_v16_cache(path, ids, classes):
    from v16.probabilities import array_digest, validate_matrix

    with np.load(path, allow_pickle=False) as data:
        p = data["probabilities"]
        if data["ids"].tolist() != ids or data["sha256"].item() != array_digest(p):
            raise ValueError(f"V16 機率快取的 ID 順序或 SHA 不符：{path}")
    validate_matrix(p, len(ids), classes)
    return p


def learning_rate(epoch, epochs, base):
    if epochs == 1:
        return base
    return MIN_LR + (base - MIN_LR) * (1 + math.cos(math.pi * (epoch - 1) / (epochs - 1))) / 2


# ---------- 共用來源 ----------

def sources(args):
    import classification_v17 as v17

    return v17.prepare_inputs(args)


def prepare(args, source):
    from v3.storage import file_sha256, json_bytes, publish

    path = args.output / "prepare.json"
    if path.is_file():
        record = read_prepared(args, source)
        print_json({k: record[k] for k in ("pseudo_images", "threshold", "require_agree", "pseudo_class_min",
                                          "pseudo_class_max", "v16_replay_mismatches", "v14_a3_epoch_seconds")})
        return
    ids, classes = source.ids["test"], source.classes
    folder = args.v16_candidate.parent / "cache" / "test"
    paths = {name: folder / name / "all.npz" for name in ("a", "b")}
    pa, pb = (load_v16_cache(paths[n], ids, len(classes)) for n in ("a", "b"))
    require_agree = not args.allow_disagree
    fused, labels, confidence, keep = select_pseudo(pa, pb, args.threshold, require_agree)
    v16_mismatch = sum(int(labels[i]) != source.v16_labels[item] for i, item in enumerate(ids))
    a3_mismatch = sum(int(pa[i].argmax()) != source.a3_labels[item] for i, item in enumerate(ids))
    if v16_mismatch or a3_mismatch:
        raise ValueError(f"V16 機率快取無法重現已提交結果：v16={v16_mismatch}, a3={a3_mismatch}")
    chosen = np.flatnonzero(keep)
    per_class = Counter(classes[int(labels[i])] for i in chosen)
    record = {
        "protocol": FORMAT, "identity": source.identity, "threshold": args.threshold,
        "require_agree": require_agree, "weight_a": WEIGHT_A,
        "v16_cache_sha256": {n: file_sha256(p) for n, p in paths.items()},
        "v16_replay_mismatches": v16_mismatch, "a3_replay_mismatches": a3_mismatch,
        "v14_a3_epoch_seconds": source.rows[3].get("epoch_seconds"),
        "pseudo_images": int(len(chosen)), "pseudo_fraction": len(chosen) / len(ids),
        "pseudo_mean_confidence": float(confidence[chosen].mean()) if len(chosen) else None,
        "pseudo_class_min": min((per_class.get(c, 0) for c in classes), default=0),
        "pseudo_class_max": max(per_class.values(), default=0),
        "pseudo_per_class": {c: per_class.get(c, 0) for c in classes},
        "pseudo": [[ids[i], int(labels[i]), float(confidence[i])] for i in chosen],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    publish(path, json_bytes(record))
    print_json({k: record[k] for k in ("pseudo_images", "pseudo_fraction", "pseudo_mean_confidence", "threshold",
                                      "require_agree", "pseudo_class_min", "pseudo_class_max",
                                      "v16_replay_mismatches", "a3_replay_mismatches",
                                      "v14_a3_epoch_seconds")})


def read_prepared(args, source):
    from v2.data_checks import read_json

    path = args.output / "prepare.json"
    if not path.is_file():
        raise SystemExit("先執行 V18 prepare。")
    record = read_json(path)
    if record["identity"] != source.identity or record["protocol"] != FORMAT:
        raise ValueError("V18 prepare.json 與目前來源不符；請改用新的 --output")
    return record


# ---------- 訓練 ----------

class PseudoImages:
    def __init__(self, samples, pseudo, basic, repeat):
        path_of = dict((item, path) for path, item in samples)
        self.rows = [(path_of[item], int(label)) for item, label, _ in pseudo] * repeat
        self.basic = basic

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from v7.impulse import clean_impulses
        from v8.data import open_rgb

        path, label = self.rows[index]
        # 測試推論固定先去顆粒，偽標籤訓練圖採相同前處理再做 V8 基本增強。
        image, _ = clean_impulses(open_rgb(path))
        return self.basic(image), label, "pseudo"


def train_epoch(model, optimizer, dataset, args, seed, name, limit=None):
    import torch
    from v8.experiment import accumulation_size, data_loader
    from v14.training import EFFECTIVE_BATCH
    from v16.probabilities import Progress

    model.train()
    loader = data_loader(dataset, args.batch_size, args.workers, seed)
    accum = EFFECTIVE_BATCH // args.batch_size
    total = len(dataset) if limit is None else min(len(dataset), limit * args.batch_size)
    progress = Progress(name, total)
    optimizer.zero_grad(set_to_none=True)
    counts, seen, loss, steps, group = Counter(), 0, 0.0, 0, 0
    for index, (images, labels, kinds) in enumerate(loader):
        if index % accum == 0:
            group = accumulation_size(total - seen, args.batch_size, accum)
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = torch.nn.functional.cross_entropy(model(images), labels, reduction="sum")
        (value / group).backward()
        seen += len(labels)
        loss += value.detach().item()
        if not math.isfinite(loss):
            raise RuntimeError("V18 loss 非有限值")
        counts.update(kinds)
        if (index + 1) % accum == 0 or seen >= total:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        progress.advance(len(labels))
        if seen >= total:
            break
    if seen != total:
        raise RuntimeError("V18 epoch 沒有跑完整個資料集")
    return {"loss": loss / seen, "images": seen, "optimizer_steps": steps, "kinds": dict(counts)}


def screen_metrics(model, source, args):
    import classification_v14 as v14
    from v8.experiment import labels_for, report_metrics

    record = {"eval_batch_size": source.identity["screen_batch_size"]}
    budget = SimpleNamespace(check=lambda: None)
    started = time.monotonic()
    predictions = v14.evaluate(model, source.info, record, args, budget)
    metrics = report_metrics(source.classes, labels_for(source.info.screen), predictions)
    return {"clean": metrics["accuracy"]["clean"], "hard": metrics["hard_accuracy"],
            "by_condition": metrics["accuracy"], "seconds": time.monotonic() - started}


def epoch_paths(folder):
    paths = sorted(folder.glob("epoch_*.pt"))
    if [p.name for p in paths] != [f"epoch_{i:02d}.pt" for i in range(1, len(paths) + 1)]:
        raise ValueError("V18 epoch checkpoints 不連續")
    return paths


def train(args, source, record):
    import torch
    from torch.utils.data import ConcatDataset

    import classification_v17 as v17
    from classification_v2 import save_checkpoint, transforms_v2, write_json
    from classification_v12 import reference_checkpoint
    from v2.data_checks import read_json
    from v3.inference import Views, infer_view_batches
    from v3.storage import file_sha256
    from v6.composite import Targets
    from v8.experiment import data_loader, init_cuda
    from v8.models import configure, make_model
    from v12.model import environment
    from v14.training import ContinuationImages
    from classification_v8 import TestImages

    folder = args.output / ("smoke" if args.smoke else "train")
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": FORMAT, "prepare_sha256": file_sha256(args.output / "prepare.json"),
                "epochs": args.epochs, "lr": args.lr, "pseudo_repeat": args.pseudo_repeat,
                "batch_size": args.batch_size, "environment": environment()}
    settings = folder / "settings.json"
    if not args.smoke:
        if settings.is_file():
            saved = read_json(settings)
            if {k: v for k, v in saved.items() if k != "batch_size"} != {
                    k: v for k, v in identity.items() if k != "batch_size"}:
                raise ValueError("V18 訓練設定與已開始的訓練不同；請改用新的 --output")
        else:
            write_json(settings, identity)
    history_path = folder / "history.json"
    history = read_json(history_path) if history_path.is_file() and not args.smoke else []
    done = len(epoch_paths(folder)) if not args.smoke else 0
    if len(history) != done:
        raise ValueError("V18 history.json 與 checkpoint 數量不一致")
    if done >= args.epochs:
        print_json({"decision": "already_complete", "history": history})
        return

    seed = source.info.inputs["seed"] + 18
    init_cuda(seed)
    parent = reference_checkpoint(args, source.info)
    model = make_model("convnext_tiny", len(source.classes))
    model.load_state_dict(v17.load_parent(source, "a3"), strict=True)
    model.cuda()
    _, optimizer = configure(model, "convnext_tiny", "full", args.lr, 1e-4)
    if done:
        last = torch.load(folder / f"epoch_{done:02d}.pt", map_location="cpu", weights_only=True)
        if last["identity"]["prepare_sha256"] != identity["prepare_sha256"] or last["epoch"] != done:
            raise ValueError("V18 續跑 checkpoint 不符")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        del last
        print(f"V18 從 epoch {done} 續跑", flush=True)

    basic, evaluation = transforms_v2()
    targets = Targets(**source.info.inputs["targets"])
    pseudo = PseudoImages(source.test.samples, record["pseudo"], basic, args.pseudo_repeat)

    if args.smoke:
        dataset = ConcatDataset([
            ContinuationImages(source.info.root, source.info.training, basic, evaluation, targets,
                               parent["config"]["augmentation"], seed, "a", 1), pseudo])
        torch.cuda.reset_peak_memory_stats()
        stats = train_epoch(model, optimizer, dataset, args, seed, "V18 smoke train", limit=8)
        model.eval()
        views = Views("multiscale", transforms_v2()[1])
        loader = data_loader(TestImages(source.test.samples[:32], views, True), 16, args.workers)
        shapes = [p.shape for _, p, _, _ in infer_view_batches(model, loader)]
        screen = screen_metrics(model, source, args)
        print_json({"decision": "smoke_ok", "train": stats, "test_view_batches": [list(s) for s in shapes],
                    "screen_after_8_batches": {k: screen[k] for k in ("clean", "hard")},
                    "pseudo_images_per_epoch": len(pseudo), "train_images_per_epoch": len(source.info.training),
                    "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30})
        return

    reference = source.rows[3]["metrics"]
    for epoch in range(done + 1, args.epochs + 1):
        epoch_seed = seed + epoch
        init_cuda(epoch_seed)
        lr = learning_rate(epoch, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        dataset = ConcatDataset([
            ContinuationImages(source.info.root, source.info.training, basic, evaluation, targets,
                               parent["config"]["augmentation"], epoch_seed, "a", epoch), pseudo])
        print(f"V18 epoch {epoch}/{args.epochs} | train {len(source.info.training):,} + pseudo {len(pseudo):,} "
              f"= {len(dataset):,} images | LR {lr:.2g}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stats = train_epoch(model, optimizer, dataset, args, epoch_seed, f"V18 train epoch {epoch}")
        torch.cuda.synchronize()
        screen = screen_metrics(model, source, args)
        row = {**stats, "epoch": epoch, "lr": lr, "batch_size": args.batch_size,
               "seconds": time.perf_counter() - started,
               "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30, "screen": screen,
               "clean_delta_vs_a3_pp": 100 * (screen["clean"] - reference["accuracy"]["clean"]),
               "hard_delta_vs_a3_pp": 100 * (screen["hard"] - reference["hard_accuracy"])}
        path = folder / f"epoch_{epoch:02d}.pt"
        save_checkpoint(path, {"identity": identity, "epoch": epoch, "training": row,
                               "model": model.state_dict(), "optimizer": optimizer.state_dict()})
        history.append({**row, "checkpoint_sha256": file_sha256(path)})
        write_json(history_path, history)
        print_json({k: row[k] for k in ("epoch", "loss", "seconds", "peak_vram_gib",
                                        "clean_delta_vs_a3_pp", "hard_delta_vs_a3_pp")} |
                   {"clean": screen["clean"], "hard": screen["hard"]})
    print_json({"decision": "train_complete", "epochs": [
        {k: r[k] for k in ("epoch", "loss", "clean_delta_vs_a3_pp", "hard_delta_vs_a3_pp")} for r in history]})


# ---------- 推論 ----------

def predict(args, source, record):
    import torch

    import classification_v14 as v14
    from classification_v2 import transforms_v2
    from classification_v8 import TestImages
    from v2.data_checks import TEST_COUNT, read_json
    from v3.inference import Prediction, Views, infer_view_batches
    from v3.reports import read_reference, submission
    from v3.storage import file_sha256, json_bytes, publish, publish_csv
    from v8.experiment import data_loader, init_cuda
    from v8.models import make_model
    from v16.probabilities import cache_or_infer

    folder = args.output / "train"
    paths = epoch_paths(folder)
    if not paths:
        raise SystemExit("尚無 V18 checkpoint，先執行 train。")
    epoch = args.epoch or len(paths)
    if epoch > len(paths):
        raise SystemExit(f"epoch {epoch} 尚未完成")
    path = folder / f"epoch_{epoch:02d}.pt"
    history = read_json(folder / "history.json")
    digest = file_sha256(path)
    if history[epoch - 1]["checkpoint_sha256"] != digest:
        raise ValueError("V18 checkpoint SHA 與 history 不符")
    dest = args.output / f"candidate_epoch_{epoch:02d}"
    if dest.exists():
        raise FileExistsError(f"V18 candidate already exists: {dest}")
    ids, classes = source.ids["test"], source.classes
    model = None

    def batches():
        nonlocal model
        init_cuda(source.info.inputs["seed"])
        model = make_model("convnext_tiny", len(classes))
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        del state
        model.cuda().eval()
        views = Views("multiscale", transforms_v2()[1])
        loader = data_loader(TestImages(source.test.samples, views, True),
                             source.identity["test_batch_size"], args.workers)
        yield from infer_view_batches(model, loader)

    cache = args.output / "cache" / "test" / f"epoch_{epoch:02d}.npz"
    metadata = {"protocol": FORMAT, "checkpoint_sha256": digest, "epoch": epoch}
    p, seconds, _ = cache_or_infer(cache, ids, len(classes), metadata, batches, f"V18 epoch {epoch} test")
    labels = p.argmax(axis=1)
    predictions = {item: Prediction(int(labels[i]), float(p[i, labels[i]]), 0.0, 0.0) for i, item in enumerate(ids)}
    with TemporaryDirectory(prefix=".candidate-", dir=args.output) as temporary:
        staged = Path(temporary)
        csv = staged / "submission_v18.csv"
        submission(csv, ids, predictions, classes, count=TEST_COUNT)
        saved = read_reference(csv, ids, classes)
        pseudo = {item: label for item, label, _ in record["pseudo"]}
        result = {"submission": str(dest / csv.name), "submission_sha256": file_sha256(csv), "images": len(ids),
                  "epoch": epoch, "checkpoint_sha256": digest, "test_seconds": seconds,
                  "pseudo_images": len(pseudo),
                  "pseudo_label_kept": sum(saved[i] == l for i, l in pseudo.items()),
                  "non_pseudo_changed_vs_v16": sum(saved[i] != source.v16_labels[i] for i in ids if i not in pseudo)}
        for name, reference in (("a3", source.a3_labels), ("v16", source.v16_labels)):
            counts, moves, changed = v14.test_comparison(ids, reference, saved, classes)
            publish_csv(staged / f"class_changes_vs_{name}.csv",
                        ("character", "reference_count", "v18_count", "removed", "added", "net"), counts)
            publish_csv(staged / f"transitions_vs_{name}.csv", ("reference_label", "v18_label", "count"), moves)
            result[f"changed_vs_{name}"] = changed
            result[f"changed_vs_{name}_pct"] = 100 * changed / len(ids)
        publish(staged / "summary.json", json_bytes(result))
        staged.rename(dest)
    print_json(result)


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V18 輸出已有階段正在使用。")
        if args.stage != "prepare":
            from v12.model import gpu_probe, permitted

            permitted(args.allow_imagenet)
            env = gpu_probe()
            if env["status"] != "ready":
                print_json(env)
                raise SystemExit("V18 GPU 預檢未通過。")
        source = sources(args)
        if args.stage == "prepare":
            prepare(args, source)
            return 0
        record = read_prepared(args, source)
        if args.stage == "train":
            train(args, source, record)
        else:
            predict(args, source, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
