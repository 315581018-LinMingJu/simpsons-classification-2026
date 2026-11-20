"""V21：多個 V20 模型的機率平均（ensemble）。

只用驗證集（baseline 切分）評估與比較，測試集只產生最終預測。
與 classification_v20.py 放在同一資料夾執行。

stage
  valprobs  GPU：對某個 V20 run 的 best.pt 計算 Clean／Test-style 驗證集機率（含水平翻轉 TTA），存成 val_probabilities.npz。
  ensemble  CPU：讀取多個 run 的驗證與測試機率，回報各自與平均後的驗證正確率，輸出 submission_v21.csv。
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import classification_v20 as v20


def run_probs(model, items, mode, size, mean, std, bg, args, name):
    import torch

    ds = v20.Images(items, mode, size, mean, std, bg)
    progress = v20.Progress(name, len(ds))
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in v20.loader(ds, args.eval_batch, args.workers):
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            p = model(x).float().softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)
            out.append((p / 2).cpu())
            progress.advance(len(x))
    return torch.cat(out).numpy().astype(np.float32)


def stage_valprobs(args):
    import torch

    run = args.runs[0]
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    root = v20.train_root(args.data / "train")
    classes, _, val = v20.baseline_split(root)
    if classes != config["classes"]:
        raise ValueError("類別順序與訓練時不同")
    bg = v20.background_paths(args.backgrounds)
    if len(bg) != config["backgrounds"]:
        raise ValueError(f"背景張數 {len(bg)} 與訓練時 {config['backgrounds']} 不同，請用相同的 --backgrounds")
    model = v20.make_model(config["model"], len(classes), pretrained=False, drop_path=0)
    model.load_state_dict(state["model"])
    model.cuda().to(memory_format=torch.channels_last).eval()
    labels = np.asarray([y for _, y in val])
    result = {"run": str(run), "epoch": state["epoch"]}
    arrays = {"labels": labels}
    for mode in ("clean", "testlike"):
        p = run_probs(model, val, mode, config["size"], config["mean"], config["std"], bg, args, f"val {mode}")
        arrays[mode] = p
        result[f"val_{mode}"] = float((p.argmax(1) == labels).mean())
    np.savez_compressed(run / "val_probabilities.npz", **arrays)
    v20.print_json(result)


def find_test_probs(run):
    found = sorted(run.glob("candidate_epoch_*/test_probabilities.npz"))
    if len(found) != 1:
        raise ValueError(f"{run} 需要恰好一個 candidate_epoch_*/test_probabilities.npz，找到 {len(found)} 個")
    return found[0]


def stage_ensemble(args):
    classes = None
    vals, tests, names, ids = [], [], [], None
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        if classes is None:
            classes = config["classes"]
        elif config["classes"] != classes:
            raise ValueError("各 run 類別順序不同")
        with np.load(run / "val_probabilities.npz") as d:
            vals.append({k: d[k] for k in ("labels", "clean", "testlike")})
        with np.load(find_test_probs(run)) as d:
            run_ids = d["ids"].tolist()
            if ids is None:
                ids = run_ids
            elif run_ids != ids:
                raise ValueError("各 run 測試 ID 順序不同")
            tests.append(d["probabilities"])
        names.append(f"{run.name}:{config['model']}@{config['size']}")
    labels = vals[0]["labels"]
    if any(not np.array_equal(v["labels"], labels) for v in vals):
        raise ValueError("各 run 驗證集順序不同")
    report = {"members": names, "per_run": {}}
    for name, v in zip(names, vals):
        report["per_run"][name] = {m: float((v[m].argmax(1) == labels).mean()) for m in ("clean", "testlike")}
    mean_val = {m: sum(v[m] for v in vals) / len(vals) for m in ("clean", "testlike")}
    report["ensemble_val"] = {m: float((mean_val[m].argmax(1) == labels).mean()) for m in ("clean", "testlike")}
    probs = sum(tests) / len(tests)
    pred = probs.argmax(1)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"ensemble-{len(args.runs)}"
    if dest.exists():
        raise FileExistsError(f"候選已存在：{dest}")
    tmp = out / f".ensemble-{os.getpid()}"
    tmp.mkdir()
    path = tmp / "submission_v21.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "character"])
        w.writerows((i, classes[int(k)]) for i, k in zip(ids, pred))
    report.update({"submission": str(dest / path.name), "submission_sha256": v20.sha256(path), "images": len(ids),
                   "predicted_class_min": int(np.bincount(pred, minlength=len(classes)).min()),
                   "predicted_class_max": int(np.bincount(pred, minlength=len(classes)).max())})
    v20_csv = Path("runs/v20/main/candidate_epoch_09/submission_v20.csv")
    if v20_csv.is_file():
        with v20_csv.open(newline="") as f:
            rows = list(csv.reader(f))[1:]
        if [r[0] for r in rows] == ids:
            report["changed_vs_v20"] = sum(r[1] != classes[int(k)] for r, k in zip(rows, pred))
    v20.write_json(tmp / "summary.json", report)
    os.replace(tmp, dest)
    v20.print_json(report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("valprobs", "ensemble"))
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--backgrounds", type=Path, default=Path("data/backgrounds/dtd/dtd/dtd/dtd/images"))
    p.add_argument("--output", type=Path, default=Path("runs/v21"))
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args(argv)
    if args.stage == "valprobs" and len(args.runs) != 1:
        p.error("valprobs 一次處理一個 run")
    {"valprobs": stage_valprobs, "ensemble": stage_ensemble}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
