"""報告用：在相同驗證集（baseline 切分）上，以四種干擾條件評估模型，量化外框／背景／雜訊的影響。

條件（每張圖固定種子，所有模型看到完全相同的圖片）：
  clean      原始驗證圖
  distort    只有作業增強 #1（幾何、色彩、反相、彈性變形、雜訊…）
  composite  只有作業增強 #2（近黑／近白像素透明＋10–30 px 邊框＋照片背景）
  full       #1＋#2＋JPEG，與 V20 的 Test-style 驗證集相同

不使用 TTA，所有模型採相同評估方式。從專案根目錄執行：
  python report/evaluate_conditions.py baseline runs/baseline/best.pt --name baseline
  python report/evaluate_conditions.py v20 runs/v20/main/best.pt --name v20_main
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final"))
sys.path.insert(0, str(ROOT / "baseline"))

import classification_v20 as v20  # noqa: E402

CONDITIONS = ("clean", "distort", "composite", "full")


def apply(condition, img, bg, index):
    rng = random.Random(v20.SEED * 1_000_003 + index)
    if condition == "clean":
        return img
    if condition == "distort":
        return v20.distort(img, rng)
    if condition == "composite":
        return v20.composite(img, bg, rng)
    return v20.testify(img, bg, rng)


class ConditionImages:
    def __init__(self, items, condition, bg, to_tensor):
        self.items, self.condition, self.bg, self.to_tensor = items, condition, bg, to_tensor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = apply(self.condition, v20.open_rgb(path), self.bg, i)
        return self.to_tensor(img), label


def load_baseline(checkpoint):
    import torch
    from PIL import ImageOps, Image

    import baseline as bl

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = bl.ResNet18(len(state["classes"]))
    model.load_state_dict(state["state_dict"])
    size = state["image_size"]

    def to_tensor(img):
        # 與 baseline.py 推論相同：補成正方形（灰色）後縮放，ImageNet Normalize
        im = ImageOps.pad(img, (size, size), method=Image.Resampling.BILINEAR, color=(128, 128, 128))
        a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.
        return torch.from_numpy((a - bl.MEAN) / bl.STD)

    return model, state["classes"], to_tensor, {"epoch": state.get("epoch"),
                                                 "validation_accuracy": state.get("validation_accuracy")}


def load_v20(checkpoint):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = v20.make_model(config["model"], len(config["classes"]), pretrained=False, drop_path=0)
    model.load_state_dict(state["model"])
    images = v20.Images([], "clean", config["size"], config["mean"], config["std"])
    return model, config["classes"], images.tensor, {"epoch": state["epoch"], "model": config["model"],
                                                     "size": config["size"], "p_testify": config["p_testify"]}


def main(argv=None):
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("baseline", "v20"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--backgrounds", type=Path, default=Path("data/backgrounds/dtd/dtd/dtd/dtd/images"))
    p.add_argument("--output", type=Path, default=Path("runs/report"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args(argv)

    root = v20.train_root(args.data / "train")
    classes, _, val = v20.baseline_split(root)
    bg = v20.background_paths(args.backgrounds)
    if not bg:
        raise SystemExit("找不到背景圖，請指定 --backgrounds")
    model, model_classes, to_tensor, info = (load_baseline if args.kind == "baseline" else load_v20)(args.checkpoint)
    if list(model_classes) != classes:
        raise ValueError("checkpoint 類別順序與 baseline 切分不同")
    model.cuda().eval().to(memory_format=torch.channels_last)
    labels = np.asarray([y for _, y in val])
    result = {"name": args.name, "kind": args.kind, "checkpoint": str(args.checkpoint), **info,
              "val_images": len(val), "backgrounds": len(bg), "accuracy": {}}
    preds = {}
    for condition in CONDITIONS:
        ds = ConditionImages(val, condition, bg, to_tensor)
        progress = v20.Progress(f"{args.name} {condition}", len(ds))
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in v20.loader(ds, args.batch, args.workers):
                out.append(model(x.cuda(non_blocking=True).to(memory_format=torch.channels_last)).argmax(1).cpu())
                progress.advance(len(x))
        preds[condition] = torch.cat(out).numpy()
        result["accuracy"][condition] = float((preds[condition] == labels).mean())
    full = v20.confusion(preds["full"], labels, len(classes))
    per_class = full.diagonal() / full.sum(1)
    result["full_worst_classes"] = [[classes[i], round(float(per_class[i]), 4)] for i in np.argsort(per_class)[:8]]
    off = full.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
    result["full_top_confusions"] = [[classes[t], classes[q], int(off[t, q])] for t, q in pairs if off[t, q]]
    args.output.mkdir(parents=True, exist_ok=True)
    v20.save_confusion(args.output / f"{args.name}_confusion_full.csv", full, classes)
    v20.write_json(args.output / f"{args.name}_conditions.json", result)
    v20.print_json(result)


if __name__ == "__main__":
    main()
