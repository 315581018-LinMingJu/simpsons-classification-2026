"""V20：重現作業的測試集干擾流程做訓練增強，並以 ImageNet-22K 預訓練 ConvNeXt 完整微調。

只使用訓練集標籤；測試集只在 predict 產生最終預測。切分完全重現 baseline.py（seed 2026、每類 20% 驗證）。

stage
  backgrounds  CPU＋網路：下載 DTD 紋理照片當背景（作業增強 #2 需要背景圖），失敗時改用程序化背景。
  preview      CPU：產生 36 張「測試風格」合成圖的拼貼，供人工比對真實測試圖。
  train        GPU：完整微調；每 epoch 驗證 Clean 與 Test-style 兩組驗證集。--smoke 只跑少量批次並估算時間。
  predict      GPU：最佳 checkpoint（依 Test-style 驗證）＋水平翻轉 TTA，輸出 submission_v20.csv。
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 2026


# ---------------------------------------------------------------- 共用工具

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


class Progress:
    """依實際完成的圖片數更新：張數、百分比、經過時間、ETA。"""

    def __init__(self, name, total, stream=None):
        self.name, self.total, self.done = name, total, 0
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.start = self.last = time.monotonic()
        self.render()

    def render(self):
        elapsed = time.monotonic() - self.start
        pct = self.done / self.total if self.total else 1
        eta = f"{elapsed / self.done * (self.total - self.done):.0f}s" if self.done else "--"
        bar = "#" * int(20 * pct) + "-" * (20 - int(20 * pct))
        end = "\n" if (not self.tty or self.done >= self.total) else ""
        self.stream.write(("\r" if self.tty else "") + f"{self.name} [{bar}] {self.done:,}/{self.total:,} "
                          f"{pct:6.2%} elapsed {elapsed:.0f}s ETA {eta}" + end)
        self.stream.flush()

    def advance(self, n):
        self.done += n
        now = time.monotonic()
        if self.tty or self.done >= self.total or now - self.last >= 15:
            self.render()
            self.last = now


# ---------------------------------------------------------------- 資料與 baseline 切分

def train_root(path):
    p = Path(path)
    for _ in range(5):
        dirs = sorted(x for x in p.iterdir() if x.is_dir() and not x.name.startswith("."))
        if len(dirs) == 1:
            p = dirs[0]
        else:
            return p
    raise ValueError("Too many nested folders")


def baseline_split(root, val_fraction=.2, seed=SEED):
    """逐行重現 baseline.py 的切分：類別排序、檔案排序、同一個 Random(seed) 依序洗牌。"""
    classes = sorted(x.name for x in root.iterdir() if x.is_dir() and not x.name.startswith("."))
    if len(classes) != 50:
        raise ValueError(f"Expected 50 character folders, found {len(classes)}")
    rng = random.Random(seed)
    train, val = [], []
    for y, c in enumerate(classes):
        files = sorted(p for p in (root / c).rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
        rng.shuffle(files)
        nv = max(1, min(len(files) - 1, round(len(files) * val_fraction)))
        val.extend((p, y) for p in files[:nv])
        train.extend((p, y) for p in files[nv:])
    return classes, train, val


def test_items(data):
    template = data / "sample_submission.csv"
    with template.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "character"]:
            raise ValueError("sample_submission.csv must have id,character")
        ids = [row["id"] for row in reader]
    folder = data / "test_images" / "test_images"
    images = {p.stem: p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS}
    if len(ids) != 10791 or len(set(ids)) != len(ids) or set(ids) != set(images):
        raise ValueError("Test images and template IDs differ")
    return ids, [images[i] for i in ids]


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------- 背景

def background_paths(folder):
    if folder is None or not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def procedural_background(size, rng):
    """沒有背景照片時的替代：低頻色塊＋紋理雜訊。"""
    w, h = size
    cells = rng.choice((2, 4, 8, 16, 32))
    base = np.array([[[rng.randint(0, 255) for _ in range(3)] for _ in range(cells)] for _ in range(cells)],
                    dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    nr = np.random.default_rng(rng.getrandbits(32))
    arr = arr + nr.normal(0, rng.uniform(0, 40), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def load_background(paths, size, rng):
    if not paths:
        return procedural_background(size, rng)
    bg = open_rgb(rng.choice(paths))
    w, h = bg.size
    s = rng.uniform(.35, 1.0)
    cw, ch = max(16, int(w * s)), max(16, int(h * s))
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return bg.crop((x0, y0, x0 + cw, y0 + ch)).resize(size, Image.Resampling.BILINEAR)


def composite(img, bg_paths, rng):
    """作業增強 #2：近黑(<20)或近白(>235)像素視為透明，外加 10–30 px 邊框後貼到背景上。"""
    arr = np.asarray(img.convert("RGB"))
    keep = ~(np.all(arr < 20, axis=2) | np.all(arr > 235, axis=2))
    px, py = rng.randint(10, 30), rng.randint(10, 30)
    h, w = arr.shape[:2]
    out = np.array(load_background(bg_paths, (w + 2 * px, h + 2 * py), rng))
    region = out[py:py + h, px:px + w]
    region[keep] = arr[keep]
    return Image.fromarray(out)


# ---------------------------------------------------------------- 作業增強 #1（張量版）

def elastic_displacement(h, w, alpha, sigma, gen):
    import torch
    import torchvision.transforms.v2.functional as F

    k = int(8 * sigma + 1) | 1
    dx = torch.rand([1, 1, h, w], generator=gen) * 2 - 1
    dy = torch.rand([1, 1, h, w], generator=gen) * 2 - 1
    dx = F.gaussian_blur(dx, [k, k], [sigma, sigma]) * alpha / w
    dy = F.gaussian_blur(dy, [k, k], [sigma, sigma]) * alpha / h
    return torch.cat([dx, dy], 1).permute(0, 2, 3, 1)


def perspective_points(w, h, d, rng):
    hw, hh = int(d * w / 2), int(d * h / 2)
    tl = [rng.randint(0, hw), rng.randint(0, hh)]
    tr = [w - 1 - rng.randint(0, hw), rng.randint(0, hh)]
    br = [w - 1 - rng.randint(0, hw), h - 1 - rng.randint(0, hh)]
    bl = [rng.randint(0, hw), h - 1 - rng.randint(0, hh)]
    return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], [tl, tr, br, bl]


def distort(img, rng):
    """依作業增強 #1 的項目，提高機率以符合實際測試圖（多數經過旋轉／仿射）。"""
    import torch
    import torchvision.transforms.v2.functional as F
    from torchvision.transforms import InterpolationMode as IM

    gen = torch.Generator().manual_seed(rng.getrandbits(32))
    t = F.to_image(img).float().div(255)
    _, h, w = t.shape
    if rng.random() < .5:
        t = F.horizontal_flip(t)
    if rng.random() < .05:
        t = F.vertical_flip(t)
    g = rng.random()
    if g < .30:
        t = F.rotate(t, rng.uniform(-35, 35), interpolation=IM.BILINEAR, fill=0)
    elif g < .60:
        t = F.affine(t, angle=rng.uniform(-50, 50), translate=[rng.uniform(-.2, .2) * w, rng.uniform(-.2, .2) * h],
                     scale=rng.uniform(.55, .95), shear=[0.0, 0.0], interpolation=IM.BILINEAR, fill=0)
    elif g < .78:
        s, e = perspective_points(w, h, rng.uniform(.2, .6), rng)
        t = F.perspective(t, s, e, interpolation=IM.BILINEAR, fill=0)
    if rng.random() < .30:
        t = F.adjust_brightness(t, rng.uniform(.6, 1.4))
        t = F.adjust_contrast(t, rng.uniform(.6, 1.4))
        t = F.adjust_saturation(t, rng.uniform(.6, 1.4))
        t = F.adjust_hue(t, rng.uniform(-.1, .1))
    if rng.random() < .15:
        t = F.rgb_to_grayscale(t, num_output_channels=3)
    if rng.random() < .15:
        t = F.invert(t)
    if rng.random() < .12:
        t = F.posterize((t * 255).round().to(torch.uint8), rng.choice((2, 3, 4))).float().div(255)
    if rng.random() < .08:
        t = F.solarize(t, rng.uniform(.5, 1.0))
    if rng.random() < .10:
        t = F.adjust_sharpness(t, 2.0)
    if rng.random() < .12:
        t = F.elastic(t, elastic_displacement(h, w, rng.uniform(50, 250), 5.0, gen), interpolation=IM.BILINEAR,
                      fill=0)
    if rng.random() < .15:
        k = rng.choice((5, 7, 9))
        t = F.gaussian_blur(t, [k, k], [rng.uniform(.1, 3.0)] * 2)
    n = rng.random()
    if n < .10:
        t = t + torch.randn(t.shape, generator=gen) * rng.uniform(.02, .08)
    elif n < .18:
        t = t * (1 + torch.randn(t.shape, generator=gen) * .1)
    elif n < .26:
        t = t + torch.poisson(torch.full(t.shape, .1), generator=gen) / 255
    elif n < .36:
        r = torch.rand(t.shape[1:], generator=gen)
        p = rng.uniform(.01, .06)
        t = t.clone()
        t[:, r < p] = 1
        t[:, r > 1 - p] = 0
    t = t.clamp(0, 1)
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def testify(img, bg_paths, rng):
    out = composite(distort(img, rng), bg_paths, rng)
    if rng.random() < .7:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=rng.randint(70, 95))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out


# ---------------------------------------------------------------- Dataset

class Images:
    """mode: train（隨機，p_testify 機率套用測試風格）、clean、testlike（每張固定種子）、test。"""

    def __init__(self, items, mode, size, mean, std, bg_paths=(), p_testify=.85):
        self.items, self.mode, self.size = items, mode, size
        self.mean, self.std, self.bg, self.p = mean, std, list(bg_paths), p_testify

    def __len__(self):
        return len(self.items)

    def tensor(self, img):
        import torch
        import torchvision.transforms.v2.functional as F

        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        t = F.to_image(img).float().div(255)
        return F.normalize(t, self.mean, self.std).to(torch.float32)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = open_rgb(path)
        if self.mode == "train":
            rng = random.Random(random.getrandbits(64))
            if rng.random() < self.p:
                img = testify(img, self.bg, rng)
            elif rng.random() < .5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif self.mode == "testlike":
            img = testify(img, self.bg, random.Random(SEED * 1_000_003 + i))
        return self.tensor(img), label


def worker_init(_):
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def loader(dataset, batch, workers, shuffle=False, seed=SEED):
    import torch

    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       pin_memory=True, persistent_workers=False, worker_init_fn=worker_init,
                                       drop_last=False, generator=torch.Generator().manual_seed(seed),
                                       prefetch_factor=4 if workers else None)


# ---------------------------------------------------------------- 模型

def make_model(name, classes, pretrained, drop_path=.2):
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=classes, drop_path_rate=drop_path)


def norm_stats(model):
    cfg = model.pretrained_cfg
    return list(cfg.get("mean", (.485, .456, .406))), list(cfg.get("std", (.229, .224, .225)))


def evaluate(model, items, mode, args, mean, std, bg, name, limit=None):
    import torch

    model.eval()
    data = items if limit is None else items[:limit]
    ds = Images(data, mode, args.size, mean, std, bg)
    progress = Progress(name, len(ds))
    preds, labels = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader(ds, args.eval_batch, args.workers):
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            logits = model(x).float() + model(torch.flip(x, dims=[3])).float()
            preds.append(logits.argmax(1).cpu())
            labels.append(y)
            progress.advance(len(y))
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float((p == y).mean()), p, y


def confusion(p, y, n):
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def save_confusion(path, matrix, classes):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for c, row in zip(classes, matrix):
            w.writerow([c, *row.tolist()])


# ---------------------------------------------------------------- stages

def stage_backgrounds(args):
    import torchvision

    root = args.backgrounds.parent
    root.mkdir(parents=True, exist_ok=True)
    found = background_paths(args.backgrounds)
    if not found:
        try:
            torchvision.datasets.DTD(root=str(root), split="train", download=True)
        except Exception as exc:  # 網路或來源失效時改用程序化背景
            print(f"DTD 下載失敗：{exc!r}；train 會改用程序化背景。", flush=True)
        found = background_paths(args.backgrounds)
    print_json({"backgrounds": len(found), "folder": str(args.backgrounds),
                "source": "DTD (Describable Textures Dataset, Cimpoi et al. 2014) via torchvision" if found
                else "procedural"})


def stage_preview(args):
    root = train_root(args.data / "train")
    _, train, _ = baseline_split(root)
    bg = background_paths(args.backgrounds)
    rng = random.Random(7)
    picks = rng.sample(train, 36)
    tile = 260
    sheet = Image.new("RGB", (6 * tile, 6 * tile), (255, 0, 255))
    for i, (path, _) in enumerate(picks):
        img = testify(open_rgb(path), bg, random.Random(i))
        img.thumbnail((tile - 4, tile - 4))
        sheet.paste(img, ((i % 6) * tile, (i // 6) * tile))
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "preview.jpg"
    sheet.save(out, quality=88)
    print_json({"preview": str(out), "backgrounds": len(bg)})


def stage_train(args):
    import torch

    root = train_root(args.data / "train")
    classes, train, val = baseline_split(root)
    bg = background_paths(args.backgrounds)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = make_model(args.model, len(classes), pretrained=True, drop_path=args.drop_path)
    mean, std = norm_stats(model)
    model.cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = args.epochs * steps_per_epoch
    warm = min(steps_per_epoch, 1000)

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return .01 + .99 * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "pretrained": "timm/" + args.model, "size": args.size, "epochs": args.epochs,
              "batch": args.batch, "lr": args.lr, "weight_decay": .05, "label_smoothing": .1,
              "drop_path": args.drop_path, "p_testify": args.p_testify, "seed": SEED,
              "split": "baseline.py (seed 2026, 20% per class)", "train_images": len(train),
              "val_images": len(val), "backgrounds": len(bg), "classes": classes, "mean": mean, "std": std}
    history, best, start = [], -1.0, 1
    last = out / "last.pt"
    if args.smoke:
        ds = Images(train, "train", args.size, mean, std, bg, args.p_testify)
        model.train()
        progress = Progress("V20 smoke train", 30 * args.batch)
        t0 = time.monotonic()
        for i, (x, y) in enumerate(loader(ds, args.batch, args.workers, shuffle=True)):
            x, y = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            progress.advance(len(y))
            if i == 4:
                t0, seen = time.monotonic(), 0
            elif i > 4:
                seen += len(y)
            if i + 1 >= 30:
                break
        rate = seen / (time.monotonic() - t0)
        acc_c, _, _ = evaluate(model, val, "clean", args, mean, std, bg, "V20 smoke val clean", 512)
        acc_t, _, _ = evaluate(model, val, "testlike", args, mean, std, bg, "V20 smoke val testlike", 512)
        print_json({"decision": "smoke_ok", "train_images_per_sec": rate,
                    "est_train_minutes_per_epoch": len(train) / rate / 60, "loss_last": float(loss),
                    "val_clean_512": acc_c, "val_testlike_512": acc_t, "train_images": len(train),
                    "val_images": len(val), "backgrounds": len(bg),
                    "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30})
        return
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise SystemExit("設定與既有訓練不同；請改用新的 --output")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        history, best, start = state["history"], state["best"], state["epoch"] + 1
        print(f"V20 從 epoch {start} 續跑", flush=True)
    else:
        write_json(out / "config.json", config)
        with (out / "split.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "character", "split"])
            w.writerows((str(p.relative_to(root)), classes[y], "train") for p, y in train)
            w.writerows((str(p.relative_to(root)), classes[y], "validation") for p, y in val)
    for epoch in range(start, args.epochs + 1):
        ds = Images(train, "train", args.size, mean, std, bg, args.p_testify)
        model.train()
        progress = Progress(f"V20 train epoch {epoch}/{args.epochs}", len(ds))
        t0, loss_sum, seen = time.monotonic(), 0.0, 0
        for x, y in loader(ds, args.batch, args.workers, shuffle=True, seed=SEED + epoch):
            x, y = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss) * len(y)
            seen += len(y)
            if not math.isfinite(loss_sum):
                raise RuntimeError("loss 非有限值")
            progress.advance(len(y))
        train_s = time.monotonic() - t0
        acc_c, pc, yc = evaluate(model, val, "clean", args, mean, std, bg, f"V20 val clean e{epoch}")
        acc_t, pt, yt = evaluate(model, val, "testlike", args, mean, std, bg, f"V20 val testlike e{epoch}")
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "val_clean": acc_c, "val_testlike": acc_t,
               "train_seconds": train_s, "epoch_seconds": time.monotonic() - t0,
               "lr_end": sched.get_last_lr()[0], "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30}
        history.append(row)
        if acc_t > best:
            best = acc_t
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "row": row},
                       out / "best.pt.tmp")
            os.replace(out / "best.pt.tmp", out / "best.pt")
            save_confusion(out / "confusion_val_clean.csv", confusion(pc, yc, len(classes)), classes)
            save_confusion(out / "confusion_val_testlike.csv", confusion(pt, yt, len(classes)), classes)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "config": config, "epoch": epoch, "history": history, "best": best}, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", last)
        write_json(out / "history.json", history)
        print_json({**row, "best_val_testlike": best})
    print_json({"decision": "train_complete", "best_val_testlike": best,
                "epochs": [{k: r[k] for k in ("epoch", "train_loss", "val_clean", "val_testlike")} for r in history]})


def stage_predict(args):
    import torch

    ckpt_path = args.output / ("best.pt" if args.checkpoint == "best" else "last.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = state["config"]
    classes = config["classes"]
    model = make_model(config["model"], len(classes), pretrained=False, drop_path=0)
    model.load_state_dict(state["model"])
    model.cuda().to(memory_format=torch.channels_last).eval()
    ids, paths = test_items(args.data)
    ds = Images([(p, 0) for p in paths], "test", config["size"], config["mean"], config["std"])
    progress = Progress("V20 test", len(ds))
    probs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader(ds, args.eval_batch, args.workers):
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            p = model(x).float().softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)
            probs.append((p / 2).cpu())
            progress.advance(len(x))
    probs = torch.cat(probs).numpy().astype(np.float32)
    labels = probs.argmax(1)
    dest = args.output / f"candidate_epoch_{state['epoch']:02d}"
    if dest.exists():
        raise FileExistsError(f"候選已存在：{dest}")
    tmp = args.output / f".candidate-{os.getpid()}"
    tmp.mkdir()
    csv_path = tmp / "submission_v20.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "character"])
        w.writerows((i, classes[int(k)]) for i, k in zip(ids, labels))
    np.savez_compressed(tmp / "test_probabilities.npz", ids=np.asarray(ids), probabilities=probs)
    summary = {"submission": str(dest / "submission_v20.csv"), "submission_sha256": sha256(csv_path),
               "images": len(ids), "epoch": state["epoch"], "checkpoint": ckpt_path.name,
               "val_clean": state["row"]["val_clean"] if "row" in state else None,
               "val_testlike": state["row"]["val_testlike"] if "row" in state else None,
               "mean_confidence": float(probs.max(1).mean()),
               "predicted_class_min": int(np.bincount(labels, minlength=len(classes)).min()),
               "predicted_class_max": int(np.bincount(labels, minlength=len(classes)).max())}
    for name, ref in (("v16", Path("runs/v16/archive-fix-49dc35247fa67babe7ccac7c1ce12136dd780ade/candidate/"
                                   "submission_v16.csv")),
                      ("v18", Path("runs/v18/candidate_epoch_02/submission_v18.csv"))):
        if ref.is_file():
            with ref.open(newline="") as f:
                rows = list(csv.reader(f))[1:]
            if [r[0] for r in rows] == ids:
                summary[f"changed_vs_{name}"] = sum(r[1] != classes[int(k)] for r, k in zip(rows, labels))
    write_json(tmp / "summary.json", summary)
    os.replace(tmp, dest)
    print_json(summary)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("backgrounds", "preview", "train", "predict"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--output", type=Path, default=Path("runs/v20"))
    p.add_argument("--backgrounds", type=Path, default=Path("data/backgrounds/dtd/dtd/images"))
    p.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--drop-path", type=float, default=.2)
    p.add_argument("--p-testify", type=float, default=.85)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--checkpoint", choices=("best", "last"), default="best")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if not args.output.resolve().is_relative_to(Path("runs").resolve()):
        p.error("--output must be under runs/")
    return args


def main(argv=None):
    args = arguments(argv)
    {"backgrounds": stage_backgrounds, "preview": stage_preview, "train": stage_train,
     "predict": stage_predict}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
