"""全層延續訓練的資料配方、更新計數與停損。"""

import math
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

from classification_v2 import write_json
from v2.data_checks import read_json
from v4.geometry import seeded
from v5.background import exterior_mask
from v8.data import TrainingImages, open_rgb
from v8.experiment import accumulation_size, data_loader
from v10.augment import bright_background, render_scene

BASE_LR = 3e-5
MIN_LR = 5e-6
EFFECTIVE_BATCH = 64
PATIENCE = 2
CLEAN_DROP = .03
HARD_DROP = .06


def learning_rate(epoch, epochs):
    return MIN_LR + (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * (epoch - 1) / (epochs - 1))) / 2


def mixture(epoch):
    # 由 70% 原配方逐步走到 55%，兩種新分支的機率不依驗證結果改變。
    return ((.15, .15), (.15, .20), (.20, .20), (.20, .25), (.20, .25))[epoch - 1]


def relight(image, targets, rng):
    rgb = np.array(image)
    mask = exterior_mask(rgb)
    if not mask.any():
        return None
    # 侵蝕外圍遮罩以保守留住深色輪廓；原圖幾何及非遮罩像素保持一致。
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) == 255
    if mask.mean() < .06:
        return None
    bg = bright_background(targets, rng).resize(image.size, Image.Resampling.BILINEAR)
    rgb[mask] = np.asarray(bg)[mask]
    return Image.fromarray(rgb)


class ContinuationImages:
    def __init__(self, root, cases, basic, evaluation, targets, augmentation, seed, arm, epoch):
        from torchvision import transforms as T

        self.base = TrainingImages(root, cases, basic, targets, augmentation, seed)
        self.root, self.cases, self.targets, self.seed = root, cases, targets, seed
        self.arm, self.ratios = arm, mixture(epoch)
        self.normalize = T.Compose([T.Resize((224, 224)), *evaluation.transforms[2:]])

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        if self.arm == "b":
            rng = seeded(self.seed, f"v14:mix:{case.item}")
            draw = rng.random()
            light, scene = self.ratios
            if draw < light + scene:
                image = open_rgb(self.root / case.item)
                if draw < light:
                    replacement = relight(image, self.targets, np.random.default_rng(rng.getrandbits(64)))
                    if replacement is not None:
                        return self.normalize(replacement), case.label, "relight"
                else:
                    rendered = render_scene(image, self.targets, self.seed, case.item)
                    if rendered.reason == "accepted":
                        return self.normalize(rendered.image), case.label, "protected_scene"
                tensor, label = self.base[index]
                return tensor, label, "v8_fallback"
        tensor, label = self.base[index]
        return tensor, label, "v8_recipe"


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, path, seconds, clock=time.monotonic):
        self.path, self.limit, self.clock = path, seconds, clock
        self.previous = read_json(path)["active_seconds"] if path.is_file() else 0.0
        self.started = self.last_saved = clock()

    @property
    def elapsed(self):
        return self.previous + self.clock() - self.started

    def save(self):
        write_json(self.path, {"active_seconds": self.elapsed})
        self.last_saved = self.clock()

    def check(self):
        if self.elapsed >= self.limit:
            raise BudgetExhausted("continuation time budget reached")
        if self.clock() - self.last_saved >= 30:
            self.save()


def train_one(model, optimizer, dataset, args, seed, budget):
    import torch

    model.train()
    batches = data_loader(dataset, args.batch_size, args.workers, seed)
    accum = EFFECTIVE_BATCH // args.batch_size
    optimizer.zero_grad(set_to_none=True)
    counts, seen, loss, steps = Counter(), 0, 0.0, 0
    for index, (images, labels, kinds) in enumerate(batches):
        budget.check()
        if index % accum == 0:
            group = accumulation_size(len(dataset) - seen, args.batch_size, accum)
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = torch.nn.functional.cross_entropy(model(images), labels, reduction="sum")
        (value / group).backward()
        seen += len(labels)
        loss += value.detach().item()
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite continuation loss")
        counts.update(kinds)
        if (index + 1) % accum == 0 or seen == len(dataset):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    if seen != len(dataset) or steps != math.ceil(seen / EFFECTIVE_BATCH):
        raise RuntimeError("Training epoch did not cover the complete manifest")
    return {"loss": loss / seen, "images": seen, "optimizer_steps": steps, "augmentation_counts": dict(counts)}


def selection(metrics, reference):
    clean, hard = metrics["accuracy"]["clean"], metrics["hard_accuracy"]
    clean_gap = clean - reference["accuracy"]["clean"]
    hard_gap = hard - reference["hard_accuracy"]
    eligible = clean_gap >= -CLEAN_DROP - 1e-9 and hard_gap >= -HARD_DROP - 1e-9
    return {"eligible": eligible, "score": .4 * clean + .6 * hard,
            "clean_delta_pp": 100 * clean_gap, "hard_delta_pp": 100 * hard_gap,
            "clear_collapse": clean < .80 or hard_gap < -.15}


def choose_epoch(history, requested=None):
    allowed = [row for row in history if row["selection"]["eligible"]]
    if requested is not None:
        allowed = [row for row in allowed if row["epoch"] == requested]
    return max(allowed, key=lambda row: (row["selection"]["score"], -row["loss"], -row["epoch"])) if allowed else None


def early_stop(history):
    if history and history[-1]["selection"]["clear_collapse"]:
        return "severe_regression"
    if len(history) >= 3:
        # 至少 0.1 pp 才重設 patience，避免近乎相同的分數一直延長訓練。
        best, last_gain = -math.inf, 0
        for row in history:
            if row["selection"]["eligible"] and row["selection"]["score"] > best + .001:
                best, last_gain = row["selection"]["score"], row["epoch"]
        if history[-1]["epoch"] - last_gain >= PATIENCE:
            return "early_stopping"
    return None
