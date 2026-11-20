"""共同的 ConvNeXt 部分微調、資料預算與實際更新計數。"""

import time
from collections import defaultdict

from classification_v2 import transforms_v2
from v4.geometry import seeded
from v8.data import TrainingImages
from v8.experiment import init_cuda, train_epoch
from v8.models import SOURCES, make_model, official_weight
from v12.model import SOURCE as BASE_SOURCE
from v12.model import make_base, weights_path

ARMS = ("tiny", "base")
WEIGHTS = {"tiny": SOURCES["convnext_tiny"], "base": BASE_SOURCE}
EFFECTIVE_BATCH = 64
PER_CLASS = 128
BUDGET = {"smoke": (2, 120), "pilot": (100, 720), "extend": (100, 720)}
HEAD_LR = 1e-3
STAGE_LR = 1e-4
WEIGHT_DECAY = 1e-4


def manifests(training, classes, seed):
    if len(classes) * PER_CLASS != BUDGET["pilot"][0] * EFFECTIVE_BATCH:
        raise ValueError("V13 budget requires the original 50 classes")
    pools = defaultdict(list)
    for case in training:
        pools[case.label].append(case)
    chosen = {}
    for label in range(len(classes)):
        pool = sorted(pools[label], key=lambda case: case.item)
        if len(pool) < 2 * PER_CLASS:
            raise ValueError(f"V13 needs {2 * PER_CLASS} training images for {classes[label]}")
        chosen[label] = seeded(seed, f"v13:class:{label}").sample(pool, 2 * PER_CLASS)
    # 交錯類別再讓共同的訓練 loader 打散。
    ordered = [rows[i] for i in range(2 * PER_CLASS) for rows in chosen.values()]
    end = len(classes) * PER_CLASS
    smoke_count = BUDGET["smoke"][0] * EFFECTIVE_BATCH
    return {"smoke": ordered[:smoke_count], "pilot": ordered[:end], "extend": ordered[end:]}


def make_arm(arm, classes, directory, local=None, from_pretrained=False):
    if arm == "tiny":
        path = official_weight("convnext_tiny", directory, local) if from_pretrained else None
        return make_model("convnext_tiny", classes, path)
    if arm != "base":
        raise ValueError(f"Unknown V13 model arm: {arm}")
    path = weights_path(directory, local) if from_pretrained else None
    return make_base(classes, path)


def configure_last_stage(model):
    import torch
    from torchvision.ops import StochasticDepth

    model.requires_grad_(False)
    stage, head = model.features[-1], model.classifier
    for module in model.modules():
        if isinstance(module, StochasticDepth):
            module.p = 0.0
    stage.requires_grad_(True)
    head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": list(head.parameters()), "lr": HEAD_LR},
         {"params": list(stage.parameters()), "lr": STAGE_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    return [stage, head], optimizer


class BudgetExceeded(RuntimeError):
    def __init__(self, count, seconds):
        super().__init__(f"V13 time cap reached after {count} updates ({seconds:.1f}s)")
        self.count, self.seconds = count, seconds


def counted_epoch(model, modules, optimizer, scaler, dataset, args, seed, planned, time_cap,
                  observe=None, clock=time.perf_counter):
    if len(dataset) != planned * EFFECTIVE_BATCH or args.batch_size * args.accum != EFFECTIVE_BATCH:
        raise ValueError("V13 effective batch or optimizer budget differs")
    steps = 0
    started = clock()

    def on_step(_optimizer, _args, _kwargs):
        nonlocal steps
        steps += 1
        if observe is not None:
            observe()
        elapsed = clock() - started
        if elapsed > time_cap:
            raise BudgetExceeded(steps, elapsed)

    with optimizer.register_step_post_hook(on_step):
        loss = train_epoch(model, modules, optimizer, scaler, dataset, args, seed)
    if steps != planned:
        raise RuntimeError(f"V13 completed {steps} optimizer updates, expected {planned}")
    return loss, steps


def checkpoint_stats(arm, model, modules, optimizer, data, info, seed, args, phase):
    import torch

    planned, seconds = BUDGET[phase]
    basic = transforms_v2()[0]
    dataset = TrainingImages(info.root, data, basic, info.targets, info.augmentation, seed)
    args.accum, args.amp = EFFECTIVE_BATCH // args.batch_size, "bf16"
    smoke = phase == "smoke"
    probes = {}
    observe = None
    if smoke:
        sentinels = {"stem": next(model.features[0].parameters()),
                     "last_stage": next(model.features[-1].parameters()),
                     "head": model.classifier[-1].weight}
        before = {key: param.detach().clone() for key, param in sentinels.items()}
        gradients_ok = {key: True for key in ("last_stage", "head")}

        def observe():
            for key in gradients_ok:
                grad = sentinels[key].grad
                gradients_ok[key] &= grad is not None and bool(torch.isfinite(grad).all() & (grad != 0).any())

    # worker 數為 0 時，兩個架構初始化消耗的 CPU RNG 也不影響資料增強。
    init_cuda(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss, steps = counted_epoch(model, modules, optimizer, scaler, dataset, args, seed, planned, seconds, observe)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if elapsed > seconds:
        raise BudgetExceeded(steps, elapsed)
    if smoke:
        probes = {f"{key}_max_delta": (param.detach() - before[key]).abs().max().item()
                  for key, param in sentinels.items()}
        probes["gradients_ok"] = gradients_ok
        if (not all(gradients_ok.values()) or probes["stem_max_delta"] != 0
                or any(not 0 < probes[f"{key}_max_delta"] < float("inf") for key in gradients_ok)):
            raise RuntimeError(f"V13 {arm} smoke parameter update failed: {probes}")
    return {"loss": loss, "images": len(dataset), "optimizer_steps": steps, "seconds": elapsed,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **probes}
