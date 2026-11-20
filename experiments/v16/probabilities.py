"""完整機率快取與批次完成後的即時進度。"""

import hashlib
import io
import math
import sys
import time

import numpy as np

from v3.storage import json_bytes, publish


class Progress:
    def __init__(self, name, total, stream=None, clock=time.monotonic):
        self.name, self.total, self.done = name, total, 0
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.start = self.previous = clock()
        self.interactive = self.stream.isatty()
        self.render(0)

    def render(self, elapsed):
        pct = self.done / self.total
        eta = (elapsed / self.done * (self.total - self.done) if self.done else None)
        eta_str = "--" if eta is None else f"{eta:.1f}s"
        blocks = int(20 * pct)
        bar = "#" * blocks + "-" * (20 - blocks)
        self.stream.write(("\r" if self.interactive else "") +
                          f"{self.name} [{bar}] {self.done:,}/{self.total:,} {pct:6.2%} "
                          f"elapsed {elapsed:.1f}s ETA {eta_str}" +
                          ("\n" if not self.interactive or self.done == self.total else ""))
        self.stream.flush()

    def advance(self, images):
        if images < 1 or self.done + images > self.total:
            raise ValueError("V16 progress exceeded the fixed image count")
        self.done += images
        now = self.clock()
        if self.interactive or self.done == self.total or now - self.previous >= 2:
            self.render(now - self.start)
            self.previous = now


def validate_matrix(p, rows, classes):
    if (p.shape != (rows, classes) or p.dtype != np.float32 or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid V16 probability matrix")


def array_digest(p):
    return hashlib.sha256(p.tobytes(order="C")).hexdigest()


def read_cache(path, ids, classes, metadata):
    with np.load(path, allow_pickle=False) as data:
        if data["identity"].item() != json_bytes(metadata).decode() or data["ids"].tolist() != ids:
            raise ValueError(f"V16 cache identity or order differs: {path}")
        p, seconds = data["probabilities"], float(data["seconds"])
        validate_matrix(p, len(ids), classes)
        if data["sha256"].item() != array_digest(p) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"V16 cache checksum or timing differs: {path}")
    return p, seconds


def cache_or_infer(path, ids, classes, metadata, get_batches, progress_name):
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("V16 requires unique, nonempty requested IDs")
    if path.is_file():
        p, seconds = read_cache(path, ids, classes, metadata)
        print(f"{progress_name} | complete cache {len(ids):,}/{len(ids):,}", flush=True)
        return p, seconds, True
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress = Progress(progress_name, len(ids))
    matrix = np.empty((len(ids), classes), dtype=np.float32)
    for batch_ids, view_probs, _, _ in get_batches():
        end = progress.done + len(batch_ids)
        if (not batch_ids or list(batch_ids) != ids[progress.done:end]
                or view_probs.shape != (10, len(batch_ids), classes)):
            raise ValueError("V16 inference batch ID order or view count differs")
        # 原 V14 推論先沿視角平均 Softmax，保留 FP32。
        mean = view_probs.mean(axis=0)
        validate_matrix(mean, len(batch_ids), classes)
        matrix[progress.done:end] = mean
        progress.advance(len(batch_ids))
    if progress.done != len(ids):
        raise ValueError("Incomplete V16 cache")
    seconds = time.monotonic() - started
    out = io.BytesIO()
    np.savez_compressed(out, identity=json_bytes(metadata).decode(), ids=np.asarray(ids), probabilities=matrix,
                        sha256=array_digest(matrix), seconds=seconds)
    publish(path, out.getvalue())
    return matrix, seconds, False


def fuse(a, b, weight):
    return np.float32(weight) * a + np.float32(1 - weight) * b


def replay_mismatches(matrix, ids, archive):
    predicted = matrix.argmax(axis=1)
    return [(item, int(archive[item]), int(predicted[i])) for i, item in enumerate(ids)
            if predicted[i] != archive[item]]
