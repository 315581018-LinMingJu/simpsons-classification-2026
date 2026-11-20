import json

import numpy as np

import classification_v21 as v21


def fake_run(path, seed, classes):
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"classes": classes, "model": "m", "size": 224}))
    labels = np.arange(20) % 3
    def probs(n):
        p = rng.random((n, 3)).astype(np.float32)
        return p / p.sum(1, keepdims=True)
    np.savez(path / "val_probabilities.npz", labels=labels, clean=probs(20), testlike=probs(20))
    (path / "candidate_epoch_01").mkdir()
    ids = [str(i) for i in range(10791)]
    np.savez(path / "candidate_epoch_01/test_probabilities.npz", ids=np.asarray(ids), probabilities=probs(10791))


def test_ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    classes = ["a", "b", "c"]
    fake_run(tmp_path / "r1", 1, classes)
    fake_run(tmp_path / "r2", 2, classes)
    v21.main(["ensemble", str(tmp_path / "r1"), str(tmp_path / "r2"), "--output", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out/ensemble-2/summary.json").read_text())
    assert summary["images"] == 10791 and len(summary["per_run"]) == 2
    lines = (tmp_path / "out/ensemble-2/submission_v21.csv").read_text().splitlines()
    assert lines[0] == "id,character" and len(lines) == 10792
