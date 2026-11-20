import numpy as np
import pytest

import classification_v18 as v18


def test_select_pseudo_threshold_and_agreement():
    pa = np.array([[.95, .05], [.6, .4], [.2, .8], [.55, .45]], dtype=np.float32)
    pb = np.array([[.90, .10], [.1, .9], [.1, .9], [.55, .45]], dtype=np.float32)
    fused, labels, conf, keep = v18.select_pseudo(pa, pb, .8, True)
    assert labels.tolist() == [0, 0, 1, 0]
    assert keep.tolist() == [True, False, True, False]
    _, _, _, loose = v18.select_pseudo(pa, pb, .5, False)
    assert loose.tolist() == [True, True, True, True]
    assert np.allclose(fused.sum(axis=1), 1)


def test_learning_rate_schedule():
    assert v18.learning_rate(1, 1, 1e-5) == 1e-5
    assert v18.learning_rate(1, 2, 1e-5) == pytest.approx(1e-5)
    assert v18.learning_rate(2, 2, 1e-5) == pytest.approx(5e-6)


def test_arguments_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = v18.arguments(["prepare"])
    assert args.threshold == .9 and args.epochs == 2 and args.pseudo_repeat == 2
    with pytest.raises(SystemExit):
        v18.arguments(["predict", "--smoke"])
    with pytest.raises(SystemExit):
        v18.arguments(["prepare", "--output", "runs/v17"])
