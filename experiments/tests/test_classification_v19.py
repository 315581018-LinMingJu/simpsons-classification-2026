import pytest

import classification_v19 as v19


def test_arguments_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = v19.arguments(["prepare"])
    assert args.threshold == .9 and args.epochs == 2 and args.pseudo_repeat == 2
    assert str(args.v18_cache).endswith("runs/v18/cache/test/epoch_02.npz")
    assert args.v18_cache.parents[2].name == "v18"
    with pytest.raises(SystemExit):
        v19.arguments(["prepare", "--output", "runs/v18"])


def test_names_are_v19():
    import inspect
    text = inspect.getsource(v19)
    assert "submission_v19.csv" in text and "submission_v18.csv" in text
    assert v19.FORMAT == "simpsons-v19-pseudo-label-round2-v1"
