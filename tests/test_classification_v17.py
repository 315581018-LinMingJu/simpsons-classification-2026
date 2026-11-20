import io
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import classification_v17 as v17
from v3.reports import read_reference
from v3.storage import file_sha256, json_bytes
from v2.data_checks import read_json
from v8.data import CONDITIONS
from v17.weights import average


class WeightTests(unittest.TestCase):
    def test_average_uses_only_model_tensors_and_preserves_dtype(self):
        a = {"w": torch.tensor([1., 3.], dtype=torch.float32), "step": torch.tensor(9), "flag": torch.tensor(True),
             "buf": torch.tensor([65504., -2.], dtype=torch.float16)}
        b = {**a, "w": torch.tensor([3., 7.]), "buf": torch.tensor([65504., 8.], dtype=torch.float16)}
        state, stats = average(a, b, a)
        torch.testing.assert_close(state["w"], torch.tensor([2., 5.]))
        torch.testing.assert_close(state["buf"], torch.tensor([65504., 3.], dtype=torch.float16))
        self.assertEqual(stats, {"averaged_float": 2, "copied_nonfloat": 2})
        for name in ("step", "flag"):
            self.assertTrue(torch.equal(state[name], a[name]))
            self.assertNotEqual(state[name].data_ptr(), a[name].data_ptr())
        torch.testing.assert_close(a["w"], torch.tensor([1., 3.]))

    def test_incompatible_or_non_finite_tensors_are_rejected(self):
        a = {"w": torch.tensor([1.]), "step": torch.tensor(9)}
        for invalid, error in [
            ({"w": a["w"]}, "names"),
            ({**a, "w": torch.tensor([1., 2.])}, "Incompatible"),
            ({**a, "w": a["w"].double()}, "Incompatible"),
            ({**a, "w": torch.tensor([float("nan")])}, "Non-finite"),
            ({**a, "w": torch.tensor([float("inf")])}, "Non-finite"),
            ({**a, "step": torch.tensor(10)}, "Non-floating"),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                average(a, invalid, a)


class ScreenTests(unittest.TestCase):
    def test_paired_report_and_exact_guardrail_boundary(self):
        labels = np.tile([0, 1], 100)
        correct = np.eye(2, dtype=np.float32)[labels]
        wrong = correct[:, ::-1].copy()

        def trial(drops):
            ref = {c: correct for c in CONDITIONS}
            candidate = {c: correct.copy() for c in CONDITIONS}
            for c, n in drops.items():
                candidate[c][:n] = wrong[:n]
            return v17.screen_result({"a3": ref, "a4": ref, "v17": candidate}, labels, 42)

        edge = trial({"clean": 4, "context": 20})
        self.assertEqual(edge["decision"], "allow_candidate")
        self.assertAlmostEqual(edge["gaps_vs_a3"]["hard_pp"], -3)
        self.assertAlmostEqual(edge["gaps_vs_a3"]["any_hard_condition_pp"], -10)
        self.assertEqual(edge["average_vs_a4"]["conditions"]["clean"]["losses"], 4)
        self.assertEqual(trial({"clean": 6})["guardrail_breaches"], ["clean_pp"])
        self.assertEqual(trial({"negative": 22})["guardrail_breaches"], ["any_hard_condition_pp"])
        self.assertEqual(trial({"context": 16, "impulse": 16})["guardrail_breaches"], ["hard_pp"])


class ReferenceTests(unittest.TestCase):
    def test_a80_b20_archive_bound_to_v14_source_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            csv = candidate / "submission_v16.csv"
            csv.write_text("id,character\n09,x\n2,y\n")
            test = SimpleNamespace(provenance={"test_image_signature": "test"})
            plan = {"protocol": "simpsons-v16-v1", "v14": {"screen_signature": "screen"},
                    "test": test.provenance, "classes": ["x", "y"],
                    "arms": {"a": {"epoch": 3, "checkpoint_sha256": "a3"},
                             "b": {"epoch": 4, "checkpoint_sha256": "b4"}}}
            plan_path = root / "prepare.json"
            plan_path.write_bytes(json_bytes(plan))
            validation = root / "validation.json"
            weights = {"weight_a": .8, "weight_b": .2}
            validation.write_bytes(json_bytes({**weights, "prepare_sha256": file_sha256(plan_path)}))
            summary = {**weights, "prepare_sha256": file_sha256(plan_path), "validation_sha256": file_sha256(validation),
                       "replay_mismatches": {"a": 0, "b": 0}, "images": 2, "submission_sha256": file_sha256(csv),
                       "checkpoint_sha256": {"a": "a3", "b": "b4"}}
            (candidate / "summary.json").write_bytes(json_bytes(summary))
            args = SimpleNamespace(v16_candidate=candidate)
            data = (args, test, ["09", "2"], ["x", "y"], "a3", plan["v14"])
            labels, _, _ = v17.v16_reference(*data)
            self.assertEqual(labels, {"09": 0, "2": 1})
            with self.assertRaisesRegex(ValueError, "source"):
                v17.v16_reference(*data[:-1], {"screen_signature": "different"})
            csv.write_text("id,character\n09,y\n2,y\n")
            with self.assertRaisesRegex(ValueError, "SHA"):
                v17.v16_reference(*data)


class PipelineTests(unittest.TestCase):
    def test_cpu_pipeline_cache_gate_and_candidate_deltas(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.callback(os.chdir, Path.cwd())
            os.chdir(root)
            output = root / "runs/v17"
            checkpoints, rows, sums = {}, {}, {}
            settings = {"arm": "a"}
            model = torch.nn.Linear(2, 2)
            for epoch in (3, 4):
                key = f"a{epoch}"
                checkpoints[key] = root / f"{key}.pt"
                with torch.no_grad():
                    model.weight.fill_(epoch)
                    model.bias.zero_()
                row = {"epoch": epoch}
                torch.save({"identity": settings, "epoch": epoch, "training": row,
                            "optimizer": {"value": 100 * epoch}, "model": model.state_dict()}, checkpoints[key])
                sums[key] = file_sha256(checkpoints[key])
                rows[epoch] = {**row, "checkpoint_sha256": sums[key]}
            source = SimpleNamespace(classes=["x", "y"], checkpoints=checkpoints, rows=rows, settings=settings,
                                     identity={"checkpoint_sha256": sums, "screen_labels": [0, 1],
                                               "screen_batch_size": 8, "test_batch_size": 16},
                                     info=SimpleNamespace(root=root, screen=[], inputs={"seed": 1}),
                                     test=SimpleNamespace(samples=[]), ids={"screen": ["s1", "s2"], "test": ["09", "2"]},
                                     a3_labels={"09": 0, "2": 0}, v16_labels={"09": 1, "2": 0}, references={})
            stack.callback(torch.set_num_threads, torch.get_num_threads())
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(v17, "prepare_inputs", return_value=source))
            stack.enter_context(patch.object(v17, "make_model", side_effect=lambda *_: torch.nn.Linear(2, 2)))
            stack.enter_context(patch.object(v17.v16, "verify_images"))
            stack.enter_context(patch.object(v17, "init_cuda"))
            stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda self: self))
            stack.enter_context(patch.object(v17, "data_loader", side_effect=lambda dataset, *_: dataset))
            stack.enter_context(patch.object(v17, "TEST_COUNT", 2))

            def configured_inference(_model, _dataset, *, warmup):
                ids = source.ids["test" if warmup else "screen"]
                for item, p in zip(ids, ([.9, .1], [.1, .9]), strict=True):
                    yield [item], np.tile(np.asarray(p, dtype=np.float32), (10, 1, 1)), None, 0

            infer = stack.enter_context(patch.object(v17, "infer_view_batches", side_effect=configured_inference))
            self.assertEqual(v17.main(["prepare"]), 0)
            saved = torch.load(output / "averaged.pt", weights_only=True)
            self.assertEqual(set(saved), {"identity", "model", "tensors"})
            torch.testing.assert_close(saved["model"]["weight"], torch.full((2, 2), 3.5))
            with self.assertRaisesRegex(SystemExit, "validate"):
                v17.main(["predict", "--allow-imagenet"])
            self.assertEqual(v17.main(["validate", "--allow-imagenet"]), 0)
            self.assertEqual(infer.call_count, 3 * len(CONDITIONS))
            self.assertEqual(v17.main(["validate"]), 0)
            self.assertEqual(infer.call_count, 3 * len(CONDITIONS))
            report = output / "validation.json"
            original = report.read_bytes()
            report.write_bytes(json_bytes({**read_json(report), "decision": "manually_changed"}))
            with self.assertRaisesRegex(ValueError, "verified Screen"):
                v17.main(["predict", "--allow-imagenet"])
            report.write_bytes(original)
            self.assertEqual(v17.main(["predict", "--allow-imagenet"]), 0)
            self.assertEqual(infer.call_count, 3 * len(CONDITIONS) + 1)
            candidate = output / "candidate"
            csv = candidate / "submission_v17.csv"
            summary = read_json(candidate / "summary.json")
            self.assertEqual(read_reference(csv, ["09", "2"], ["x", "y"]), {"09": 0, "2": 1})
            self.assertEqual((summary["changed_vs_a3"], summary["changed_vs_v16"]), (1, 2))
            self.assertEqual(summary["submission_sha256"], file_sha256(csv))
            self.assertEqual({k: file_sha256(path) for k, path in checkpoints.items()}, sums)

    def test_output_cannot_target_existing_runs(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            v17.arguments(["prepare", "--output", "runs/v16"])


if __name__ == "__main__":
    unittest.main()
