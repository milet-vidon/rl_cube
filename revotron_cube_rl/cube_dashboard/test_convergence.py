import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import convergence, inference, quality, training


class ConvergenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.runs = Path(tmp.name) / "runs"
        self.home = self.runs / "contact_hold90_long_v2"
        self.rd = self.home / "round_001"
        self.fit = self.rd / "fit"
        self.fit.mkdir(parents=True)

    def test_joint_is_an_intersection_and_requires_verified_receipts(self):
        lanes = [
            (True, True, "teacher_wrist_endpoint"), (True, False, "teacher_wrist_endpoint"),
            (True, True, "planner_rejected"), (False, True, "teacher_wrist_endpoint")
        ]
        data = dict(role="validate", cube_edge_m=.09, num_envs=4, seeds=list(range(4)), plan_sha256="c"*64,
                    per_lane=[dict(seed=i, ever_hold=True, ending_hold=e, ever_whole_turn=t,
                                   teacher_executed_left_steps=0, end_reason=reason)
                              for i, (e,t,reason) in enumerate(lanes)])
        dest = self.rd / "validate/wave_0000_0004/sample_test"
        dest.mkdir(parents=True)
        report = dest / "report.json"
        report.write_text(json.dumps(data))
        mark = self.rd / "validate/accepted_0000.json"
        mark.write_text(json.dumps(dict(width=4, dest=str(dest), plan_sha="c"*64,
                                       report_sha=hashlib.sha256(report.read_bytes()).hexdigest())))
        (self.rd / "validation_summary.json").write_text(json.dumps(
            dict(envs=4, held=4, still_held_at_end=3, whole_turn=3, mean_controlled_hold_s=0.0)))
        def row():
            return quality.quality_history(self.home, [(1,self.rd)])[0]
        self.assertEqual((row()["joint_pass"],row()["endpoint_held"]), (1,2))
        report.write_text(report.read_text() + " ")
        self.assertIsNone(row()["joint_pass"])
        self.assertEqual(row()["detail_status"], "invalid_evidence")

    def test_incomplete_fit_and_unpublished_models_stay_null(self):
        result = self.fit / "result.json"
        def row():
            rows = [{}]
            convergence.add_fit_trends(self.home, [(1,self.rd)], rows, training._load, {})
            return rows[0]
        result.write_text(json.dumps(dict(completed=False, best_metrics={"phase_balanced_normalized_mse":0.5})))
        self.assertIsNone(row()["fit_mse"])
        result.write_text(json.dumps(dict(completed=True, best_metrics={"phase_balanced_normalized_mse":0.125})))
        with patch.object(convergence, "_drift", side_effect=AssertionError("unpublished checkpoint")):
            self.assertEqual(row()["fit_mse"], 0.125)
            self.assertIsNone(row()["parameter_delta_pct"])

    def test_drift_checks_digest_parent_and_excludes_buffers_and_optimizer(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch not installed")
        def save(name, initial, net_value, parent_sha):
            path = self.fit / name
            state = {key: torch.tensor([float(net_value)]) for key in convergence._LAYER_KEYS}
            state.update({key: torch.tensor([float(initial)]) for key in ("mean","std","lower","upper","center","scale")})
            torch.save(dict(metadata=dict(schema="contact_hold_success_demo_state_bc_v1",
                    warm_start_sha256=inference.R15_SHA,parent_checkpoint_sha256=parent_sha),
                    model=state, physical90_optimizer={"dummy":torch.tensor([float(initial)])}), path)
            return inference._identity(path, convergence._LIMIT), hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(inference,"RUNS",self.runs):
            parent = save("parent.pt", 1, 2, "a"*64)
            child = save("child.pt", 5000, 3, parent[1])
            self.assertAlmostEqual(convergence._relative_change(parent, child), 50.)
            unrelated = save("other.pt", 0, 3, "a"*64)
            with self.assertRaises(ValueError):
                convergence._relative_change(parent, unrelated)
            with self.assertRaises(inference.InferenceUnavailable):
                convergence._relative_change(parent, (child[0], "d"*64))


if __name__ == "__main__":
    unittest.main()
