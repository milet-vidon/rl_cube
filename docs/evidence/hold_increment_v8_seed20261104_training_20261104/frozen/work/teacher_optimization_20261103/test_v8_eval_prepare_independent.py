"""Independent V8 entry boundary checks; loader mocked, no model/sim claim."""
import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluate_state_hold_increment_v8 as entry
from test_state_hold_increment_eval_v8_prepare import V8PrepareContracts


class IndependentPrepare(unittest.TestCase):
    def test_frozen_sources_and_exact_seven_two_selection(self):
        root = Path(__file__).parent
        for name, expected in {
            'state_hold_increment_eval_runtime_v8.py': '8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c',
            'evaluate_state_hold_increment_v8.py': '54a9f1ae958cacd55434397e72b04c10743c8a16c1be24d74f3518010ed497c3',
            'train_hold_increment_v8.py': '055f8f0d6ed724c2932a2effa59d402c2e22472491519f9a1b09fb553e5a42cb',
        }.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), expected)
        t = entry.train_hold_increment_v8
        self.assertEqual(len(t.RUN_NAMES), 8)
        self.assertEqual(t.ZERO_BOUNDARY_RUNS, tuple(n for i, n in enumerate(t.RUN_NAMES) if i != 1))
        self.assertEqual(t.HOLD_BOUNDARY_RUNS, (t.RUN_NAMES[4], t.RUN_NAMES[5]))
        self.assertEqual(t.HOLD_BOUNDARY_JOINTS, ('left_ring_MCP_joint', 'left_ring_PIP_joint', 'left_ring_DIP_joint'))

    def test_additional_resealed_prepare_metadata_fail_closed(self):
        fixture = V8PrepareContracts()
        edits = [
            lambda q,m,a,p: m['final_cpu_sequential_metrics'].update(step499_reviewed_zero_histories_bitwise_baseline=False),
            lambda q,m,a,p: m['final_cpu_sequential_metrics'].update(inference='batched_not_streamed'),
            lambda q,m,a,p: m.update(source_sha256={}),
            lambda q,m,a,p: q.update(update_anchor_is_previous=[int(v) for v in q['update_anchor_is_previous']]),
            lambda q,m,a,p: m.update(right28_learned=True),
            lambda q,m,a,p: m.update(eligible_for_rl_episode_training=True),
            lambda q,m,a,p: q.update(branch_coverage=q['branch_coverage'][:-1]),
            lambda q,m,a,p: q.update(eligible_for_development_bc_transition_supervision=False),
        ]
        for index, change in enumerate(edits):
            with self.subTest(index=index), self.assertRaises(ValueError):
                fixture.invoke(change)

    def test_missing_snapshotted_training_dependency_is_rejected(self):
        fixture = V8PrepareContracts()
        original = entry.train_hold_increment_v8.source_hashes()
        with patch.object(entry.train_hold_increment_v8, 'source_hashes',
                          return_value=original | {'unlisted_dependency.py': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'lacks exact evaluation snapshot'):
                fixture.invoke()

    def test_loader_cannot_be_called_in_unqualified_mode(self):
        tree = ast.parse(Path(entry.__file__).read_text())
        prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
        calls = [n for n in ast.walk(prepare) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'load_checkpoint']
        self.assertEqual(len(calls), 1)
        self.assertNotIn('require_offline_qualified', {k.arg for k in calls[0].keywords})
        import inspect
        signature = inspect.signature(entry.train_hold_increment_v8.load_checkpoint)
        self.assertIs(signature.parameters['require_offline_qualified'].default, True)


if __name__ == '__main__': unittest.main()
