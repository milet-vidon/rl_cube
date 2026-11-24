"""Independent entry-boundary tests; mock trained loader, never construct Isaac."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import evaluate_state_residual_v6 as entry
from test_phase_residual_v5 import NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EntryBoundaryPeerReview(unittest.TestCase):
    def invoke(self, change=None):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            candidate, baseline, qualification = (root / n for n in ('candidate.pt', 'base.pt', 'qualification.json'))
            candidate.write_bytes(b'mocked candidate bytes')
            baseline.write_bytes(b'mocked frozen baseline bytes')
            source_hashes = entry.train_phase_gated_residual_v6.source_hashes()
            base_meta = {'joint_names': list(NAMES), 'output_indices': list(range(28))}
            q = {
                'schema': entry.train_phase_gated_residual_v6.QUALIFICATION_SCHEMA,
                'eligible_for_bc_transition_supervision': True,
                'eligible_for_rl_episode_training': False,
                'pure_student_episode_success_claim': False,
                'independent_generalization_claim': False,
                'original_result_files_modified': False,
                'source_sha256': source_hashes,
                'model_schema': entry.phase_gated_residual_policy_v6.SCHEMA,
                'feature_schema': entry.phase_gated_residual_policy_v6.FEATURE_SCHEMA,
                'gate_contract': deepcopy(entry.phase_gated_residual_policy_v6.GATE_CONTRACT),
                'frozen_baseline_checkpoint_sha256': sha(baseline),
                'physics_family_sha256': 'fixture family, not physical evidence',
                'residual_bound_joint_names': list(NAMES[:28]),
                'residual_bound_rad': [.001] * 28,
            }
            metadata = base_meta | {'feature_schema': entry.phase_residual_policy_v5.FEATURE_SCHEMA}
            args = SimpleNamespace(policy='v6', source=root/'source', output=root/'output',
                checkpoint=candidate, expected_checkpoint_sha256=sha(candidate), baseline=baseline,
                qualification=qualification, dataset=[root/'mixture', root/'prefix'], seed=9501,
                name='peer host fixture', max_steps=3500)
            if change:
                change(q, metadata, args)
            qualification.write_text(json.dumps(q))
            args.qualification_sha256 = sha(qualification)
            metadata.setdefault('reviewed_qualification_sha256', args.qualification_sha256)
            preflight = {'names': list(NAMES), 'left': list(range(28))}
            with ExitStack() as stack:
                stack.enter_context(patch.object(entry, 'source_preflight', return_value=preflight))
                stack.enter_context(patch.object(entry, 'load_baseline', return_value=({'metadata': base_meta}, sha(baseline))))
                stack.enter_context(patch.object(entry, 'checkpoint_preflight', return_value={'mock': 'not simulator validation'}))
                loader = stack.enter_context(patch.object(entry, 'load_checkpoint', return_value=(SimpleNamespace(mean=torch.zeros(284)), metadata)))
                result = entry.prepare(args)
                identity = result[-1]
                self.assertEqual(loader.call_args.kwargs['expected_physics_family'], q['physics_family_sha256'])
                for name, digest in identity['source_sha256'].items():
                    self.assertEqual(sha(args.output/'evaluation_source_snapshot'/name), digest)
                self.assertEqual(sha(args.output/'checkpoints/evaluated_policy.pt'), args.expected_checkpoint_sha256)
                self.assertEqual(sha(args.output/'reviewed_transition_qualification.json'), args.qualification_sha256)
                self.assertEqual(identity['left_teacher_position_action_steps'], 0)
                return identity

    def test_valid_entry_binds_snapshots_and_qualification_without_simulator(self):
        identity = self.invoke()
        self.assertEqual(identity['policy'], 'v6')
        self.assertIn('state_residual_eval_runtime_v6.py', identity['source_sha256'])
        self.assertIn('eval_prefix_dataset_v6.py', identity['source_sha256'])
        self.assertIn('phase_gated_residual_policy_v6.py', identity['source_sha256'])
        self.assertIn('train_phase_gated_residual_v6.py', identity['source_sha256'])

    def test_changed_qualification_semantics_and_gate_fail_before_runtime(self):
        changes = [
            lambda q, m, a: q.update(eligible_for_rl_episode_training=True),
            lambda q, m, a: q.update(original_result_files_modified=True),
            lambda q, m, a: q.update(source_sha256={}),
            lambda q, m, a: q.update(gate_contract={}),
            lambda q, m, a: q.update(frozen_baseline_checkpoint_sha256='wrong'),
        ]
        for number, change in enumerate(changes):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.invoke(change)

    def test_mismatched_checkpoint_qualification_or_joint_order_rejected(self):
        changes = [
            lambda q, m, a: m.update(reviewed_qualification_sha256='different reviewed document'),
            lambda q, m, a: m.update(output_indices=[1, 0] + list(range(2, 28))),
            lambda q, m, a: setattr(a, 'expected_checkpoint_sha256', 'wrong pinned bytes'),
        ]
        for number, change in enumerate(changes):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.invoke(change)

    def test_duplicate_dataset_path_rejected(self):
        with self.assertRaises(ValueError):
            self.invoke(lambda q, m, a: setattr(a, 'dataset', [a.dataset[0], a.dataset[0]]))


if __name__ == '__main__':
    unittest.main()
