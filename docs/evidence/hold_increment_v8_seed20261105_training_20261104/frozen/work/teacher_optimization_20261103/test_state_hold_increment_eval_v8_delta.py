"""Independent host proof that V8 routing preserves the executed V7 protocol.

Synthetic states exercise real actor/float32 projection but are not simulator
success evidence. Qualification/checkpoint tests live in the entry peer suite.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import hold_increment_policy_v8 as policy_module
from test_hold_increment_policy_v8 import policy, force_modes
from test_state_zero_gated_eval_v7 import FakeRuntime, joint_safety
from state_hold_increment_eval_runtime_v8 import evaluate

ROOT = Path(__file__).resolve().parent


class V8RouteContracts(unittest.TestCase):
    def test_entire_runtime_is_only_import_feature_and_policy_route_delta(self):
        original = (ROOT / 'state_zero_gated_eval_runtime_v7.py').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         'caadc319e816bdb3c00b5a744b71fbaf7c5f60ffd2d8f1893593fe441767b66b')
        changed = (ROOT / 'state_hold_increment_eval_runtime_v8.py').read_text()
        changed = changed.replace(
            'from hold_increment_policy_v8 import EpisodeActor, features as v8_features',
            'from zero_gated_residual_policy_v7 import EpisodeActor, features as v7_features')
        changed = changed.replace("== 'v8'", "== 'v7'").replace("=='v8'", "=='v7'")
        changed = changed.replace('v8_features(', 'v7_features(')
        self.assertEqual(changed, original.decode())

    def test_main_execution_cleanup_and_cli_only_change_version(self):
        def body(filename):
            text = (ROOT / filename).read_text().replace("'v8'", "'v7'")
            return ast.dump(next(node for node in ast.parse(text).body
                                 if isinstance(node, ast.FunctionDef) and node.name == 'main'),
                            include_attributes=False)
        self.assertEqual(body('evaluate_state_hold_increment_v8.py'),
                         body('evaluate_state_zero_gated_v7.py'))

    def run_actor(self, mode, steps=3):
        torch.manual_seed(319)
        model = policy(); force_modes(model, mode)
        model.head.bias.data.fill_(.5)
        runtime = FakeRuntime(success_step=steps)
        runtime.args.policy = 'v8'
        runtime.teacher.name = 'squeeze'
        runtime.actor = policy_module.EpisodeActor(model, list(range(28)))
        runtime.actor.reset(runtime.args.name)

        def schedule(obs, before, after):
            runtime.current_schedule = {'phase': after['name'],
                                        'phase_progress': (71/84, 72/84, 1.)[len(runtime.rows)]}
            return dict(runtime.current_schedule)
        runtime.schedule_resolver = schedule
        journal = io.StringIO()
        with patch.dict('sys.modules', {'joint_safety': joint_safety}):
            summary = evaluate(runtime, journal, max_steps=10)
        rows = [json.loads(line) for line in journal.getvalue().splitlines()
                if json.loads(line)['record_type'] == 'executed_transition']
        return runtime, model, summary, rows

    def test_hold_has_one_query_one_advance_actual_previous_and_no_teacher_takeover(self):
        runtime, model, summary, rows = self.run_actor(policy_module.HOLD)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary['teacher_query_count'], 3)
        self.assertEqual(runtime.actor.next_step, 3)
        self.assertEqual(summary['left_teacher_position_action_steps'], 0)
        self.assertTrue(summary['original_wrist_gate_qualified'])
        for row in rows[1:]:
            expected = np.asarray(row['actor_observation_pre']['applied_target'][:28], np.float32)
            actual = np.asarray(row['predicted_left_target'], np.float32)
            np.testing.assert_array_equal(actual.view(np.int32), expected.view(np.int32))
            self.assertTrue(row['execution']['right_native_bits_preserved'])
            self.assertTrue(row['execution']['actual_projection_bits_verified'])
        self.assertFalse(summary['pure_student_episode_claim'])

    def test_update_actual_history_replays_bitwise_and_public_boundary_is_retained(self):
        runtime, model, summary, rows = self.run_actor(policy_module.UPDATE)
        replay = policy_module.EpisodeActor(model, list(range(28)))
        replay.reset(runtime.args.name)
        hidden = None
        for step, row in enumerate(rows):
            obs = row['actor_observation_pre'] | {'previous_executed_source': 'student'}
            actual = np.asarray(row['predicted_left_target'], np.float32)
            replayed = np.asarray(replay.act(obs, episode_id=runtime.args.name, step=step), np.float32)
            np.testing.assert_array_equal(actual.view(np.int32), replayed.view(np.int32))
            x = torch.from_numpy(policy_module.features(obs)).reshape(1, 1, -1)
            previous = torch.tensor(obs['applied_target'][:28], dtype=torch.float32).reshape(1, 1, 28)
            with torch.no_grad():
                _, hidden, _, diagnostic = model.forward_with_diagnostics(x, previous, hidden)
            if step == 0:
                np.testing.assert_array_equal(actual.view(np.int32),
                    diagnostic['baseline_proposal'].numpy().reshape(28).view(np.int32))
                self.assertFalse(diagnostic['public_phase_active'].any())
            else:
                self.assertTrue(diagnostic['public_phase_active'].all())
                mask = model.update_anchor_is_previous
                self.assertTrue(torch.equal(diagnostic['update_anchor'][..., mask], previous[..., mask]))
                self.assertTrue(torch.equal(diagnostic['update_anchor'][..., ~mask],
                                             diagnostic['baseline_proposal'][..., ~mask]))
            if step:
                self.assertEqual(obs['applied_target'], rows[step-1]['actor_observation_post']['applied_target'])
        self.assertEqual(summary['fully_audited_transition_steps'], 3)
        self.assertEqual(replay.next_step, 3)


if __name__ == '__main__':
    unittest.main()
