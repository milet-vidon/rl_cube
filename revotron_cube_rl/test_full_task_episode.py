import io
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

from full_task_episode import FullTaskEpisode


class FakeEnv:
    """A counter/return-contract fake; never models contact physics."""
    def __init__(self, outcomes=(), maximum=20):
        self.auto_reset = False
        self.num_envs = 1
        self.max_episode_length = maximum
        self.episode_length_buf = [0]
        self.cube = SimpleNamespace(pose_reset_count=2)
        self.episodes, self.successes = 7, 3
        self.outcomes = list(outcomes)
        self.actions = []
        self.metrics = {}

    def step(self, action):
        self.actions.append(action)
        self.episode_length_buf[0] += 1
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        success, drop = outcome.get('success', False), outcome.get('drop', False)
        timeout = self.episode_length_buf[0] >= self.max_episode_length
        done = success or drop or timeout
        reward = 5. if success else -.002
        self.metrics = {'success': success, 'drop': drop, 'reward': reward,
                        'terms': {'terminal': float(done), 'timeout': float(timeout),
                                  'drop': float(drop), 'success_bonus': 5. * success,
                                  'potential': 0. if done else .3}}
        if done:
            self.episodes += 1
            self.successes += int(success)
        return {}, [reward], [outcome.get('raw_done', done)], {}


def build(outcomes=(), maximum=20, target=15):
    env, journal = FakeEnv(outcomes, maximum), io.StringIO()
    episode = FullTaskEpisode(env, journal,
        {'task_id': f'contact_turn_nZ_{target}', 'target_degrees': target,
         'source_sha256': {'environment.py': 'frozen', 'original_audit.py': 'frozen'},
         'physics_domain': 'explicit_fake_for_host_contract_only'})
    return episode, env, journal


def advance(episode, env, *, failure=None):
    index = episode.steps
    action = [[.001] * 56]
    status = episode.step(action, phase='opposition_hold', step_id=index)
    row = {'step': index, 'phase': 'opposition_hold', **deepcopy(env.metrics)}
    row['post_step_audit_added_after_generic_trajectory_append'] = True
    status = episode.finish_step(row, failure=failure,
        audit={'original_audits_completed': True, 'original_report': {'qualified': failure is None}})
    return status


def successful_result(episode):
    return {'full_episode': True, 'execution_interrupted': False, 'failure': None,
            'dynamic_grasp_passed': True, 'measured': {'accepted': True},
            'wrist': {'accepted': True}, 'assembled_hold_s': 3.,
            'center_transport_passed': True, 'cube_pose_writes_during_control': 0,
            'face_turn_validated': episode.target == 90, 'steps': episode.steps,
            'push': {'center': {'passed': True}, 'executed': True, 'qualified': True,
                     'opposition_grasp_passed': True, 'turn_started': True,
                     'turn_hold_s': 3., 'target_degrees': episode.target, 'failure': None}}


class FullTaskTests(unittest.TestCase):
    def records(self, journal):
        return [json.loads(line) for line in journal.getvalue().splitlines()]

    def test_repeated_pickup_done_is_one_milestone_with_raw_reward_and_counts(self):
        episode, env, journal = build([{'success': True}] * 3)
        for _ in range(3):
            status = advance(episode, env)
            self.assertFalse(status['done'])
            self.assertEqual(status['reward'], 0.)
            self.assertFalse(status['eligible_for_ppo'])
        records = [r for r in self.records(journal) if r['record_type'] == 'transition']
        self.assertEqual([r['pickup_milestone_first_observed'] for r in records], [True, False, False])
        self.assertEqual([r['raw']['reward_returned'] for r in records], [5.] * 3)
        self.assertEqual((env.episodes, env.successes), (10, 6))
        self.assertEqual(status['task_episodes'], 0)
        self.assertEqual(status['raw_pickup_done_frames'], 3)
        self.assertTrue(records[-1]['row']['post_step_audit_added_after_generic_trajectory_append'])

    def test_drop_locks_before_finish_and_does_not_step_again(self):
        episode, env, journal = build([{'drop': True}])
        self.assertTrue(episode.step([[0.] * 56], phase='turn', step_id=0)['terminated'])
        episode.finish_step({'step': 0, 'phase': 'turn'}, audit={'original_audits_completed': True})
        with self.assertRaises(RuntimeError): episode.step([[0.] * 56], phase='turn', step_id=1)
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(episode.status()['reason'], 'cube_drop')

    def test_timeout_priority_over_pickup_and_final_success(self):
        episode, env, journal = build([{'success': True}], maximum=1)
        status = advance(episode, env)
        self.assertTrue(status['truncated'])
        self.assertEqual(status['reason'], 'timeout')
        self.assertFalse(episode.conclude(successful_result(episode))['success'])
        self.assertEqual(status['task_episodes'], 1)
        self.assertEqual(len([r for r in self.records(journal) if r['record_type'] == 'episode_final']), 1)

    def test_safety_failure_blocks_next_physics_and_stage(self):
        episode, env, journal = build()
        status = advance(episode, env, failure='robot_safety_gate')
        self.assertTrue(status['terminated'])
        self.assertEqual(status['reason'], 'robot_safety_gate')
        with self.assertRaises(RuntimeError): episode.step([[0.] * 56], phase='next_stage', step_id=1)
        self.assertEqual(len(env.actions), 1)

    def test_pending_transaction_blocks_next_step(self):
        episode, env, journal = build()
        episode.step([[0.] * 56], phase='turn', step_id=0)
        with self.assertRaises(RuntimeError): episode.step([[0.] * 56], phase='turn', step_id=1)
        self.assertEqual(len(env.actions), 1)
        final = self.records(journal)[-1]
        self.assertEqual(final['record_type'], 'episode_final')
        self.assertEqual(final['unfinished_transition']['step'], 0)

    def test_original_final_gate_required_and_target_scope_preserved(self):
        for target in (15, 30, 90):
            episode, env, journal = build(target=target)
            advance(episode, env)
            self.assertFalse(episode.status()['success'])
            status = episode.conclude(successful_result(episode))
            self.assertTrue(status['success'])
            self.assertTrue(status['terminated'])
            self.assertEqual(status['face_turn_validated'], target == 90)
            self.assertFalse(status['six_face_solution_validated'])
            self.assertEqual(status['task_successes'], 1)
            episode.conclude(successful_result(episode))
            self.assertEqual(episode.status()['task_successes'], 1)

    def test_missing_gate_wrong_target_or_pose_write_cannot_pass(self):
        for mutate in (lambda r: r['push'].update(qualified=False),
                       lambda r: r['push'].update(target_degrees=90),
                       lambda r: r['wrist'].update(accepted=False),
                       lambda r: r.update(cube_pose_writes_during_control=1),
                       lambda r: r.update(face_turn_validated=True)):
            episode, env, journal = build()
            advance(episode, env)
            result = successful_result(episode); mutate(result)
            self.assertFalse(episode.conclude(result)['success'])

    def test_no_step_abort_has_final_event_and_no_fake_transition(self):
        episode, env, journal = build()
        episode.abort('opposition_preflight_rejected', detail={'passed': False})
        self.assertEqual([r['record_type'] for r in self.records(journal)], ['episode_start', 'episode_final'])
        self.assertEqual(len(env.actions), 0)
        self.assertEqual(episode.status()['completed_control_steps'], 0)

    def test_unknown_raw_done_is_fail_closed_and_preserves_evidence(self):
        episode, env, journal = build([{'raw_done': True}])
        with self.assertRaises(ValueError): episode.step([[0.] * 56], phase='turn', step_id=0)
        self.assertTrue(episode.status()['done'])
        self.assertIn('inconsistent', episode.status()['reason'])
        self.assertTrue(self.records(journal)[-1]['unfinished_transition']['raw']['done_returned'])

    def test_nonfinite_action_fails_before_any_physics_and_is_archived(self):
        episode, env, journal = build()
        with self.assertRaises(ValueError): episode.step([[float('nan')] * 56], phase='turn', step_id=0)
        self.assertEqual(len(env.actions), 0)
        self.assertTrue(episode.status()['done'])
        self.records(journal)

    def test_horizon_mutation_or_external_step_or_reset_is_rejected(self):
        for mutate in (lambda e: setattr(e, 'max_episode_length', 100),
                       lambda e: e.episode_length_buf.__setitem__(0, 1),
                       lambda e: setattr(e.cube, 'pose_reset_count', 3)):
            episode, env, journal = build(); mutate(env)
            with self.assertRaises(ValueError): episode.step([[0.] * 56], phase='turn', step_id=0)
            self.assertEqual(len(env.actions), 0)

    def test_row_link_and_explicit_original_audit_completion_required(self):
        for row, audit in (({'step': 4, 'phase': 'turn'}, {'original_audits_completed': True}),
                           ({'step': 0, 'phase': 'turn'}, {})):
            episode, env, journal = build()
            episode.step([[0.] * 56], phase='turn', step_id=0)
            with self.assertRaises(ValueError): episode.finish_step(row, audit=audit)
            self.assertTrue(episode.status()['done'])

    def test_action_identity_is_preserved_and_no_reset_is_called(self):
        episode, env, journal = build()
        action = [[.00001] * 56]
        episode.step(action, phase='turn', step_id=0)
        self.assertIs(env.actions[0], action)
        self.assertEqual(env.cube.pose_reset_count, 2)

    def test_conclude_pending_is_failure_and_never_success(self):
        episode, env, journal = build()
        episode.step([[0.] * 56], phase='turn', step_id=0)
        with self.assertRaises(RuntimeError): episode.conclude(successful_result(episode))
        self.assertFalse(episode.status()['success'])


if __name__ == '__main__':
    unittest.main()
