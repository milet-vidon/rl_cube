from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from dagger_collection_core_v1 import FrozenTeacherOracle, DaggerTransitionCollector, actor_observation, synchronize_executed_memory


NAMES = ['left_middle_MCP_joint']+['left_test_'+str(i) for i in range(1,28)]+['right_test_'+str(i) for i in range(28)]


def observation():
    return {'q': [0.]*56, 'joint_velocity': [0.]*56, 'applied_target': [0.]*56,
            'cube_position': [.3,.2,1.], 'cube_quaternion_wxyz': [1.,0.,0.,0.],
            'cube_linear_velocity': [0.]*3, 'cube_angular_velocity': [0.]*3,
            'finger_contact_impulses': {'left': {'middle': 0.}, 'right': {}},
            'finger_contact_points': [], 'phase': 'squeeze', 'phase_progress': .9,
            'previous_executed_source': 'teacher', 'secret_expert_targets': [99.]*56}


def physics_identity():
    return {'environment_class': 'test_only', 'config_sha256': 'test', 'assets_sha256': {'test':'test'},
            'source_sha256': {'test':'test'}, 'cube_constraint_topology': 'test', 'damping': {'test':0.},
            'armature': {'test':0.}, 'friction': {'test':0.}, 'solver_options': {'test':'test'}}


def gate():
    return {'robot_safety': True, 'joint_limits': True, 'cube_integrity': True,
            'no_pose_write': True, 'stage_gate_qualified': True,
            'original_audit_source_sha256': {'unit_test_stub': 'test_only_not_runtime_certificate'}}


class Teacher:
    def __init__(self, state):
        self.env = SimpleNamespace(scales=[1.]*56)
        self.state = state; self.tick = 0; self.failure_reason = None; self.name = 'squeeze'
        self.base = SimpleNamespace(finger_arrival=SimpleNamespace(tick=7, targets={
            '_applied_left_middle_MCP_joint': .03, 'left_middle_MCP_joint': .2},
            reference={'left_middle_MCP_joint': .2}, anchors={'middle': {}}))
    def action(self):
        self.tick += 1
        action = [0.]*56; action[0] = .005
        return [action]


class DaggerTests(unittest.TestCase):
    def setUp(self):
        self.state = observation(); self.teacher = Teacher(self.state)
        self.state['oracle_env_inputs'] = {'metrics': {'height_m': .1, 'cube_table_contact': False, 'linear_speed': 0., 'angular_speed': 0.}, 'contact_impulse': [0.,0.]}
        self.state['control_settings'] = {'joint_stiffness': [10.]*56, 'joint_damping': [8.]*56}
        self.oracle = FrozenTeacherOracle(self.teacher, NAMES, lambda: self.state,
            lambda action: [a+d for a,d in zip(self.state['applied_target'],action[0])],
            'test_native_decoder', lambda obs,before,after: {'phase':after['name'], 'phase_progress':obs['phase_progress']},
            'sync_actual_applied')
        self.log = io.StringIO()
        self.collector = DaggerTransitionCollector(self.oracle, self.log, 'test', {
            'student_checkpoint_sha256': 'test_checkpoint', 'source_sha256': {'teacher': 'test'},
            'physics_identity': physics_identity()}, list(range(28)))

    def prepare(self, source='student', **kw):
        return self.collector.prepare(self.state, lambda obs: [.002]*28, gate(), execute_source=source,
                                      selection_reason='logged_test_schedule', selection_probability=.5,
                                      selection_draw=.25, **kw)

    def test_actor_cannot_receive_expert_labels_or_hidden_state(self):
        self.assertNotIn('secret_expert_targets', actor_observation(self.state))
        decision = self.prepare()
        self.assertNotIn('expert_supervision', decision['actor_observation_pre'])
        self.assertEqual(decision['expert_supervision']['requested_target'][0], .005)
        self.assertEqual(decision['selected_requested_target'][0], .002)

    def test_memory_sync_preserves_ik_intent_tick_and_logs_changes(self):
        decision = self.prepare()
        memory = self.teacher.base.finger_arrival
        self.assertEqual(memory.targets['_applied_left_middle_MCP_joint'], 0.)
        self.assertEqual(memory.targets['left_middle_MCP_joint'], .2)
        self.assertEqual(memory.tick, 7)
        self.assertEqual(len(decision['expert_supervision']['execution_memory_changes']), 1)

    def test_actual_executed_target_not_proposal_is_logged_and_reused(self):
        self.prepare(); self.state['applied_target'] = [.001]*56
        record = self.collector.complete(self.state, gate())
        self.assertEqual(record['actual_applied_target'][0], .001)
        self.assertEqual(record['student_proposed_target'][0], .002)
        self.prepare()
        self.assertEqual(self.teacher.base.finger_arrival.targets['_applied_left_middle_MCP_joint'], .001)

    def test_teacher_intervention_never_becomes_pure_student_success(self):
        self.prepare('teacher'); self.state['applied_target'][0] = .005
        record = self.collector.complete(self.state, gate(), stage_complete=True)
        self.assertTrue(record['teacher_intervention']); self.assertFalse(record['pure_student_episode_claim'])
        self.assertEqual(record['teacher_intervention_count_so_far'], 1)

    def test_no_duplicate_query_and_prestate_mismatch_rejected(self):
        self.prepare()
        with self.assertRaises(RuntimeError): self.prepare()
        self.assertEqual(self.teacher.tick, 1)
        altered = deepcopy(self.state); altered['q'][0] = 1.
        with self.assertRaises(ValueError): self.oracle.query(altered)

    def test_failed_pre_gate_or_missing_phase_gate_prevents_query(self):
        bad = gate(); bad['robot_safety'] = False
        with self.assertRaises(ValueError):
            self.collector.prepare(self.state, [.002]*56, bad, execute_source='student', selection_reason='test', selection_probability=0., selection_draw=0.)
        self.assertEqual(self.teacher.tick, 0)
        self.assertTrue(self.collector.terminal)
        self.assertEqual(json.loads(self.log.getvalue())['record_type'], 'abort')
        self.setUp()
        bad = gate(); bad.pop('stage_gate_qualified')
        with self.assertRaises(ValueError):
            self.collector.prepare(self.state, [.002]*56, bad, execute_source='student', selection_reason='test', selection_probability=0., selection_draw=0., phase_transition=True)
        self.assertEqual(self.teacher.tick, 0)

    def test_failed_post_gate_recorded_terminal_without_claim_or_reset(self):
        self.prepare(); bad = gate(); bad['joint_limits'] = False
        record = self.collector.complete(self.state, bad)
        self.assertTrue(record['terminal']); self.assertFalse(record['pure_student_episode_claim'])
        self.assertIn('joint_limits', record['failure_reason'])
        self.assertEqual(len(self.log.getvalue().splitlines()), 1)
        with self.assertRaises(RuntimeError): self.prepare()

    def test_teacher_physics_write_is_rejected(self):
        def action():
            self.state['cube_position'][0] += 1.
            return [[0.]*56]
        self.teacher.action = action
        with self.assertRaisesRegex(RuntimeError, 'mutated physics'): self.oracle.query(self.state.copy())

    def test_missing_arrival_memory_is_initialized_explicitly(self):
        del self.teacher.base.finger_arrival.targets['_applied_left_middle_MCP_joint']
        self.state['applied_target'][0] = .012
        decision = self.prepare()
        self.assertEqual(self.teacher.base.finger_arrival.targets['_applied_left_middle_MCP_joint'], .012)
        self.assertIsNone(decision['expert_supervision']['execution_memory_changes'][0]['before'])

    def test_native_teacher_action_and_right_hand_are_preserved(self):
        decision = self.prepare()
        self.assertEqual(decision['student_proposed_target'][28:], decision['expert_supervision']['requested_target'][28:])
        packet = self.collector.execution_packet()
        self.assertEqual(packet['native_teacher_normalized_action'], [[.005]+[0.]*55])
        self.assertEqual(len(packet['student_left_target']), 28)

    def test_same_action_phase_resolution_before_actor(self):
        original = self.teacher.action
        def action():
            result = original(); self.teacher.name = 'primary_lift'; return result
        self.teacher.action = action
        decision = self.prepare()
        self.assertEqual(decision['actor_observation_pre']['phase'], 'primary_lift')

    def test_oracle_env_inputs_and_control_changes_are_checked(self):
        altered = deepcopy(self.state); altered['oracle_env_inputs']['metrics']['height_m'] = .2
        with self.assertRaises(ValueError): self.oracle.query(altered)
        def action():
            self.state['control_settings']['joint_stiffness'][0] = 20.
            return [[0.]*56]
        self.teacher.action = action
        with self.assertRaisesRegex(RuntimeError, 'control-setting'): self.oracle.query(self.state)

    def test_physics_identity_is_required_and_changes_fingerprint(self):
        from dagger_collection_core_v1 import physics_fingerprint, require_same_physics
        first = physics_identity(); changed = deepcopy(first); changed['damping'] = {'cube': 'D6_linear'}
        self.assertNotEqual(physics_fingerprint(first), physics_fingerprint(changed))
        with self.assertRaisesRegex(ValueError, 'Do not pool'): require_same_physics([first, changed])
        del changed['friction']
        with self.assertRaises(ValueError): physics_fingerprint(changed)

    def test_explicitly_allowed_original_gain_change_is_recorded(self):
        def action():
            self.state['control_settings']['joint_stiffness'][0] = 20.
            return [[0.]*56]
        self.teacher.action = action
        self.oracle.control_change_validator = lambda before, after, tb, ta: before['joint_stiffness'][0] == 10. and after['joint_stiffness'][0] == 20.
        decision = self.prepare()
        self.assertTrue(decision['expert_supervision']['teacher_runtime_control_setting_change'])

    def test_center_phase_is_rejected_and_abort_is_retained(self):
        self.teacher.name = 'left_center_transport'
        with self.assertRaisesRegex(ValueError, 'wrist hold only'): self.prepare()
        self.assertTrue(self.collector.terminal)
        self.assertEqual(json.loads(self.log.getvalue())['record_type'], 'abort')

    def test_json_schema_accepts_transition_and_abort(self):
        import jsonschema
        from pathlib import Path
        schema=json.loads((Path(__file__).parent/'left_contact_dagger_schema_v1.json').read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        self.prepare();record=self.collector.complete(self.state,gate())
        jsonschema.validate(record,schema)
        self.setUp();self.teacher.name='left_center_transport'
        with self.assertRaises(ValueError): self.prepare()
        jsonschema.validate(json.loads(self.log.getvalue()),schema)

    def test_nonfinite_pre_and_post_are_journaled_as_terminal_abort(self):
        self.state['q'][0] = float('nan')
        with self.assertRaises(ValueError): self.prepare()
        record = json.loads(self.log.getvalue()); self.assertTrue(record['invalid_evidence_fields'])
        self.assertTrue(self.collector.terminal)
        self.setUp(); self.prepare(); self.state['q'][0] = float('inf')
        with self.assertRaises(ValueError): self.collector.complete(self.state, gate())
        record = json.loads(self.log.getvalue()); self.assertTrue(record['invalid_evidence_fields'])
        self.assertTrue(self.collector.terminal); self.assertIsNone(self.collector.pending)

    def test_execution_packet_cannot_mutate_cached_expert_action(self):
        self.prepare(); packet = self.collector.execution_packet()
        packet['native_teacher_normalized_action'][0][0] = 999.
        self.assertEqual(self.oracle.last_native_action[0][0], .005)


if __name__ == '__main__': unittest.main()
