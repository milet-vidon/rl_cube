"""Host-tested DAgger transaction core; no simulator launcher or frozen edits.

Runtime gates and the adapter to the existing single-Isaac evaluation loop are
deliberately explicit dependencies. This module does not certify that adapter.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'left_contact_dagger_actual_state_transition_v1'
STATE_FIELDS = ('q', 'joint_velocity', 'applied_target', 'cube_position',
                'cube_quaternion_wxyz', 'cube_linear_velocity', 'cube_angular_velocity',
                'finger_contact_impulses', 'finger_contact_points')
ACTOR_FIELDS = STATE_FIELDS + ('phase', 'phase_progress', 'previous_executed_source')
TEACHER_FIELDS = ('name', 'mode', 'index', 'tick', 'wait', 'lost', 'hold',
                  'contact_streak', 'primary_streak', 'failure_reason', 'accepted',
                  'tilt_error_deg', 'right_index', 'right_tick', 'right_hold',
                  'targets', 'reference', 'nominal_target', 'hold_gains_set',
                  'hold_center', 'hold_reference_palm')
TEACHER_CHILDREN = ('base', 'finger_arrival', 'right_arrival')
SUPPORTED_PHASES = ('ready', 'preshape', 'lower', 'approach', 'squeeze', 'primary_lift',
                    'elbow_outward', 'tilt_90', 'wrist_reconfigure', 'tilt_hold')
PHYSICS_FIELDS = ('environment_class', 'config_sha256', 'assets_sha256', 'source_sha256',
                  'cube_constraint_topology', 'damping', 'armature', 'friction', 'solver_options')


def physics_fingerprint(identity):
    if any(key not in identity for key in PHYSICS_FIELDS):
        raise ValueError('Incomplete physics identity')
    if any(identity[key] is None or identity[key] == {} or identity[key] == '' for key in PHYSICS_FIELDS):
        raise ValueError('Physics identity fields must be explicitly populated')
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def require_same_physics(identities):
    fingerprints = {physics_fingerprint(identity) for identity in identities}
    if len(fingerprints) != 1:
        raise ValueError('Do not pool different physics identities into one training distribution')
    return next(iter(fingerprints))


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('Unsupported teacher-state value: '+type(value).__name__)


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def safe_abort_evidence(value):
    """Keep corrupt/nonfinite failure evidence without failing the journal."""
    invalid = []
    def visit(item, path):
        try:
            item = plain(item)
        except Exception as exc:
            invalid.append(path)
            return {'unsupported_evidence_type': type(item).__name__, 'conversion_error': str(exc)}
        if isinstance(item, float) and not math.isfinite(item):
            invalid.append(path)
            return {'nonfinite_evidence': repr(item)}
        if isinstance(item, dict): return {k: visit(v, path+'.'+k) for k, v in item.items()}
        if isinstance(item, list): return [visit(v, path+'['+str(i)+']') for i,v in enumerate(item)]
        return item
    return visit(value, '$'), invalid


def vector(value, size):
    result = plain(value)
    if len(result) != size or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in result):
        raise ValueError('Invalid finite vector length '+str(size))
    return list(result)


def actor_observation(observation):
    """Exclude expert memory/labels, but retain privileged simulator observations.

    This is for a state-based teacher policy, not a vision/deployment actor.
    """
    result = {key: deepcopy(plain(observation[key])) for key in ACTOR_FIELDS}
    for key, count in (('q', 56), ('joint_velocity', 56), ('applied_target', 56),
                       ('cube_position', 3), ('cube_quaternion_wxyz', 4)):
        result[key] = vector(result[key], count)
    for key in ('cube_linear_velocity', 'cube_angular_velocity'):
        result[key] = vector(result[key], 3)
    if not 0 <= result['phase_progress'] <= 1:
        raise ValueError('Invalid actor progress')
    canonical(result)
    return result


def teacher_snapshot(teacher):
    result = {'class': type(teacher).__module__+'.'+type(teacher).__name__}
    for name in TEACHER_FIELDS:
        if hasattr(teacher, name):
            result[name] = plain(getattr(teacher, name))
    # Plans can be large: hash them and preserve execution cursors above. The
    # runner must separately archive the full plan whenever its hash changes.
    for name in ('path', 'phases', 'right_phases'):
        value = getattr(teacher, name, None)
        if value is not None:
            result[name+'_sha256'] = hashlib.sha256(canonical(value).encode()).hexdigest()
    for name in TEACHER_CHILDREN:
        child = getattr(teacher, name, None)
        if child is not None:
            result[name] = teacher_snapshot(child)
    return result


def synchronize_executed_memory(teacher, names, applied_target):
    """Explicit bookkeeping adaptation, never hidden as unmodified oracle state.

    Preserve cached IK goals and modulo-three tick. Only fields denoting last
    applied targets are replaced by commands actually executed by the learner.
    Existing arrival reference/anchor definitions initialize missing finger
    memory explicitly, including first activation. This changes the original
    nominal-fallback initialization and is logged as an adaptation.
    """
    targets = vector(applied_target, len(names))
    changes = []
    def visit(node, path):
        memory = getattr(node, 'targets', None)
        if isinstance(memory, dict):
            reference = getattr(node, 'reference', {})
            anchors = getattr(node, 'anchors', {})
            side = 'right' if path.endswith('.right_arrival') else 'left'
            initialize = {n for n in names if n in reference and any(n.startswith(side+'_'+f+'_') for f in anchors)}
            for i, name in enumerate(names):
                key = '_applied_'+name
                if key in memory or name in initialize:
                    old = float(memory[key]) if key in memory else None; new = targets[i]
                    if old != new:
                        changes.append({'path': path+'.targets.'+key, 'before': old, 'after': new})
                    memory[key] = new
        for name in TEACHER_CHILDREN:
            child = getattr(node, name, None)
            if child is not None:
                visit(child, path+'.'+name)
    visit(teacher, 'teacher')
    return changes


class FrozenTeacherOracle:
    """Query the original teacher once on the live learner-state environment.

    read_state() must return actual state, never a reference trajectory row.
    The caller owns teacher construction and required gate/planner integration.
    """
    def __init__(self, teacher, names, read_state, decode_target, decoder_id,
                 schedule_resolver, memory_mode='preserve', control_change_validator=None):
        if memory_mode not in ('preserve', 'sync_actual_applied'):
            raise ValueError('Unknown expert memory mode')
        self.teacher = teacher; self.names = list(names); self.read_state = read_state
        if not decoder_id or not callable(decode_target) or not callable(schedule_resolver):
            raise ValueError('Native target decoder and same-action schedule resolver are required')
        self.decode_target = decode_target; self.decoder_id = decoder_id
        self.schedule_resolver = schedule_resolver; self.control_change_validator = control_change_validator
        self.memory_mode = memory_mode; self.query_count = 0; self.last_native_action = None

    def query(self, observation):
        before = deepcopy(plain(self.read_state()))
        for key in STATE_FIELDS+('oracle_env_inputs', 'control_settings'):
            if canonical(before[key]) != canonical(observation[key]):
                raise ValueError('Teacher query state differs from actor pre-state: '+key)
        teacher_before = teacher_snapshot(self.teacher)
        changes = []
        if self.memory_mode == 'sync_actual_applied':
            changes = synchronize_executed_memory(self.teacher, self.names, observation['applied_target'])
        self.teacher.finger_contacts = deepcopy(observation['finger_contact_impulses'])
        if hasattr(self.teacher, 'contact_points'):
            self.teacher.contact_points = deepcopy(observation['finger_contact_points'])
        synced = teacher_snapshot(self.teacher)
        native_action = self.teacher.action()
        self.last_native_action = native_action.detach().clone() if hasattr(native_action, 'detach') else deepcopy(native_action)
        target = vector(self.decode_target(native_action), len(self.names))
        action = plain(native_action)
        if len(action) == 1 and isinstance(action[0], list):
            action = action[0]
        action = vector(action, len(self.names))
        after = deepcopy(plain(self.read_state()))
        for key in STATE_FIELDS+('oracle_env_inputs',):
            if canonical(before[key]) != canonical(after[key]):
                raise RuntimeError('Teacher action unexpectedly mutated physics state: '+key)
        teacher_after = teacher_snapshot(self.teacher)
        changed_settings = canonical(before['control_settings']) != canonical(after['control_settings'])
        if changed_settings and (self.control_change_validator is None or not self.control_change_validator(before['control_settings'], after['control_settings'], teacher_before, teacher_after)):
            raise RuntimeError('Unapproved teacher control-setting mutation')
        schedule = plain(self.schedule_resolver(observation, teacher_before, teacher_after))
        if schedule.get('phase') not in SUPPORTED_PHASES or not 0 <= schedule.get('phase_progress', -1) <= 1:
            raise ValueError('Oracle v1 supports pickup through wrist hold only; invalid same-action schedule')
        if teacher_after.get('name') is not None and schedule['phase'] != teacher_after['name']:
            raise ValueError('Actor schedule does not match the action just labelled by teacher')
        self.query_count += 1
        return {'requested_target': target, 'normalized_action': action,
                'teacher_state_before_sync': teacher_before,
                'teacher_state_after_sync': synced,
                'teacher_state_after_query': teacher_after, 'public_same_action_schedule': schedule,
                'native_target_decoder_id': self.decoder_id,
                'oracle_env_inputs': before['oracle_env_inputs'],
                'control_settings_before': before['control_settings'], 'control_settings_after': after['control_settings'],
                'teacher_runtime_control_setting_change': changed_settings,
                'execution_memory_mode': self.memory_mode, 'execution_memory_changes': changes,
                'query_index': self.query_count-1,
                'failure_reason': getattr(self.teacher, 'failure_reason', None)}


def validate_gate(report, stage_required=False):
    # The runtime bridge must call original audits and report their source hash.
    if not isinstance(report.get('original_audit_source_sha256'), dict) or not report['original_audit_source_sha256']:
        raise ValueError('Missing original gate provenance')
    for name in ('robot_safety', 'joint_limits', 'cube_integrity', 'no_pose_write'):
        if report.get(name) is not True:
            raise ValueError('Gate failed: '+name)
    if stage_required and report.get('stage_gate_qualified') is not True:
        raise ValueError('Stage gate missing or failed')


class DaggerTransitionCollector:
    """Two-phase transaction: prepare a decision, then record actual execution.

    No step/reset/pose write is hidden here. The external single-Isaac loop must
    use execution_packet() through the unchanged environment step. Requested
    target fields are labels; teacher execution uses its original native action.
    """
    def __init__(self, oracle, journal, episode, run_provenance, left_indices):
        self.oracle = oracle; self.journal = journal; self.episode = str(episode)
        self.provenance = plain(run_provenance); self.pending = None; self.next_step = 0
        self.terminal = False; self.interventions = 0
        if not self.provenance.get('student_checkpoint_sha256') or not self.provenance.get('source_sha256'):
            raise ValueError('Missing learner/source provenance')
        self.provenance['policy_observation_scope'] = 'privileged_simulator_state_not_vision_actor'
        self.provenance['physics_identity_sha256'] = physics_fingerprint(self.provenance['physics_identity'])
        self.left_indices = list(left_indices)
        if len(self.left_indices) != 28 or len(set(self.left_indices)) != 28 or any(i < 0 or i >= 56 for i in self.left_indices):
            raise ValueError('Expected exactly 28 distinct left joint indices')
        actual_left = {i for i,name in enumerate(self.oracle.names) if name.startswith('left_') or name.endswith('_L')}
        if set(self.left_indices) != actual_left:
            raise ValueError('Left indices must match the actual left-hand/arm joint schema')

    def prepare(self, observation, student_target, pre_gates, *, execute_source,
                selection_reason, selection_probability, selection_draw, phase_transition=False):
        if self.pending is not None or self.terminal:
            raise RuntimeError('Previous transition incomplete or episode terminal')
        frozen = deepcopy(plain(observation))
        try:
            return self._prepare_once(frozen, student_target, pre_gates, execute_source=execute_source,
                selection_reason=selection_reason, selection_probability=selection_probability,
                selection_draw=selection_draw, phase_transition=phase_transition)
        except Exception as exc:
            self.abort(str(exc), frozen)
            raise

    def abort(self, reason, observation):
        self.terminal = True
        evidence, invalid = safe_abort_evidence({'observation': observation, 'pending_transition': self.pending})
        self.journal.write(canonical({'schema': SCHEMA, 'record_type': 'abort', 'episode': self.episode,
            'step': self.next_step, 'reason': str(reason), 'run_provenance': self.provenance,
            **evidence, 'invalid_evidence_fields': invalid,
            'pure_student_episode_claim': False})+'\n'); self.journal.flush()
        self.pending = None

    def _prepare_once(self, observation, student_target, pre_gates, *, execute_source,
                      selection_reason, selection_probability, selection_draw, phase_transition):
        validate_gate(pre_gates, stage_required=phase_transition)
        if execute_source not in ('student', 'teacher') or not selection_reason:
            raise ValueError('Selection source/reason must be explicit')
        if not 0 <= selection_probability <= 1 or not 0 <= selection_draw <= 1:
            raise ValueError('Invalid logged selection probability/draw')
        if not callable(student_target):
            raise TypeError('student_target must be an actor callable, not a precomputed stale-phase action')
        expert = self.oracle.query(observation)
        if expert['failure_reason']:
            raise RuntimeError('Expert reports failure; refusing collection: '+str(expert['failure_reason']))
        schedule = expert['public_same_action_schedule']
        obs = actor_observation({**observation, **schedule})
        # The callback receives only observations/public schedule, after the
        # oracle has resolved the exact phase of this action once. No labels.
        left_proposal = vector(student_target(deepcopy(obs)), 28)
        proposal = list(expert['requested_target'])
        for i, value in zip(self.left_indices, left_proposal): proposal[i] = value
        selected = proposal if execute_source == 'student' else expert['requested_target']
        self.pending = {'schema': SCHEMA, 'episode': self.episode, 'step': self.next_step,
                        'actor_observation_pre': obs, 'student_proposed_target': proposal,
                        'student_proposed_left_target': left_proposal, 'right_controller': 'original_teacher',
                        'expert_supervision': expert, 'selected_requested_target': selected,
                        'executed_source': execute_source, 'teacher_intervention': execute_source == 'teacher',
                        'selection_reason': selection_reason, 'selection_probability': selection_probability,
                        'selection_draw': selection_draw, 'phase_transition': bool(phase_transition),
                        'pre_gates': deepcopy(pre_gates), 'run_provenance': self.provenance,
                        'sample_is_on_student_rollout': execute_source == 'student',
                        'pure_student_episode_claim': False}
        return deepcopy(self.pending)

    def execution_packet(self):
        if self.pending is None:
            raise RuntimeError('No prepared transition')
        # Keep the original native normalized action for teacher execution and
        # for the right-hand slice in learner execution. Never target-roundtrip
        # teacher commands through Python doubles or a new action conversion.
        action = self.oracle.last_native_action
        action = action.detach().clone() if hasattr(action, 'detach') else deepcopy(action)
        return {'native_teacher_normalized_action': action,
                'executed_source': self.pending['executed_source'],
                'left_indices': list(self.left_indices),
                'student_left_target': list(self.pending['student_proposed_left_target'])}

    def complete(self, observation_post, post_gates, *, stage_complete=False, failure_reason=None):
        if self.pending is None:
            raise RuntimeError('No prepared transition')
        try:
            post = actor_observation(observation_post)
        except Exception as exc:
            self.abort(str(exc), observation_post)
            raise
        gate_failure = None
        try:
            validate_gate(post_gates, stage_required=stage_complete)
        except ValueError as exc:
            gate_failure = str(exc)
        record = self.pending
        record.update({'actor_observation_post': post, 'actual_applied_target': post['applied_target'],
                       'post_gates': deepcopy(post_gates), 'stage_complete': bool(stage_complete),
                       'failure_reason': failure_reason or gate_failure,
                       'terminal': bool(failure_reason or gate_failure or stage_complete)})
        self.interventions += int(record['teacher_intervention'])
        record['teacher_intervention_count_so_far'] = self.interventions
        # Only the original final episode auditor may certify episode success.
        # An isolated segment can follow an unrecorded teacher warmup, so this
        # transaction core never calls it a pure-student episode.
        record['pure_student_episode_claim'] = False
        record['unassisted_collected_segment_completed'] = bool(stage_complete and not record['failure_reason'] and self.interventions == 0)
        self.journal.write(canonical(record)+'\n'); self.journal.flush()
        self.pending = None; self.next_step += 1; self.terminal = record['terminal']
        return deepcopy(record)
