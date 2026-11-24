"""Explicit single-target task episodes over the unchanged pickup environment.

Teacher/BC collection only: task reward is always zero, never a PPO reward.
The base pickup reward, done and counters remain untouched diagnostic evidence.
Every physics step is a transaction completed after the original stage audits.
This module neither resets the scene nor changes controls or gate thresholds.
"""
from copy import deepcopy
import json
import math

SCHEMA = 'single_face_full_task_episode_v1'


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('Unsupported episode evidence: ' + type(value).__name__)


def frozen(value):
    result = deepcopy(plain(value))
    json.dumps(result, allow_nan=False)
    return result


def scalar(value):
    value = plain(value)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
        raise ValueError('Expected one finite scalar')
    return value


def flag(value):
    value = scalar(value)
    if value not in (0, 1):
        raise ValueError('Expected an explicit boolean/zero-one flag')
    return bool(value)


def diagnostic(value):
    """Keep invalid evidence on failure without writing non-standard JSON."""
    try:
        value = plain(value)
    except Exception as error:
        return {'unserializable_type': type(value).__name__, 'error': str(error)}
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_nonfinite': repr(value)}
    if isinstance(value, dict):
        return {k: diagnostic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [diagnostic(v) for v in value]
    return value


class FullTaskEpisode:
    """One fresh environment episode with explicit task-level termination.

    The caller must replace *all three* full-task driver step sites, finish each
    after its original audits, and consume status()['done'] at every stage.
    Planning/gate failures without a physics step call abort(), not finish_step().
    All final accepted targets are development subtasks, not six-face solving.
    """

    def __init__(self, env, journal, provenance, episode_id='0'):
        self.env, self.journal = env, journal
        self.provenance = frozen(provenance)
        self.target = self.provenance.get('target_degrees')
        if self.target not in (15, 30, 90) or isinstance(self.target, bool):
            raise ValueError('Declare one full opposed target: 15, 30 or 90 degrees')
        if not self.provenance.get('task_id') or not self.provenance.get('source_sha256'):
            raise ValueError('Frozen task ID and original source hashes are required')
        if not self.provenance.get('physics_domain'):
            raise ValueError('Declare the actual physics domain; do not mix datasets')
        if env.auto_reset or int(env.num_envs) != 1:
            raise ValueError('Requires one environment with auto_reset=False')
        if int(scalar(env.episode_length_buf)) != 0:
            raise ValueError('Construct immediately after reset, before any physics step')
        self.maximum_steps = int(env.max_episode_length)
        if self.maximum_steps <= 0:
            raise ValueError('Declare a positive finite horizon before starting')
        self.episode_id = str(episode_id)
        self.initial_reset_count = int(env.cube.pose_reset_count)
        self.initial_raw_counts = self._raw_counts()
        self.pending = None
        self.steps = 0
        self.terminated = self.truncated = self.success = False
        self.reason = None
        self.pickup_milestone_step = None
        self.raw_success_frames = self.raw_done_frames = 0
        self.task_episodes = self.task_successes = 0
        self.final_written = False
        self._emit({'record_type': 'episode_start', 'provenance': self.provenance,
                    'maximum_control_steps': self.maximum_steps,
                    'initial_raw_counters': self.initial_raw_counts,
                    'task': self.status()})

    def _raw_counts(self):
        return {'episodes': int(self.env.episodes), 'successes': int(self.env.successes)}

    def _emit(self, event):
        record = {'schema': SCHEMA, 'episode_id': self.episode_id, **diagnostic(event)}
        self.journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        self.journal.flush()

    def status(self):
        return {'task_id': self.provenance['task_id'], 'target_degrees': self.target,
                'scope': 'single_target_face_turn_development',
                'terminated': self.terminated, 'truncated': self.truncated,
                'done': self.terminated or self.truncated, 'success': self.success,
                'reason': self.reason, 'reward': 0.0, 'eligible_for_ppo': False,
                'task_episodes': self.task_episodes, 'task_successes': self.task_successes,
                'completed_control_steps': self.steps, 'step_audit_pending': self.pending is not None,
                'pickup_milestone_step': self.pickup_milestone_step,
                'raw_pickup_success_frames': self.raw_success_frames,
                'raw_pickup_done_frames': self.raw_done_frames,
                'face_turn_validated': bool(self.success and self.target == 90),
                'six_face_solution_validated': False}

    def _stop(self, reason, *, truncated=False, success=False):
        # The first terminal decision is immutable. Timeout is evaluated first
        # in step(), before any post-step or final gate can claim success.
        if not self.status()['done']:
            self.truncated = bool(truncated)
            self.terminated = not self.truncated
            self.success = bool(success)
            self.reason = str(reason)
            self.task_episodes += 1
            self.task_successes += int(self.success)

    def _guard(self):
        if self.status()['done'] or self.final_written:
            raise RuntimeError('Task episode terminal; no further physics step allowed')
        if self.pending is not None:
            raise RuntimeError('finish_step is required before the next physics step')
        if self.env.auto_reset or int(self.env.num_envs) != 1:
            raise ValueError('Single environment / no-auto-reset contract changed')
        if int(self.env.max_episode_length) != self.maximum_steps:
            raise ValueError('Finite horizon changed during the episode')
        if int(scalar(self.env.episode_length_buf)) != self.steps:
            raise ValueError('Unaccounted physics step or environment reset')
        if int(self.env.cube.pose_reset_count) != self.initial_reset_count:
            raise ValueError('Cube reset/pose write during task')

    def step(self, action, *, phase, step_id):
        """Execute the identical native action once and consume raw done.

        Returns task status, not the base reward/done tuple. Original raw values
        are held for the independent durable transition written by finish_step.
        Drop/timeout lock the task immediately, even before auditing the row.
        """
        try:
            self._guard()
            if step_id != self.steps or not isinstance(phase, str) or not phase:
                raise ValueError('Contiguous trajectory step ID and explicit phase required')
            action_data = frozen(action)
            self.pending = {'step': self.steps, 'phase': phase, 'action': action_data,
                            'action_dtype': str(getattr(action, 'dtype', type(action).__name__)),
                            'raw_counters_before': self._raw_counts()}
            returned = self.env.step(action)  # no cloning, decoding or target roundtrip
            if not isinstance(returned, (tuple, list)) or len(returned) != 4:
                raise ValueError('Expected frozen base four-return step API')
            raw_reward, raw_done = scalar(returned[1]), flag(returned[2])
            metrics = frozen(self.env.metrics)
            terms = metrics['terms']
            pickup, drop = flag(metrics['success']), flag(metrics['drop'])
            length = int(scalar(self.env.episode_length_buf))
            timeout = length >= self.maximum_steps
            raw = {'reward_returned': float(raw_reward), 'done_returned': raw_done,
                   'returned_api_arity': 4, 'returned_info': frozen(returned[3]),
                   'metrics': metrics, 'counters_after': self._raw_counts(),
                   'episode_length': length, 'pickup_success': pickup, 'drop': drop,
                   'derived_timeout': timeout, 'raw_truncated_returned': None}
            self.pending['raw'] = raw
            if timeout:
                self._stop('timeout', truncated=True)
            elif drop:
                self._stop('cube_drop')
            if (length != self.steps + 1 or raw_done != (pickup or drop or timeout)
                    or flag(terms['terminal']) != raw_done or flag(terms['timeout']) != timeout
                    or flag(terms['drop']) != drop):
                raise ValueError('Raw done/terms/step count inconsistent with frozen base contract')
            expected_counts = {'episodes': self.pending['raw_counters_before']['episodes'] + int(raw_done),
                               'successes': self.pending['raw_counters_before']['successes'] + int(pickup and raw_done)}
            if raw['counters_after'] != expected_counts:
                raise ValueError('Raw episode counters differ from frozen base contract')
            if int(self.env.cube.pose_reset_count) != self.initial_reset_count:
                raise ValueError('Cube reset/pose write during physics step')
            self.raw_success_frames += int(pickup)
            self.raw_done_frames += int(raw_done)
            new_milestone = pickup and self.pickup_milestone_step is None
            if new_milestone:
                self.pickup_milestone_step = self.steps
            self.pending['pickup_milestone_first_observed'] = bool(new_milestone)
            return self.status()
        except Exception as error:
            # Guard misuse also fails closed. Preserve any physical transition
            # that occurred as unfinished evidence; never invent an audited row.
            self.abort('step_protocol_error:' + str(error), detail={'error_type': type(error).__name__})
            raise

    def finish_step(self, row, *, failure=None, audit):
        """Commit after all original per-step stage checks, then read done."""
        if self.pending is None or self.final_written:
            raise RuntimeError('No unfinished physics step to audit')
        try:
            frozen_row, frozen_audit = frozen(row), frozen(audit)
            if (frozen_row.get('step') != self.pending['step']
                    or frozen_row.get('phase') != self.pending['phase']):
                raise ValueError('Audit row does not match executed step and phase')
            if frozen_audit.get('original_audits_completed') is not True:
                raise ValueError('Explicit completion of original stage audits is required')
            if failure:
                self._stop(str(failure), truncated=(failure == 'execution_interrupted'))
            self.steps += 1
            event = {**self.pending, 'record_type': 'transition', 'row': frozen_row,
                     'audit': frozen_audit, 'original_stage_failure': failure}
            self.pending = None
            event['task'] = self.status()
            self._emit(event)
            if self.status()['done']:
                self._final_event(detail={'last_transition_step': event['step']})
            return self.status()
        except Exception as error:
            self.abort('finish_protocol_error:' + str(error))
            raise

    def _final_event(self, *, detail=None):
        if not self.final_written:
            self._emit({'record_type': 'episode_final', 'after_completed_steps': self.steps,
                        'task': self.status(), 'raw_counters': self._raw_counts(),
                        'unfinished_transition': self.pending, 'detail': detail})
            self.final_written = True

    def abort(self, reason, *, truncated=False, detail=None):
        """Planning/gate/interruption failure: final event, no fake transition."""
        if not isinstance(reason, str) or not reason:
            raise ValueError('An explicit abort reason is required')
        self._stop(reason, truncated=truncated)
        self._final_event(detail=diagnostic(detail))
        return self.status()

    def conclude(self, original_result):
        """Apply the unmodified original full result for the declared target.

        A 15/30-degree target may complete its own development subtask. Only a
        successful 90-degree task retains face_turn_validated=True. No result
        here promotes a general rotation policy or a solved six-face cube.
        """
        if self.pending is not None:
            self.abort('conclude_with_unfinished_step', detail=diagnostic(original_result))
            raise RuntimeError('finish_step required before final conclusion')
        if self.status()['done']:
            return self.status()
        try:
            result = frozen(original_result)
            push = result.get('push', {})
            checks = {
                'full_episode': result.get('full_episode') is True,
                'not_interrupted': result.get('execution_interrupted') is False,
                'no_failure': result.get('failure') is None and push.get('failure') is None,
                'dynamic_grasp': result.get('dynamic_grasp_passed') is True,
                'measured_grasp': result.get('measured', {}).get('accepted') is True,
                'wrist': result.get('wrist', {}).get('accepted') is True,
                'assembled_hold': result.get('assembled_hold_s', 0) >= 3.,
                'center': result.get('center_transport_passed') is True and push.get('center', {}).get('passed') is True,
                'executed': push.get('executed') is True,
                'original_turn_gate': push.get('qualified') is True,
                'opposition_grasp': push.get('opposition_grasp_passed') is True,
                'turn_started': push.get('turn_started') is True,
                'turn_hold': push.get('turn_hold_s', 0) >= 3.,
                'declared_target': push.get('target_degrees') == self.target,
                'ninety_only_field': result.get('face_turn_validated') is (self.target == 90),
                'no_pose_writes': result.get('cube_pose_writes_during_control') == 0,
                'all_steps_recorded': result.get('steps') == self.steps and self.steps > 0,
                'finite_horizon': self.steps < self.maximum_steps,
            }
            if all(checks.values()):
                self._stop('declared_target_task_gate_passed', success=True)
            else:
                self._stop(result.get('failure') or push.get('failure') or 'original_full_task_gate_not_met')
            self._final_event(detail={'original_result': result, 'original_gate_checks': checks})
            return self.status()
        except Exception as error:
            self.abort('conclude_protocol_error:' + str(error), detail=diagnostic(original_result))
            raise
