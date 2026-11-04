"""Actual-rollout left-policy evaluation over the frozen ordinary-cube baseline.

No left teacher position action is executed, including at reset. The original
teacher still owns public phase scheduling, right actions and its declared K20
gain event. Those roles are explicit; this is not a whole-robot neural claim.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from left_dagger_baseline_runtime_v3 import BaselineRuntime
from dagger_collection_core_v1 import (FrozenTeacherOracle, STATE_FIELDS, validate_gate,
                                      physics_fingerprint, safe_abort_evidence, actor_observation)
from collect_left_dagger_draft_v1 import arrival_initialized
from zero_gated_residual_policy_v7 import EpisodeActor, features as v7_features
from phase_contact_policy import features as v4_features


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_joint_contract(policy_metadata, baseline_metadata, names, left_indices):
    actual_left = [i for i, name in enumerate(names) if name.endswith('_L') or name.startswith('left_')]
    if len(names) != 56 or len(actual_left) != 28 or list(left_indices) != actual_left:
        raise ValueError('Actual environment left28 ordering is invalid')
    for metadata in (policy_metadata, baseline_metadata):
        if metadata['joint_names'] != list(names) or metadata['output_indices'] != actual_left:
            raise ValueError('Policy/baseline/environment joint names or ordered left indices differ')


def actor_input(observation, schedule):
    """Only measured fields plus public same-action schedule enter the actor.

    The legacy feature helper requires this unused compatibility string. It is
    constant, never the actual executor identity, and contributes zero features.
    """
    return {**{key: deepcopy(observation[key]) for key in STATE_FIELDS},
            'phase': schedule['phase'], 'phase_progress': schedule['phase_progress'],
            'previous_executed_source': 'student'}


class V4EpisodeActor:
    def __init__(self, model, left):
        self.model = model.cpu().eval(); self.left = list(left)
        self.episode_id = None; self.next_step = 0
    def reset(self, episode_id): self.episode_id, self.next_step = str(episode_id), 0
    def act(self, observation, *, episode_id, step):
        if str(episode_id) != self.episode_id or step != self.next_step:
            raise RuntimeError('V4 control must also begin at reset with contiguous actual steps')
        x = torch.from_numpy(v4_features(observation, observation['phase'], observation['phase_progress'])).unsqueeze(0)
        previous = torch.tensor(np.asarray(observation['applied_target'], dtype=np.float32)[self.left]).unsqueeze(0)
        with torch.no_grad(): result = self.model.targets(x, previous)[0].float().tolist()
        self.next_step += 1
        return result


def preview_left_execution(env, native_teacher_action, predicted_left, left_indices):
    """Exact v3 left replacement and original projection, before any physics."""
    from joint_safety import project_target
    action = native_teacher_action.detach().clone()
    if action.shape != (1, 56) or action.dtype != torch.float32 or not torch.isfinite(action).all():
        raise ValueError('Finite original float32 native action required')
    left = torch.as_tensor(predicted_left, dtype=env.targets.dtype, device=env.targets.device).reshape(1, -1)
    if left.shape != (1,28) or not torch.isfinite(left).all(): raise ValueError('Invalid policy left28 target')
    previous = env.targets.detach().clone()
    action[:, left_indices] = (left - previous[:, left_indices]) / env.scales[:, left_indices]
    if not torch.isfinite(action).all(): raise ValueError('Nonfinite policy normalized action')
    requested = previous + env.scales * action.cpu()
    projected, soft_hit = project_target(previous, requested, env.limits,
        torch.minimum(env.scales,env.max_target_delta),env.command_velocity_limits,env.control_dt)
    return {'action': action, 'previous_applied_target': previous[0].tolist(),
            'requested_target': requested[0].tolist(), 'expected_applied_target': projected[0].tolist(),
            'soft_limit_hit': soft_hit[0].tolist()}


class EvaluationRuntime(BaselineRuntime):
    def __init__(self, args, preflight, model, policy_metadata, baseline_metadata, evaluation_identity):
        self.policy_model = model.cpu().eval()
        self.policy_metadata, self.baseline_metadata = policy_metadata, baseline_metadata
        self.evaluation_identity = deepcopy(evaluation_identity)
        self.actor = None
        super().__init__(args, preflight)

    def write(self, name, value):
        if name == 'runtime_protocol.json':
            value = {**value, 'scope':'actual_rollout_left_position_policy_from_reset_to_wrist',
                     'collection_mode':'closed_loop_policy_evaluation_no_left_teacher_takeover',
                     'teacher_probability':0., 'left_teacher_position_action_steps':0,
                     'right_and_stage_controller':'original_teacher',
                     'gain_schedule':'original teacher first tilt_hold left-finger K20; audited separately',
                     'checkpoint_roles':self.evaluation_identity}
        super().write(name, value)

    def _initialize(self):
        super()._initialize()
        actual_family = physics_fingerprint(self.physics_identity)
        if actual_family != self.evaluation_identity['expected_physics_family_sha256']:
            raise ValueError('Actual constructed environment physics family differs from qualified baseline data')
        validate_joint_contract(self.policy_metadata, self.baseline_metadata, self.env.cfg['joint_names'], self.left_indices)
        self.actor = (EpisodeActor(self.policy_model, self.left_indices) if self.args.policy == 'v7'
                      else V4EpisodeActor(self.policy_model, self.left_indices))
        self.actor.reset(self.args.name)
        self.evaluation_identity.update(actual_physics_family_sha256=actual_family,
            joint_order_verified_against_actual_env=True, actor_hidden_reset_count=1,
            actual_actor_feature_schema=self.policy_metadata.get('feature_schema','frozen_v4_198_phase_basis_features'))
        self.write('evaluation_protocol.json', self.evaluation_identity)

    def execute_left(self, native_teacher_action, predicted_left):
        preview = preview_left_execution(self.env, native_teacher_action, predicted_left, self.left_indices)
        packet = {'native_teacher_normalized_action':native_teacher_action, 'executed_source':'student',
                  'left_indices':list(self.left_indices), 'student_left_target':list(predicted_left)}
        super().execute_packet(packet)
        actual = torch.as_tensor(self.last_row['native_action'],dtype=torch.float32).reshape(1,56)
        if not torch.equal(actual.view(torch.int32), preview['action'].cpu().view(torch.int32)):
            raise RuntimeError('Executed native action differs bitwise from previewed left-only composition')
        expected = torch.as_tensor(preview['expected_applied_target'],dtype=torch.float32).reshape(1,56)
        if not torch.equal(expected.view(torch.int32),self.env.targets.cpu().view(torch.int32)):
            raise RuntimeError('Actual command differs bitwise from original projected policy target')
        right=[i for i in range(56)if i not in self.left_indices]
        if not torch.equal(actual[:,right].view(torch.int32),native_teacher_action.cpu()[:,right].view(torch.int32)):
            raise RuntimeError('Original right-hand native action changed')
        return {**{key:value for key,value in preview.items()if key!='action'},
                'native_action':actual[0].tolist(), 'right_native_bits_preserved':True,
                'actual_projection_bits_verified':True, 'left_teacher_position_action_executed':False}

    def finalize_evaluation(self, summary):
        result = super().finalize(summary)
        result.update(kind='closed_loop_left_'+self.args.policy+'_from_reset_to_wrist_evaluation',
            collection_mode='closed_loop_policy_evaluation_no_left_teacher_takeover',
            evaluation_identity=self.evaluation_identity,
            left_policy_position_steps=summary['physical_steps'], left_teacher_position_action_steps=0,
            right_teacher_action_steps=summary['physical_steps'], left_policy_position_targets_from_reset=True,
            left_policy_wrist_gate_passed=bool(summary['original_wrist_gate_qualified'] and not self.failure),
            whole_robot_pure_neural_episode=False, pure_student_episode_claim=False,
            eligible_training_episode=False, corrective_labels_qualified_for_training=False,
            original_baseline_checkpoint_role='frozen phase base and original source/clock reference',
            evaluated_policy_checkpoint_sha256=self.evaluation_identity['evaluated_policy_checkpoint_sha256'])
        self.write('result.json',result)
        self.write('artifact_manifest.json',{'sha256':{p.name:sha(p)for p in self.output.iterdir()if p.is_file()and p.name!='artifact_manifest.json'}})
        return result


def evaluate(runtime, journal, *, max_steps, stopped=lambda:False):
    """Single query, single actor advance, single physics action, then audits."""
    oracle = FrozenTeacherOracle(runtime.teacher,runtime.joint_names,runtime.capture,runtime.decode_target,
        runtime.decoder_id,runtime.schedule_resolver,memory_mode='preserve',
        control_change_validator=runtime.validate_control_change)
    complete_steps = 0; gain_events = 0; qualified = False; attempted = None
    initial_counter = int(runtime.env.episode_length_buf[0])
    if initial_counter != 0:
        raise RuntimeError('From-reset evaluation requires actual environment episode counter zero')
    physics_dispatches = 0
    def execution_evidence():
        counter = int(runtime.env.episode_length_buf[0])
        return {'physical_steps':counter-initial_counter,
                'physical_steps_definition':'completed control steps from actual non-resetting environment episode counter',
                'environment_episode_counter':counter,
                'fully_audited_transition_steps':complete_steps,
                'action_execution_dispatch_attempts':physics_dispatches,
                'trajectory_rows_recorded':len(runtime.rows)}
    def abort(reason, decision):
        evidence = execution_evidence()
        before = None if decision is None else decision.get('environment_episode_counter_before_step')
        emit({'record_type':'abort','step':step,'reason':reason,'unfinished_decision':decision,
              **evidence, 'physics_step_completed_for_unfinished_decision':bool(
                  before is not None and evidence['environment_episode_counter'] > before),
              'execution_dispatch_without_counter_advance':bool(before is not None and
                  evidence['environment_episode_counter']==before)})
    def emit(record):
        evidence,invalid = safe_abort_evidence(record)
        journal.write(json.dumps({**evidence,'invalid_evidence_fields':invalid},allow_nan=False)+'\n');journal.flush()
    for step in range(max_steps):
        if stopped(): runtime.failure='execution_interrupted';break
        attempted = {'step':step,'record_type':'decision_attempt','left_teacher_position_action_executed':False}
        try:
            observation = runtime.capture()
            pre_gates, _ = runtime.pre_gates(observation)
            validate_gate(pre_gates)
            oracle.memory_mode = 'sync_actual_applied' if arrival_initialized(runtime.teacher) else 'preserve'
            expert = oracle.query(observation)
            gain_events += int(expert['teacher_runtime_control_setting_change'])
            attempted['original_controller_audit'] = expert
            if expert['failure_reason']:
                runtime.failure=str(expert['failure_reason']);break
            runtime.archive_changed_plans(expert['teacher_state_after_query'])
            # Counterfactual labels are audit only, separately nested and never
            # passed to actor_input or the recurrent model. Qualification later.
            attempted['counterfactual_expert_projection_diagnostic_only'] = runtime.project_expert_target(oracle.last_native_action)
            safe_input = actor_input(observation,expert['public_same_action_schedule'])
            actual_features = (v7_features(safe_input)if runtime.args.policy=='v7'else
                               v4_features(safe_input,safe_input['phase'],safe_input['phase_progress']))
            if stopped(): runtime.failure='execution_interrupted';break
            prediction = runtime.actor.act(safe_input,episode_id=runtime.args.name,step=step)
            attempted.update(actor_observation_pre={k:v for k,v in safe_input.items()if k!='previous_executed_source'},
                actual_actor_features=actual_features.tolist(),predicted_left_target=prediction,
                actor_does_not_receive_expert_or_executor_identity=True)
            if stopped(): runtime.failure='execution_interrupted';break
            attempted['environment_episode_counter_before_step'] = int(runtime.env.episode_length_buf[0])
            physics_dispatches += 1
            execution = runtime.execute_left(oracle.last_native_action,prediction)
            attempted['execution'] = execution
            post = runtime.capture()
            attempted['actor_observation_post'] = {k:post[k]for k in STATE_FIELDS}
            # Reject nonfinite completed states before any successful audit record;
            # sanitization is reserved for retaining the resulting abort evidence.
            actor_observation(actor_input(post,expert['public_same_action_schedule']))
            post_gates, qualified, failure = runtime.post_gates(post)
            complete_steps += 1
            emit({**attempted,'record_type':'executed_transition','transition_finished':True,
                  'execution':execution,'actor_observation_post':{k:post[k]for k in STATE_FIELDS},
                  'raw_environment_termination':deepcopy(runtime.last_termination),
                  'raw_environment_metrics':deepcopy(runtime.env.metrics),
                  'original_stage_post_gates':post_gates,'original_wrist_gate_qualified':qualified,
                  'failure_reason':failure,'terminal':bool(failure or qualified),
                  'corrective_labels_qualified_for_training':False})
            attempted = None
            if failure or qualified: break
        except Exception as error:
            runtime.failure='evaluation_protocol_error:'+str(error)
            abort(runtime.failure,attempted)
            attempted = None
            break
    if not qualified and not runtime.failure: runtime.failure='evaluation_step_budget_before_original_wrist_gate'
    if attempted is not None:
        abort(runtime.failure,attempted)
    summary = {'scope':'left_position_policy_from_reset_pickup_through_wrist_only',
               'steps':execution_evidence()['physical_steps'], **execution_evidence(),
               'original_wrist_gate_qualified':bool(qualified and not runtime.failure),
               'left_teacher_position_action_steps':0,'teacher_query_count':oracle.query_count,
               'original_teacher_gain_schedule_events':gain_events,'actor_hidden_reset_count':1,
               'center_oracle_implemented':False,'pure_student_episode_claim':False,
               'corrective_labels_qualified_for_training':False,'failure':runtime.failure}
    emit({'record_type':'evaluation_final','summary':summary})
    return summary
