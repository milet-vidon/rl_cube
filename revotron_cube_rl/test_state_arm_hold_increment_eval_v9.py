from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
try: import joint_safety
except ImportError: import baseline_joint_safety as joint_safety

from state_arm_hold_increment_eval_runtime_v9 import (EvaluationRuntime, actor_input, evaluate,
    preview_left_execution, validate_joint_contract, V4EpisodeActor)
from evaluate_state_arm_hold_increment_v9 import snapshot_bytes
from test_phase_residual_v5 import observation, baseline, NAMES
from phase_basis_policy_v4 import restored
from phase_contact_policy import features as old_features
from phase_residual_policy_v5 import features


def gate():
    return {'robot_safety':True,'joint_limits':True,'cube_integrity':True,'no_pose_write':True,
        'stage_gate_qualified':True,'original_audit_source_sha256':{'fixture':'not_physics_evidence'}}


class RecordingActor:
    def __init__(self):self.inputs=[];self.resets=0;self.next_step=0
    def reset(self,episode_id):self.episode_id=episode_id;self.resets+=1;self.next_step=0
    def act(self,obs,*,episode_id,step):
        assert episode_id==self.episode_id and step==self.next_step
        assert 'expert_supervision'not in obs and 'oracle_env_inputs'not in obs
        self.inputs.append(deepcopy(obs));self.next_step+=1
        return [value+.003 for value in obs['applied_target'][:28]]


class NativeTeacher:
    def __init__(self):
        self.name='ready';self.tick=0;self.failure_reason=None;self.accepted=False
        self.base=SimpleNamespace(finger_arrival=SimpleNamespace(targets={},reference={},anchors={},tick=7))
        self.actions=[];self.fail_first=False
    def action(self):
        self.tick+=1
        if self.tick==2:self.name='squeeze'
        if self.fail_first:self.failure_reason='original_teacher_gate_failure'
        self.base.finger_arrival.targets['_applied_'+NAMES[0]]=.7
        self.base.finger_arrival.targets[NAMES[0]]=.8
        self.base.finger_arrival.tick+=1
        action=torch.full((1,56),.9,dtype=torch.float32)
        action[:,28:]=torch.linspace(-.5,.5,28)
        self.actions.append(action.clone())
        return action


class FakeRuntime(EvaluationRuntime):
    """Uses production execute_left + inherited v3 execute_packet, no simulator."""
    def __init__(self,raw_done_step=None,success_step=3):
        self.args=SimpleNamespace(policy='v9',name='host_case')
        self.joint_names=NAMES;self.left_indices=list(range(28));self.failure=None;self.last_termination=None
        self.rows=[];self.last_row=None;self.current_schedule={'phase':'ready','phase_progress':0.}
        self.teacher=NativeTeacher();self.actor=RecordingActor();self.actor.reset(self.args.name)
        self.source_sha256={'environment.py':'fixture','joint_safety.py':'fixture'}
        self.raw_done_step=raw_done_step;self.success_step=success_step;self.executed=[]
        self.env=SimpleNamespace(targets=torch.zeros(1,56),scales=torch.full((1,56),.05),
            limits=torch.tensor([[[-2.,2.]]*56]),max_target_delta=torch.full((1,56),.04),
            command_velocity_limits=torch.ones(1,56),control_dt=1/30,episode_length_buf=torch.zeros(1,dtype=torch.long),
            max_episode_length=3500,metrics={})
        def step(action):
            self.executed.append(action.clone());e=self.env
            e.targets,_=joint_safety.project_target(e.targets,e.targets+e.scales*action,e.limits,
                torch.minimum(e.scales,e.max_target_delta),e.command_velocity_limits,e.control_dt)
            e.episode_length_buf+=1;done=int(e.episode_length_buf[0])==self.raw_done_step
            e.metrics={'success':False,'drop':False,'terms':{'terminal':float(done),'timeout':0.}}
            return {},torch.tensor([0.]),torch.tensor([done]),{}
        self.env.step=step
        self.projection_parameters=self.read_projection_parameters();self.projection_parameters_file_sha256='fixture'
    def capture(self):
        obs=observation();obs['applied_target']=self.env.targets[0].tolist();obs['q']=self.env.targets[0].tolist()
        obs.update(self.current_schedule)
        obs['previous_executed_source']='reset'if not self.rows else 'student'
        obs['oracle_env_inputs']={'episode_length':int(self.env.episode_length_buf[0]),'max_episode_length':3500,'metrics':deepcopy(self.env.metrics)}
        obs['control_settings']={}
        return obs
    def pre_gates(self,obs):return gate(),False
    def post_gates(self,obs):
        if self.last_termination['sampling_terminal']:self.failure='original_environment_done'
        qualified=len(self.rows)>=self.success_step and not self.failure
        return gate(),bool(qualified),self.failure
    def schedule_resolver(self,obs,before,after):
        self.current_schedule={'phase':after['name'],'phase_progress':min(1.,len(self.rows)/10)}
        return dict(self.current_schedule)
    def validate_control_change(self,*args):return False
    def archive_changed_plans(self,*args):pass
    def _record_row(self,action,packet):
        self.last_row={'step':len(self.rows),'native_action':action[0].tolist(),'executed_source':'student'}
        self.rows.append(self.last_row)


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.module_patch=patch.dict('sys.modules',{'joint_safety':joint_safety});self.module_patch.start();self.addCleanup(self.module_patch.stop)

    def test_joint_names_and_order_must_match_baseline_and_actual_env(self):
        metadata={'joint_names':NAMES,'output_indices':list(range(28))}
        validate_joint_contract(metadata,metadata,NAMES,list(range(28)))
        changed=deepcopy(metadata);changed['output_indices'][0:2]=[1,0]
        with self.assertRaises(ValueError):validate_joint_contract(changed,metadata,NAMES,list(range(28)))
        changed=deepcopy(metadata);changed['joint_names'][0]='left_different'
        with self.assertRaises(ValueError):validate_joint_contract(changed,metadata,NAMES,list(range(28)))

    def test_actor_input_excludes_expert_and_actual_executor_identity(self):
        obs=observation();obs.update(expert_supervision={'left':[999.]*28},oracle_env_inputs={'labels':[888.]},previous_executed_source='teacher')
        safe=actor_input(obs,{'phase':'ready','phase_progress':0.})
        self.assertNotIn('expert_supervision',safe);self.assertNotIn('oracle_env_inputs',safe)
        self.assertEqual(safe['previous_executed_source'],'student')
        other=deepcopy(obs);other['previous_executed_source']='reset';other['expert_supervision']['left']=[-999.]*28
        np.testing.assert_array_equal(features(safe),features(actor_input(other,{'phase':'ready','phase_progress':0.})))

    def test_preview_replaces_left_only_and_preserves_right_bits_and_input(self):
        runtime=FakeRuntime();native=runtime.teacher.action();before=native.clone()
        native[:,28]=-0.;native[:,29]=0.;before=native.clone()
        preview=preview_left_execution(runtime.env,native,[.003]*28,list(range(28)))
        self.assertTrue(torch.equal(preview['action'][:,28:].view(torch.int32),before[:,28:].view(torch.int32)))
        self.assertTrue(torch.equal(native.view(torch.int32),before.view(torch.int32)))
        self.assertFalse(torch.equal(preview['action'][:,:28],before[:,:28]))

    def test_left_policy_executes_from_reset_with_own_actual_history_and_same_action_phase(self):
        runtime=FakeRuntime();log=io.StringIO();result=evaluate(runtime,log,max_steps=10)
        records=[json.loads(line)for line in log.getvalue().splitlines()]
        transitions=[row for row in records if row['record_type']=='executed_transition']
        self.assertEqual(len(transitions),3);self.assertEqual(runtime.actor.resets,1)
        self.assertEqual(len(runtime.teacher.actions),3);self.assertEqual(len(runtime.actor.inputs),3)
        self.assertEqual(result['left_teacher_position_action_steps'],0)
        self.assertEqual(runtime.actor.inputs[0]['applied_target'][:28],[0.]*28)
        self.assertAlmostEqual(runtime.actor.inputs[1]['applied_target'][0],.003,places=7)
        self.assertEqual(runtime.actor.inputs[1]['phase'],'squeeze')
        self.assertEqual(transitions[0]['original_controller_audit']['execution_memory_mode'],'preserve')
        self.assertEqual(transitions[1]['original_controller_audit']['execution_memory_mode'],'sync_actual_applied')
        changes=transitions[1]['original_controller_audit']['execution_memory_changes']
        self.assertAlmostEqual(changes[0]['after'],.003,places=7)
        self.assertEqual(runtime.teacher.base.finger_arrival.targets[NAMES[0]],.8)
        for executed,teacher in zip(runtime.executed,runtime.teacher.actions):
            self.assertTrue(torch.equal(executed[:,28:].view(torch.int32),teacher[:,28:].view(torch.int32)))
            self.assertFalse(torch.equal(executed[:,:28],teacher[:,:28]))
        self.assertTrue(result['original_wrist_gate_qualified'])
        self.assertFalse(result['pure_student_episode_claim'])
        self.assertTrue(all(len(row['actual_actor_features'])==284 for row in transitions))

    def test_raw_done_stops_before_next_query_actor_or_physics(self):
        runtime=FakeRuntime(raw_done_step=2);log=io.StringIO();result=evaluate(runtime,log,max_steps=10)
        self.assertEqual(len(runtime.executed),2);self.assertEqual(runtime.teacher.tick,2);self.assertEqual(runtime.actor.next_step,2)
        self.assertEqual(result['failure'],'original_environment_done')
        self.assertFalse(result['original_wrist_gate_qualified'])

    def test_teacher_gate_failure_never_executes_fallback_teacher_left(self):
        runtime=FakeRuntime();runtime.teacher.fail_first=True;log=io.StringIO();result=evaluate(runtime,log,max_steps=10)
        self.assertEqual(runtime.executed,[]);self.assertEqual(runtime.actor.inputs,[])
        self.assertEqual(result['failure'],'original_teacher_gate_failure')
        self.assertIn('"record_type": "abort"',log.getvalue())

    def test_exception_after_physics_retains_physical_step_and_unfinished_evidence(self):
        runtime=FakeRuntime();original=runtime.execute_left
        def fail_after(*args):
            original(*args)
            raise RuntimeError('post-physics bitwise check failure')
        runtime.execute_left=fail_after
        log=io.StringIO();result=evaluate(runtime,log,max_steps=10)
        self.assertEqual(len(runtime.executed),1)
        self.assertEqual(result['steps'],1);self.assertEqual(result['physical_steps'],1)
        self.assertEqual(result['fully_audited_transition_steps'],0)
        abort=json.loads(log.getvalue().splitlines()[0])
        self.assertTrue(abort['physics_step_completed_for_unfinished_decision'])
        self.assertFalse(result['original_wrist_gate_qualified'])

    def test_nonfinite_post_state_is_abort_not_sanitized_successful_transition(self):
        runtime=FakeRuntime();original=runtime.capture
        def corrupt_post():
            state=original()
            if runtime.executed:state['cube_linear_velocity'][0]=float('nan')
            return state
        runtime.capture=corrupt_post
        log=io.StringIO();result=evaluate(runtime,log,max_steps=10)
        records=[json.loads(line)for line in log.getvalue().splitlines()]
        self.assertEqual(result['physical_steps'],1)
        self.assertEqual(result['fully_audited_transition_steps'],0)
        self.assertEqual(records[0]['record_type'],'abort')
        self.assertTrue(records[0]['invalid_evidence_fields'])
        self.assertFalse(result['original_wrist_gate_qualified'])

    def test_stop_before_start_adds_no_query_or_action(self):
        runtime=FakeRuntime();log=io.StringIO();result=evaluate(runtime,log,max_steps=10,stopped=lambda:True)
        self.assertEqual(runtime.executed,[]);self.assertEqual(runtime.teacher.tick,0)
        self.assertEqual(result['failure'],'execution_interrupted')

    def test_from_reset_claim_rejects_nonzero_actual_episode_counter(self):
        runtime=FakeRuntime();runtime.env.episode_length_buf+=1
        with self.assertRaisesRegex(RuntimeError,'counter zero'):
            evaluate(runtime,io.StringIO(),max_steps=10)
        self.assertEqual(runtime.teacher.tick,0);self.assertEqual(runtime.executed,[])

    def test_v4_matched_actor_uses_same_dynamic_schedule_and_actual_previous_command(self):
        policy=restored(baseline());actor=V4EpisodeActor(policy,list(range(28)));actor.reset('test')
        obs=observation();obs['applied_target'][:28]=[.1]*28;obs['phase']='squeeze';obs['phase_progress']=.7
        expected=policy.targets(torch.from_numpy(old_features(obs,'squeeze',.7)).unsqueeze(0),torch.tensor([[.1]*28]))[0].float().tolist()
        self.assertEqual(actor.act(obs,episode_id='test',step=0),expected)
        with self.assertRaises(RuntimeError):actor.act(obs,episode_id='other',step=0)

    def test_checkpoint_snapshot_pins_evaluated_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)as root:
            root=Path(root);source=root/'source';dest=root/'snapshot';source.write_bytes(b'first')
            import hashlib
            digest=hashlib.sha256(b'first').hexdigest();self.assertEqual(snapshot_bytes(source,dest,digest),digest)
            source.write_bytes(b'second');self.assertEqual(dest.read_bytes(),b'first')
            with self.assertRaises(ValueError):snapshot_bytes(source,root/'wrong',digest)


if __name__=='__main__':unittest.main()
