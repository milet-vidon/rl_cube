import io
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
try:
    import joint_safety as original_safety
except ImportError:
    import baseline_joint_safety as original_safety

from left_dagger_baseline_runtime_v3 import BaselineRuntime
from dagger_projected_collection_v3 import ProjectedDaggerCollector,expand_record_provenance,PROVENANCE_FILE,persist_provenance
from dagger_collection_core_v1 import FrozenTeacherOracle
from test_dagger_collection_core_v1 import NAMES,Teacher,observation,physics_identity,gate


def runtime_stub():
    runtime=BaselineRuntime.__new__(BaselineRuntime)
    runtime.env=SimpleNamespace(targets=torch.zeros(1,56),scales=torch.full((1,56),.05),
        limits=torch.tensor([[[-2.,2.]]*56]),max_target_delta=torch.full((1,56),.04),
        command_velocity_limits=torch.ones(1,56),control_dt=1/30,
        metrics={},episode_length_buf=torch.zeros(1,dtype=torch.long),max_episode_length=3500)
    runtime.joint_names=NAMES;runtime.left_indices=list(range(28));runtime.failure=None;runtime.last_termination=None
    runtime.source_sha256={'environment.py':'test_only','joint_safety.py':'test_only'}
    runtime.projection_parameters=runtime.read_projection_parameters();runtime.projection_parameters_file_sha256='test_only'
    runtime.calls=[];runtime.rows=[];runtime.stub_done=False
    def step(action):
        runtime.calls.append(action.clone())
        e=runtime.env
        requested=e.targets+e.scales*action.cpu()
        e.targets,_=original_safety.project_target(e.targets,requested,e.limits,torch.minimum(e.scales,e.max_target_delta),e.command_velocity_limits,e.control_dt)
        e.episode_length_buf+=1
        return None,None,torch.tensor([runtime.stub_done]),{}
    runtime.env.step=step
    runtime._record_row=lambda a,p:runtime.rows.append({'action':a.clone(),'source':p['executed_source']})
    return runtime


def packet(source='teacher'):
    return {'native_teacher_normalized_action':torch.linspace(-.9,.9,56).reshape(1,56),
            'executed_source':source,'left_indices':list(range(28)), 'student_left_target':[.018]*28}


class ProjectionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.module_patch=patch.dict('sys.modules',{'joint_safety':original_safety});self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.runtime=runtime_stub()

    def test_native_teacher_action_and_projected_label_are_bitwise_original(self):
        p=packet();before=p['native_teacher_normalized_action'].clone()
        label=self.runtime.project_expert_target(before)
        self.assertTrue(torch.equal(self.runtime.env.targets,torch.zeros(1,56)))
        self.runtime.execute_packet(p)
        self.assertTrue(torch.equal(self.runtime.calls[0].view(torch.int32),before.view(torch.int32)))
        self.assertTrue(self.runtime.verify_teacher_projected_target(label['projected_target']))
        self.assertTrue(torch.equal(p['native_teacher_normalized_action'],before))
        self.assertIn('joint_safety.py',label['projection_source_sha256'])

    def test_student_replaces_only_left_and_preserves_original_right_packet(self):
        p=packet('student');before=p['native_teacher_normalized_action'].clone()
        label=self.runtime.project_expert_target(before)
        self.runtime.execute_packet(p)
        self.assertTrue(torch.equal(self.runtime.calls[0][:,28:].view(torch.int32),before[:,28:].view(torch.int32)))
        self.assertTrue(torch.equal(p['native_teacher_normalized_action'],before))
        self.assertEqual(len(self.runtime.calls),1)
        self.assertFalse(self.runtime.verify_teacher_projected_target(label['projected_target']))
        self.assertTrue(torch.allclose(self.runtime.env.targets[:,:28],torch.full((1,28),.018)))

    def test_noncontiguous_left_indices_preserve_all_other_joints(self):
        p=packet('student');self.runtime.left_indices=p['left_indices']=list(range(0,56,2))
        before=p['native_teacher_normalized_action'].clone();self.runtime.execute_packet(p)
        self.assertTrue(torch.equal(self.runtime.calls[0][:,1::2],before[:,1::2]))

    def test_any_raw_done_stops_before_second_action(self):
        self.runtime.stub_done=True;self.runtime.execute_packet(packet())
        self.assertTrue(self.runtime.last_termination['sampling_terminal'])
        with self.assertRaisesRegex(RuntimeError,'terminal'):self.runtime.execute_packet(packet())
        self.assertEqual(len(self.runtime.calls),1)

    def test_nonfinite_student_rejected_before_step(self):
        p=packet('student');p['student_left_target'][3]=float('nan')
        with self.assertRaises(ValueError):self.runtime.execute_packet(p)
        self.assertEqual(self.runtime.calls,[])

    def test_projection_reads_actual_previous_command_and_rejects_changed_parameters(self):
        self.runtime.env.targets[:,0]=.1
        label=self.runtime.project_expert_target(torch.full((1,56),100.))
        self.assertAlmostEqual(label['projection_previous_applied_target'][0],.1)
        self.assertGreater(label['requested_target'][0],5.)
        self.assertLessEqual(label['projected_target'][0],.134)
        self.assertTrue(label['projection_soft_limit_hit'][0])
        self.runtime.env.control_dt=.1
        with self.assertRaisesRegex(RuntimeError,'configuration changed'):self.runtime.project_expert_target(torch.zeros(1,56))


class ProjectedJournalTests(unittest.TestCase):
    def setUp(self):
        self.state=observation();self.state['oracle_env_inputs']={};self.state['control_settings']={}
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.addCleanup(self.directory.cleanup)
        self.teacher=Teacher(self.state);self.log=io.StringIO()
        self.oracle=FrozenTeacherOracle(self.teacher,NAMES,lambda:self.state,
            lambda a:a[0],'test_only',lambda o,b,a:{'phase':a['name'],'phase_progress':.5})
        self.verify=True
        self.runtime=SimpleNamespace(output=Path(self.directory.name),project_expert_target=lambda a:{'requested_target':a[0],
            'projected_target':[.003]+[0.]*55,'projection_parameters_file_sha256':'test_only'},
            verify_teacher_projected_target=lambda label:self.verify)
        self.collector=ProjectedDaggerCollector(self.oracle,self.log,'test',{
            'student_checkpoint_sha256':'test_only','source_sha256':{'test':'test'},
            'physics_identity':physics_identity()},list(range(28)),runtime=self.runtime)

    def prepare(self,source):
        def actor(obs):
            self.assertNotIn('expert_supervision',obs);self.assertNotIn('projected_target',obs)
            return [.002]*28
        return self.collector.prepare(self.state,actor,gate(),execute_source=source,
            selection_reason='test_only',selection_probability=.5,selection_draw=.25)

    def test_student_actual_is_never_used_as_expert_label(self):
        self.prepare('student');self.state['applied_target']=[.002]*56
        row=self.collector.complete(self.state,gate())
        self.assertEqual(row['expert_supervision']['requested_target'][0],.005)
        self.assertEqual(row['expert_supervision']['projected_target'][0],.003)
        self.assertEqual(row['actual_applied_target'][0],.002)
        self.assertIsNone(row['expert_projection_matches_teacher_actual_bitwise'])
        self.assertEqual(self.teacher.tick,1)

    def test_teacher_projection_mismatch_aborts_with_evidence(self):
        self.prepare('teacher');self.verify=False
        with self.assertRaisesRegex(RuntimeError,'validation failed'):self.collector.complete(self.state,gate())
        self.assertTrue(self.collector.terminal);self.assertIsNone(self.collector.pending)
        row=json.loads(self.log.getvalue());self.assertEqual(row['record_type'],'abort')
        self.assertIn('projected_target',row['pending_transition']['expert_supervision'])

    def test_teacher_projection_match_recorded_without_claim(self):
        self.prepare('teacher');row=self.collector.complete(self.state,gate())
        self.assertTrue(row['expert_projection_matches_teacher_actual_bitwise'])
        self.assertFalse(row['pure_student_episode_claim'])

    def test_provenance_expansion_restores_original_record_and_validated_identity(self):
        self.prepare('student');row=self.collector.complete(self.state,gate())
        full=json.loads((self.runtime.output/PROVENANCE_FILE).read_text())
        expected=deepcopy(row);expected['run_provenance']=full
        restored=expand_record_provenance(row,self.runtime.output)
        self.assertEqual(restored,expected)
        self.assertEqual(full['policy_observation_scope'],'privileged_simulator_state_not_vision_actor')
        self.assertEqual(full['physics_identity'],physics_identity())
        self.assertEqual(full['source_sha256'],{'test':'test'})
        with self.assertRaises(FileExistsError):persist_provenance(self.runtime.output,full)

    def test_tampered_provenance_file_ref_or_physics_hash_is_rejected(self):
        self.prepare('student');row=self.collector.complete(self.state,gate())
        altered=deepcopy(row);altered['run_provenance']['physics_identity_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Physics identity'):expand_record_provenance(altered,self.runtime.output)
        altered=deepcopy(row);altered['run_provenance']['file']='../collection_provenance.json'
        with self.assertRaisesRegex(ValueError,'path'):expand_record_provenance(altered,self.runtime.output)
        path=self.runtime.output/PROVENANCE_FILE;path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'file SHA256'):expand_record_provenance(row,self.runtime.output)

    def test_abort_top_and_pending_references_both_restore(self):
        self.prepare('teacher');self.collector.abort('test abort',self.state)
        row=json.loads(self.log.getvalue());restored=expand_record_provenance(row,self.runtime.output)
        self.assertEqual(restored['run_provenance'],restored['pending_transition']['run_provenance'])
        self.assertEqual(restored['record_type'],'abort')
        self.assertEqual(restored['pending_transition']['expert_supervision']['projected_target'][0],.003)
        self.assertFalse(restored['pure_student_episode_claim'])


if __name__=='__main__':unittest.main()
