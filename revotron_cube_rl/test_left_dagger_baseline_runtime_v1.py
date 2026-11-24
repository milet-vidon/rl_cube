import unittest
from types import SimpleNamespace
from left_dagger_baseline_runtime_v1 import BaselineRuntime,termination_status


class RuntimeHostTests(unittest.TestCase):
    def test_original_done_causes_are_terminal_and_truncation_is_not_invented(self):
        cases=[({'success':True},10,3500,'generic_pickup_success'),
               ({'drop':True},10,3500,'drop'),({},3500,3500,'timeout')]
        for metrics,steps,limit,cause in cases:
            r=termination_status(True,metrics,steps,limit)
            self.assertTrue(r['sampling_terminal']);self.assertIn(cause,r['done_causes'])
            self.assertIsNone(r['raw_truncated']);self.assertTrue(r['done_matches_original_causes'])
        r=termination_status(False,{'success':False,'drop':False},1647,3500)
        self.assertFalse(r['sampling_terminal']);self.assertFalse(r['derived_timeout_truncated'])
        self.assertTrue(termination_status(True,{},10,3500)['unknown_done_cause'])
        inconsistent=termination_status(False,{'success':True},10,3500)
        self.assertFalse(inconsistent['done_matches_original_causes']);self.assertTrue(inconsistent['sampling_terminal'])

    def test_phase_progress_uses_actual_same_action_phase_and_clamps_wait(self):
        from collections import Counter
        runtime=BaselineRuntime.__new__(BaselineRuntime)
        runtime.preflight={'phase_frame_counts':{'squeeze':85,'primary_lift':120,'center_hold':241}}
        runtime.phase_seen=Counter({'squeeze':84,'primary_lift':0})
        self.assertEqual(runtime.schedule_resolver({}, {'name':'squeeze'}, {'name':'primary_lift'}),
                         {'phase':'primary_lift','phase_progress':0.})
        runtime.phase_seen['squeeze']=100
        self.assertEqual(runtime.schedule_resolver({}, {}, {'name':'squeeze'})['phase_progress'],1.)
        with self.assertRaises(ValueError):runtime.schedule_resolver({}, {}, {'name':'center_hold'})

    def test_native_target_decoder_preserves_operation_order_dtype(self):
        import torch
        runtime=BaselineRuntime.__new__(BaselineRuntime)
        torch.manual_seed(5)
        runtime.env=SimpleNamespace(targets=torch.randn(1,56,dtype=torch.float32),scales=torch.rand(1,56,dtype=torch.float32))
        action=torch.randn(1,56,dtype=torch.float32)
        expected=runtime.env.targets+runtime.env.scales*action.cpu()
        self.assertTrue(torch.equal(torch.tensor(runtime.decode_target(action)),expected[0]))

    def test_gain_validator_only_original_first_left_k20(self):
        runtime=BaselineRuntime.__new__(BaselineRuntime);runtime.joint_names=['Joint1_L','left_middle_MCP_joint','right_middle_MCP_joint']
        before={'joint_stiffness':[100.,30.,30.],'joint_damping':[8.,1.,1.],'joint_armature':[0.,0.,0.]}
        after={**before,'joint_stiffness':[100.,20.,30.]}
        tb={'hold_gains_set':False};ta={'hold_gains_set':True,'name':'tilt_hold'}
        self.assertTrue(runtime.validate_control_change(before,after,tb,ta))
        self.assertFalse(runtime.validate_control_change(before,{**after,'joint_stiffness':[100.,20.,20.]},tb,ta))
        self.assertFalse(runtime.validate_control_change(before,after,{'hold_gains_set':True},ta))
        self.assertFalse(runtime.validate_control_change(before,after,tb,{**ta,'name':'primary_lift'}))

    def test_backend_name_mapping_is_not_assumed_user_id_order(self):
        import numpy as np
        runtime=BaselineRuntime.__new__(BaselineRuntime);runtime.backend_indices=[2,0,1]
        runtime.env=SimpleNamespace(robot=SimpleNamespace(root_view=SimpleNamespace(get_dof_stiffnesses=lambda:np.array([[1.,2.,3.]]))))
        self.assertEqual(runtime._backend('get_dof_stiffnesses'),[3.,1.,2.])

    def test_checkpoint_must_match_names_manifest_and_all_inference_hashes(self):
        from copy import deepcopy
        from left_dagger_baseline_runtime_v1 import validate_checkpoint_metadata,BASELINE
        preflight={'names':['a'],'left':[0],'metadata':{'manifest_sha256':'source'}}
        hashes={'phase_contact_policy.py':'contact','phase_basis_policy_v4.py':'basis'}
        meta={'joint_names':['a'],'output_indices':[0],'sources':[{'run':BASELINE,'manifest_sha256':'source'}],'source_sha256':hashes}
        self.assertTrue(validate_checkpoint_metadata(meta,preflight,hashes))
        for key,value in [('joint_names',['b']),('sources',[{'run':BASELINE,'manifest_sha256':'wrong'}]),('source_sha256',{'phase_contact_policy.py':'wrong','phase_basis_policy_v4.py':'basis'})]:
            altered=deepcopy(meta);altered[key]=value
            with self.assertRaises(ValueError):validate_checkpoint_metadata(altered,preflight,hashes)

    def test_physics_authorship_covers_core_joints_material_table_and_bindings(self):
        from left_dagger_baseline_runtime_v1 import physics_authorship
        def attr(name,value):return SimpleNamespace(GetName=lambda:name,Get=lambda:value)
        def rel(name,value):return SimpleNamespace(GetName=lambda:name,GetTargets=lambda:value)
        def prim(path,attrs,rels=()):return SimpleNamespace(GetPath=lambda:path,GetAttributes=lambda:attrs,GetRelationships=lambda:rels)
        stage=SimpleNamespace(Traverse=lambda:[
            prim('/World/Core',[attr('physics:mass',.1),attr('physics:velocity',[1.,2.,3.])]),
            prim('/World/Joints/a',[attr('drive:linear:physics:damping',.0)],[rel('physics:body0',['/World/Core'])]),
            prim('/World/Material',[attr('physics:staticFriction',.9)]),
            prim('/World/Table',[],[rel('material:binding:physics',['/World/Material'])])])
        data=physics_authorship(stage)
        self.assertEqual(len(data),4);self.assertNotIn('physics:velocity',data['/World/Core']['attributes'])
        self.assertEqual(data['/World/Table']['relationships']['material:binding:physics'],['/World/Material'])


if __name__=='__main__':unittest.main()
