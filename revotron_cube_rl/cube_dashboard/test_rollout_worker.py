"""Observer checks, no Isaac or GPU allocation."""
import hashlib
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from cube_dashboard import rollout_worker as w


class Results(unittest.TestCase):
    def test_wrong_panel_or_teacher_or_trace_mismatch_rejected(self):
        rows=[dict(steps=1,lane=i,seed=s,phase_steps={'ready':1},teacher_executed_left_steps=0,student_executed_left_steps=1,
                   max_height_m=.03,best_controlled_hold_s=0.,end_reason='step_limit',ever_whole_turn=False,final_table_contact=False)
              for i,s in enumerate(w.SEEDS)]
        report={'role':'validate','seeds':w.SEEDS[:],'parent_sha':'c','cube_edge_m':.09,'asset_sha256':'a','grasp_sha256':'b','physics_device':'cuda:0','cross_lane_contacts':0,'per_lane':rows}
        plan={'parent_sha':'c','asset_sha256':'a','grasp_sha256':'b'}
        def make():return w.result_card({'id':'test'},report,plan,1,{'ready':1},.9,'t')
        ok=make();self.assertTrue(ok['physics_stepped']);self.assertIsNone(ok['summary']['face_turn'])
        self.assertEqual(ok['seed'],20000000)
        for k,v in [('cross_lane_contacts',1),('asset_sha256','different'),('role','collection'),('seeds',[21611,21612])]:
            with self.subTest(key=k), patch.dict(report,{k:v}), self.assertRaises(ValueError):make()
        for k,v in [('steps',2),('teacher_executed_left_steps',1),('phase_steps',{'wrong':1})]:
            with self.subTest(key=k), patch.dict(rows[0],{k:v}), self.assertRaises(ValueError):make()

    def test_lane_offset_and_quaternion_order(self):
        import numpy as np
        z=.5**.5
        got=w._rotated_point([5,3,1,0,0,z,z],[1,0,0],np.array([4,4,0]))
        np.testing.assert_allclose(got,[1,0,1],atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires project env')
    def test_refuses_unreserved_gpu_before_launch(self):
        with patch.dict('os.environ',{},clear=True), self.assertRaisesRegex(RuntimeError,'空闲'):
            w.require_reservations()


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires project env')
class RealModelInputs(unittest.TestCase):
    @unittest.skipUnless((w.inference.RUNS/'contact_hold90_long_v2/settings.json').is_file(), 'no local v2 assets')
    def test_real_plan_matches_frozen_training_code_and_fixed_seeds(self):
        card, plan=w.make_plan('r15')
        self.assertEqual(plan['parent_sha'],w.inference.R15_SHA)
        self.assertEqual(plan['seeds'],[20000000,20000001,20000002,20000003])
        self.assertEqual(plan['role'],'validate')
        self.assertEqual(plan['code']['contact_hold90_cuda_sample.py'],hashlib.sha256((w.inference.ROOT/'contact_hold90_cuda_sample.py').read_bytes()).hexdigest())
        with self.assertRaises(ValueError):w.make_plan('../../best.pt')

    def test_observer_copies_and_preserves_real_model_and_native_step_results(self):
        import torch
        import contact_hold90_cuda_domain as domain
        import contact_hold_policy as policy
        transitions=[]
        class Scene:
            def step_physics(self,cmd):
                transitions.append(cmd.clone());return 'native step'
        class Meter:
            def update(self,*args):
                return 'native score'
        sampler=SimpleNamespace(HoldMeter=Meter)
        recorder=SimpleNamespace(left=[0])
        recorder.bind=lambda s:None
        def observe_model(_m,args,output):
            recorder.raw=output.clone();return None
        recorder.forward=observe_model
        def record(t,*args):
            self.assertEqual(t,0);self.assertEqual(recorder.executed.tolist(),[4.])
        recorder.capture=record
        class Net(torch.nn.Module):
            def forward(self,x):return x*2
        def fake_sample(args):
            env=domain.open_scene()
            model=policy.restored()
            out=model(torch.tensor([[2.]]))
            self.assertTrue(torch.equal(recorder.raw,out))
            self.assertEqual(env.step_physics(out),'native step')
            self.assertEqual(sampler.HoldMeter().update(0,None,None,None),'native score')
        sampler.sample=fake_sample
        with patch.object(domain,'open_scene',Scene),patch.object(policy,'restored',Net):
            w.observed_validation(sampler,SimpleNamespace(role='validate'),recorder)
            self.assertIs(domain.open_scene,Scene);self.assertIs(policy.restored,Net);self.assertIs(sampler.HoldMeter,Meter)
        self.assertEqual(len(transitions),1);self.assertEqual(transitions[0].tolist(),[[4.]])


if __name__=='__main__':unittest.main()
