"""CPU checks of the same tensor state machine invoked on live CUDA lanes."""
import json
import unittest
from types import SimpleNamespace
import torch
import contact_hold90_cuda_domain as d

SCENE=d.ROOT/'runs/contact_hold_size_r15_90mm_retarget_approach_20261112_084150_canonical_run/from_reset/seed_21611'
G=SCENE/'retarget_grasp90.json'
CFG=json.loads((SCENE/'config.json').read_text())


class DummyPlan:
    def __init__(self,n):self.accepted=torch.ones(n,dtype=torch.bool)
    def goal(self,phase,tick):
        return torch.zeros(len(phase),56),torch.full_like(tick,128)


class DomainTest(unittest.TestCase):
    def test_source_geometry_and_new_calibrated_reset_ik(self):
        src=d.verify_domain(SCENE,G)
        self.assertEqual(src['edge_m'],.09)
        fn=d._ik_builder(str(G.resolve()))
        # Wrapper around GPU entry point uses privately held constants, not 57mm cache.
        from contact_hold_cuda_ik import constants
        old=constants()
        new=fn.__wrapped__.__globals__['physical_constants']
        import numpy as np
        self.assertGreater(np.max(abs(new[1]-old[1])),.1)
        self.assertFalse(np.array_equal(old[0],new[0]))
        self.assertAlmostEqual(d.height_above_table(.945,.9),0,places=12)

    def test_reset_and_ready_start_at_the_same_retracted_wrist_pose(self):
        import numpy as np
        from kinematics import RobotKinematics
        k=RobotKinematics();cfg=d.reset_config(CFG);home=cfg['home']
        self.assertIsNot(home,CFG['home'])
        for side in ('left','right'):
            for link in (side+'_palm_link',side+'_wrist_camera_frame'):
                old=k.fk_link(CFG['home'],link);new=k.fk_link(home,link)
                np.testing.assert_allclose(new[:3,3]-old[:3,3],[-.020,0.,0.],atol=1e-4)
                np.testing.assert_allclose(new[:3,:3],old[:3,:3],atol=1e-4)
        self.assertTrue(all(home[n]==CFG['home'][n] for n in home if not n.startswith('Joint')))
        fn=d._ik_builder(str(G.resolve()))
        model,knots,arm,_,_=fn.__wrapped__.__globals__['physical_constants']
        # The native solver takes its 14 initial angles after 208 transforms and 28 joint limits.
        np.testing.assert_allclose(model[236:250],[home[CFG['joint_names'][i]] for i in arm],atol=1e-8)
        np.testing.assert_allclose(knots[0],[home[n] for n in CFG['joint_names']],atol=1e-8)

    def test_reset_rejects_bad_position_or_wrist_contact_before_learning(self):
        goal=torch.zeros(2,7)
        actual=goal.clone()
        s=SimpleNamespace(views={'Core':SimpleNamespace(get_transforms=lambda:actual)},hand_impulse=torch.zeros(2,2))
        actual[1,0]=.016
        with self.assertRaisesRegex(ValueError,'lane 1 seed 99'):d._check_reset(s,goal,[98,99])
        actual.zero_();s.hand_impulse[0,1]=.04
        with self.assertRaisesRegex(ValueError,'lane 0 seed 98'):d._check_reset(s,goal,[98,99])
        s.hand_impulse.zero_()
        self.assertEqual(d._check_reset(s,goal,[98,99])['lanes'],2)

    def test_palm_only_streak_and_20_frame_loss(self):
        n=3
        plan=DummyPlan(n); t=d.make_teacher(plan,CFG,{},G)
        self.assertEqual(t.g['kind'],'90mm_left_grasp_retarget_kinematic')
        # No finger feedback needed to exercise only the aggregate contact gates.
        t.fingers=SimpleNamespace(update=lambda flag,q,raw,contacts,core:raw)
        t.phase[:]=4;t.tick[:]=118
        q=torch.zeros(n,56);v=torch.zeros_like(q);core=torch.zeros(n,7);core[:,2]=.945;core[:,6]=1
        cv=torch.zeros(n,6); c=torch.zeros(n,13);hands=torch.tensor([[.004,0.],[0.,.006],[0.,0.]])
        def step():
            return t.action(torch.ones(n,dtype=torch.bool),q,v,core,q,c,cv,hands)
        for _ in range(10):step()
        self.assertEqual(t.phase.tolist(),[5,4,4])
        hands[0,0]=0.
        for _ in range(19):step()
        self.assertEqual(t.failure[0].item(),0)
        step()
        self.assertEqual(t.failure[0].item(),1)
        self.assertTrue(t.ever_touched[1])

if __name__=='__main__':
    unittest.main()
