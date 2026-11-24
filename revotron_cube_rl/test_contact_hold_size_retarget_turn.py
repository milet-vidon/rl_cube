import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import contact_hold_size_retarget_turn as turn
from test_contact_hold_physical_metrics import CFG,row

def samples(n):
    rows=[row(i)|dict(cube_quaternion_wxyz=[1.,0,0,0],q=[0.]*56) for i in range(n)]
    return rows

def valid_plan():
    phases=[]
    for stage in turn.PHASES:
        dest=[0.]*56
        if stage=='right_face_push':dest[-1]=.03
        phases.append((stage,[dest]))
    return dict(phases=phases,passed=True,face='nZ',target_deg=15.,edge_m=.09,
                contact_pad='right_index_DIP_rubber_link',contact_style='face',checks=64)

class TurnTest(unittest.TestCase):
    def test_adaptation_uses_90_mm_bodies_and_correct_pad_depth(self):
        src=turn.adapted_source();compile(src,'test','exec')
        self.assertIn('(.09,.09,.06)',src);self.assertIn('(.09,.09,.03)',src)
        self.assertIn('-.045+.0015',src)
        self.assertNotIn('(.057,.057,.019)',src)
        self.assertEqual(len(turn.validated_actions(valid_plan())),6)
        p=valid_plan();p['edge_m']=.057
        with self.assertRaises(ValueError):turn.validated_actions(p)

    def test_only_qualifying_measured_previous_hold_opens_preflight(self):
        cfg=dict(CFG,joint_names=['J']*56)
        env=SimpleNamespace(cfg=cfg);token=object(); rows=samples(40)
        # Table contact interrupts the qualifying streak.
        rows[9]['cube_table_contact']=True
        with patch.object(turn,'plan_for_row',return_value={'passed':False,'failure':'no_collision_free_path'}) as planner:
            trial=turn.GuidedFaceTrial(Path('/unused'))
            for i in range(40):
                act,ev=trial.action(env,rows[i-1] if i else None,token,i)
                self.assertIs(act,token);self.assertFalse(ev['guided_face_push_commanded'])
                self.assertEqual(ev['guided_face_state'],'waiting_for_hold')
            self.assertFalse(planner.called)
            act,ev=trial.action(env,rows[-1],token,40)
            self.assertEqual(planner.call_count,1)
            self.assertEqual(ev['guided_face_supported_at_post_step'],39)
            self.assertEqual(ev['guided_face_state'],'preflight_rejected')
            self.assertEqual(ev['guided_face_reason'],'no_collision_free_path')

    def test_wrong_contact_and_moving_cube_invalidate_scene(self):
        a,b=samples(2)
        self.assertEqual(turn.guard(a,b,[{'robot_body':'right_ring_DIP','cubelet':'nZ'}],'right_face_push'),
                         'unplanned_right_contact_body_or_layer')
        self.assertEqual(turn.guard(a,b,[{'robot_body':'right_index_DIP','cubelet':'nZ'}],'right_index_route'),
                         'premature_right_contact')
        b['cube_position'][0]+=.003
        self.assertEqual(turn.guard(a,b,[],'right_index_route'),'cube_moved_outside_contact_plan')

    def test_summary_disambiguates_requested_phase_and_executed_target(self):
        r=dict(guided_face_state='following_plan',guided_face_reason=None,guided_face_supported_at_post_step=0,
               executed_left_source='student_plus_size_plan_delta',guided_face_total_push_commands=0)
        rows=[r|dict(step=i,phase='tilt_90' if i else 'squeeze',guided_face_push_commanded=i>0,
                     guided_face_total_push_commands=i,guided_face_phase_commanded='right_face_push' if i else None,
                     applied_target=[0.]*56) for i in range(3)]
        rows[-1]['applied_target'][0]=.01
        rows[-1]['applied_target'][-1]=.005
        audit=turn.summarize_commands(rows)
        self.assertEqual(audit['actual_guided_face_push_frames'],1)
        self.assertEqual(audit['guided_face_push_phase_frames'],2)
        self.assertEqual(audit['actual_student_left_arm_changing_in_tilt_frames'],1)
        self.assertEqual(audit['first_applied_whole_tilt_step'],2)
        self.assertTrue(audit['preflight_requested'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch from Isaac runtime')
    def test_preserve_left_student_and_report_actual_right_phase(self):
        import torch
        cfg=dict(CFG,joint_names=[f'left_j{i}' for i in range(28)]+[f'right_j{i}' for i in range(28)])
        env=SimpleNamespace(cfg=cfg,targets=torch.zeros((1,56)),scales=torch.ones((1,56)))
        existing=torch.full((1,56),.05);obs=samples(41);trial=turn.GuidedFaceTrial('/unused')
        layer=SimpleNamespace(LayerContacts=lambda:SimpleNamespace(points=[]))
        with patch.dict(sys.modules,face_push_runtime=layer),patch.object(turn,'plan_for_row',return_value=valid_plan()),patch('contact_hold_size_face_score.face_samples',return_value=([0.],[True])):
            for i in range(34):
                act,ev=trial.action(env, obs[i-1] if i else None, existing, i)
                self.assertTrue(torch.equal(act[:,:28],existing[:,:28]))
            self.assertTrue(ev['guided_face_push_commanded']);self.assertTrue(ev['guided_face_push_command_moved_target'])
            self.assertAlmostEqual(float(act[0,-1]),.03)
            obs[33]['cube_table_contact']=True
            act,ev=trial.action(env,obs[33],existing,34)
            self.assertEqual(ev['guided_face_state'],'aborted');self.assertFalse(ev['guided_face_push_commanded'])
            self.assertTrue(torch.all(act[:,28:]==0));self.assertTrue(torch.equal(act[:,:28],existing[:,:28]))

if __name__=='__main__':unittest.main()
