import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import natural_face90_cartesian_contact_worker as stage
from test_natural_face90_cartesian_worker import Fixture


class ContactWorkerTests(unittest.TestCase):
    def test_rocking_dropouts_and_unsafe_do_not_mint_drive_rewards(self):
        reward = stage.ContactReward(1, 'cpu')
        b = lambda x:torch.tensor([x],dtype=torch.bool)
        f = lambda x:torch.tensor([x],dtype=torch.float64)
        def step(angle, first=False, active=True, unsafe=False):
            reward.update(b(True),b(unsafe),b(not first),b(True),b(True),b(active),f(angle),
                          b(True),f(.02),f(.01),torch.zeros(1,3))
        step(0.,first=True,active=False)
        self.assertEqual(reward.totals[0,1],stage.FIRST_BONUS)
        step(.014)
        step(.014)
        small = float(reward.totals[0,2])
        self.assertGreater(small,.5)
        step(-.07)
        step(.014)
        step(.014)
        self.assertEqual(float(reward.totals[0,2]),small)
        step(2.3)
        step(0.,active=False)
        step(2.3)
        self.assertEqual(float(reward.totals[0,2]),small)
        step(2.3)
        self.assertAlmostEqual(float(reward.totals[0,2]),stage.DRIVE_MAX)
        step(3.,unsafe=True)
        self.assertLess(reward.totals.sum(),0.)

    def test_physical_replay_matches_online_credit_and_refuses_changed_first_contact(self):
        reward = stage.ContactReward(1,'cpu')
        meter = stage._Replay()
        frames = []
        for step,angle in enumerate([0.,0.,.008,.018,.018,.015,.018,.11,.14]):
            first = step == 1
            touch = step >= 1
            active = step >= 2
            unsafe = step == 8
            f = lambda value: torch.tensor([value],dtype=torch.float64)
            b = lambda value: torch.tensor([value],dtype=torch.bool)
            reward.update(b(True),b(unsafe),b(step>1),b(touch),b(touch and not unsafe),b(active and not unsafe),
                          f(angle),b(True),f(.05 if step==0 else .03),f(.03),torch.zeros(1,3))
            previous = frames[-1]['face_deg'] if frames else 0.
            row = dict(lane=0,seed=94,step=step,pre_contact=step>1,pre_index_contact=step>1,
                pre_other_digit_contact=False,index_distal_impulse_Ns=.1 if touch else 0.,
                other_digit_target_impulse_Ns=0.,right_target_impulse_Ns=.1 if touch else 0.,
                qualified_index_frame=touch and not unsafe,pre_support_ok=True,drive_command=active,
                face_deg=angle,pre_material_face_deg=previous,pre_material_marker_x_m=.030,
                pre_material_marker_y_m=0.,pre_cartesian_pad_x_m=.030,pre_cartesian_pad_y_m=0.,
                pre_cartesian_pad_z_m=-.095 if step==0 else -.075,pad_x_m=.030,pad_y_m=0.,pad_z_m=-.075,
                pre_cartesian_latent_0=0.,pre_cartesian_latent_1=0.,pre_cartesian_latent_2=0.)
            meter.step(row,unsafe)
            frames.append(row)
        np.testing.assert_allclose(reward.totals[0],meter.totals,rtol=1e-8)
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            def save(name, rows):
                (home/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save('turn_contact_evidence.jsonl',frames)
            save('policy_transitions.jsonl',[dict(terminal=True,lane=0,seed=94,learned_steps=len(frames),unsafe=True,
                                                  return_sum=float(reward.totals.sum()))])
            learn = dict(curriculum_stage='contact2',reward_contract=stage.reward_contract('contact2'),
                         contact_reward_fields=list(stage.REWARD_FIELDS),contact_reward_totals_by_lane=reward.totals.tolist(),
                         learned_transitions=len(frames))
            req = dict(curriculum_stage='contact2',seeds=[94])
            stage.verify_stage_report(home/'report.json',req,[dict(unsafe=True)],learn)
            frames[1]['qualified_index_frame']=False
            save('turn_contact_evidence.jsonl',frames)
            with self.assertRaisesRegex(ValueError,'differ'):
                stage.verify_stage_report(home/'report.json',req,[dict(unsafe=True)],learn)

    def test_actual_ppo_and_promotion_keep_weights_optimizer_and_stage_lineage(self):
        from natural_face90_placo_client import self_check
        fixture = Fixture(32,self_check()['receipt'])
        req = dict(recipe_schema=stage.CONTACT_SCHEMA,curriculum_stage='contact2',policy_seed=5,target_deg=15,
                   learn=True,iteration=1,role='train',software={},seeds=list(range(32)),
                   pad_orientation_geometry={},geometry_sha256='g',scene_sha256='s',
                   pad_preshape_offset_deg=-6, contact_point_offset_deg=7,checkpoint=None,checkpoint_sha=None,
                   placo=fixture.placo_receipt)
        target, steps, all_lanes = fixture.home, torch.ones(32,dtype=torch.long), torch.ones(32,dtype=torch.bool)
        def create(req, home):
            home.mkdir()
            with patch.object(stage.cart,'CartesianPhase',stage.ContactPhase):
                phase = fixture.create(req,home)
            phase.guide.control_evidence=lambda:dict(pre_support_ok=all_lanes,pre_contact=~all_lanes,
                                                     pre_index_contact=~all_lanes,pre_other_digit_contact=~all_lanes)
            phase.obs['target_impulse']=torch.zeros(32)
            return phase
        with (tempfile.TemporaryDirectory() as d, patch.object(stage,'check_software',return_value={}),
              patch.object(stage.base,'fingerprint',return_value={}),
              patch.object(stage.cart.recipe.ScriptedPostPhase,'propose',lambda self,t,*a:t.clone())):
            d=Path(d)
            contact = create(req,d/'contact')
            contact.propose(target,all_lanes,steps)
            fixture.pad[:,2] += .002
            contact.monitor.last_step += 1
            contact.after_step(all_lanes,all_lanes,~all_lanes,['']*32,steps)
            steps += 1
            contact.propose(target,all_lanes,steps)
            contact.monitor.last_step += 1
            contact.after_step(all_lanes,~all_lanes,~all_lanes,['post_horizon']*32,steps)
            learning=contact.finish_learning()
            self.assertTrue(learning['actor_updated'])
            ck=learning['output_checkpoint']
            sha=learning['output_checkpoint_sha']
            face_req=dict(req,curriculum_stage='face15',recipe_schema=stage.FACE_SCHEMA,iteration=2,
                          checkpoint=ck,checkpoint_sha=sha,contact_origin=dict(checkpoint=ck,sha=sha,iteration=1))
            face=create(face_req,d/'face')
            for a,b in zip(face.model.parameters(),contact.model.parameters(),strict=True):
                torch.testing.assert_close(a,b)
            torch.testing.assert_close(face.algorithm.optimizer.state_dict(),contact.algorithm.optimizer.state_dict())
            face.propose(target,all_lanes,steps)
            face.after_step(all_lanes,~all_lanes,~all_lanes,['post_horizon']*32,steps)
            successor=face.finish_learning()
            self.assertEqual(successor['reward_contract'],stage.reward_contract('face15'))
            saved=torch.load(successor['output_checkpoint'],weights_only=True)
            self.assertEqual(saved['contract']['curriculum_stage'],'face15')
            self.assertEqual(saved['prior_checkpoint_sha'],sha)
            with self.assertRaisesRegex(ValueError,'Expected the committed'):
                stage.restore_stage(face.model,None,torch.load(ck,weights_only=True),face.contract,
                                    {k:v for k,v in face_req.items() if k!='contact_origin'})


if __name__ == '__main__':
    unittest.main()
