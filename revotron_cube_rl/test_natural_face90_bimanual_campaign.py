"""Exercise curriculum stages, checkpoint continuity and dashboard state without physics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_face90_bimanual as campaign
from cube_dashboard import remote_reader as reader


class BimanualCampaignTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.home=self.root/'r27_recipe_bimanual_ppo'
        self.home.mkdir()
        self.pins={'policy.py':'0'*64}
        self.cfg=dict(lanes=8,iterations=3,validate_every=1,rng=327,source_sha256=self.pins,placo={'backend':'placo'},
                      schema=campaign.SCHEMA)
        campaign.save(self.home/'config.json',self.cfg)
        campaign.save(self.home/'state.json',campaign.initial())

    def test_seed_roles_and_old_actor_rejected(self):
        with patch.object(campaign,'fingerprint',return_value=self.pins):
            req=campaign.request(self.cfg,'train',campaign.block(campaign.TRAIN_BASE,1,8),1,2,None,None)
            campaign.validate_request(req)
            for change in [dict(checkpoint='single.pt',checkpoint_sha='e'*64),dict(target_deg=15),
                           dict(seeds=campaign.block(campaign.VALID_BASE,1,8)),dict(learn=False)]:
                with self.assertRaises(ValueError):
                    campaign.validate_request(dict(req,**change))
            req=campaign.request(self.cfg,'validate',campaign.block(campaign.VALID_BASE,2,8),2,15,'same.pt','e'*64)
            campaign.validate_request(req)
            with self.assertRaises(ValueError):
                campaign.validate_request(dict(req,independent_test=True))

    def test_stages_require_safety_and_contact_attributed_physics(self):
        base=dict(episodes=16,unsafe=0,goal_episodes=2,held_episodes=0)
        self.assertTrue(campaign.qualifies(2,base))
        self.assertFalse(campaign.qualifies(2,dict(base,unsafe=1)))
        self.assertFalse(campaign.qualifies(2,dict(base,goal_episodes=1)))
        for goal in [15,90]:
            self.assertFalse(campaign.qualifies(goal,dict(base,goal_episodes=16,held_episodes=13)))
            self.assertTrue(campaign.qualifies(goal,dict(base,held_episodes=14)))

    def test_train_and_validation_propagate_checkpoint_and_resume(self):
        train=[]
        panels=[]
        def wave(slot,req,fd):
            campaign.validate_request(req)
            path=slot/'attempt_12345678'
            path.mkdir(parents=True)
            campaign.save(slot/'request.json',req)
            if req['learn']:
                ck=path/'checkpoint.pt'
                ck.write_bytes(f"fixture iteration {req['iteration']}".encode())
                learning=dict(output_checkpoint=str(ck),output_checkpoint_sha=campaign.digest(ck))
                train.append((req['target_deg'],req['checkpoint'],str(ck)))
            else:
                learning={}
                panels.append((req['target_deg'],req['independent_test'],set(req['seeds']),req['checkpoint']))
            report=path/'report.json'
            campaign.save(report,dict(expert_learning=learning))
            campaign.save(slot/'accepted.json',dict(attempt=path.name,request_sha=campaign.digest(slot/'request.json'),
                                                      report_sha=campaign.digest(report)))
            row=dict(success=True,unsafe=False,natural_ready=True,turn_started=True,progress=1.,right_contact=.8,
                     posture_quality=.9,hold_fraction=1.,bilateral_frames=4,bimanual_goal_reached=True,
                     bimanual_turn_held=True,peak_bilateral_drive_deg=req['target_deg'])
            return [dict(row,seed=seed) for seed in req['seeds']],learning
        with patch.object(campaign,'fingerprint',return_value=self.pins), patch.object(campaign,'wave',side_effect=wave), \
                patch.object(campaign,'ROOT',self.root):
            campaign.campaign(self.home,self.cfg,1)
            campaign.campaign(self.home,self.cfg,1)
        self.assertEqual([x[0] for x in train],[2,15,90])
        self.assertEqual([x[1] for x in train],[None,train[0][2],train[1][2]])
        pools={g:set.union(*(s for goal,i,s,ck in panels if goal==g and i)) for g in [15,90]}
        self.assertEqual([len(pools[g]) for g in (15,90)],[32,32])
        self.assertFalse(pools[15]&pools[90])
        with patch.object(reader,'gpu',return_value={}):
            snap=reader.snapshot(self.root,self.home.name,{})
        self.assertTrue(snap['qualified'])
        self.assertEqual(snap['bimanual_validation']['target_deg'],90)
        self.assertEqual(snap['bimanual_validation']['held_episodes'],16)
        Path(train[-1][2]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            campaign.restore(self.home)

    def test_prepared_dashboard_shows_pending_and_correct_runner(self):
        with patch.object(reader,'gpu',return_value={}):
            snap=reader.snapshot(self.root,self.home.name,{})
        self.assertEqual((snap['kind'],snap['status'],snap['target_deg']),('bimanual_training','not_started',2))
        self.assertFalse(snap['qualified'])
        args=['python','natural_face90_bimanual.py']
        self.assertIsNone(reader.process_run([*args,'prepare'],self.root,self.root))
        self.assertEqual(reader.process_run([*args,'long'],self.root,self.root),self.home)
        self.assertEqual(reader.process_run([*args,'long','--run-dir',self.home.name],self.root,self.root),self.home)


if __name__=='__main__':
    unittest.main()
