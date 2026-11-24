import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import natural_face90_recipe_long as runner
from natural_face90_recipe_guide import ArmExpert, IndexApproach, guide_point, restore_expert
from natural_face90_recipe_policy import StateExpert
from natural_face90_v5_controller import NaturalFace90Controller


class GuidedAction(unittest.TestCase):
    def test_phases_are_measured_and_follow_outward_nz(self):
        radial=torch.tensor([[1.,0.]]*4,dtype=torch.double)
        tip=torch.tensor([[.08,0.,-.07],[.03,0.,-.06],[.03,0.,-.045],[.03,0.,-.045]])
        false=torch.zeros(4,dtype=torch.bool)
        goal=guide_point(radial,tip,torch.zeros(4),torch.tensor([0,0,1,0]).bool(),
                        torch.tensor([0,0,0,1]).bool())
        self.assertEqual(goal[0,1],0)
        torch.testing.assert_close(goal[:,2],torch.tensor([-.067,-.0445,-.0445,-.1],dtype=torch.double))
        self.assertLess(goal[2,1],0)
        self.assertEqual(guide_point(radial,tip,torch.zeros(4),false,false)[2,1],0)

    def test_arm_guide_holds_handoff_fingers_and_keeps_lane_offsets_out_of_ik(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        names=cfg['joint_names']
        home=torch.tensor([cfg['home'][name] for name in names])[None].float().repeat(2,1)
        step=torch.full((1,56),.012)
        e=SimpleNamespace(cfg=cfg,names=names,home=home,step=step,vel=30*step,
                          limits=torch.tensor([-3.15,3.15]).expand(1,56,2))
        c=NaturalFace90Controller(e,torch.zeros(2,6))
        g=IndexApproach(c)
        q=e.home.double(); arm=c.rp.ix
        c.radial[:,0]=1
        core=torch.tensor([[.45,.15,1.32,0.,0.,0.,1.]]*2,dtype=torch.double)
        tip=c.tip.fk(q[:,c.tip.ix])[0]+c.root
        shift=torch.tensor([[0.,0.,0.],[8.,-4.,0.]],dtype=torch.double)
        local=tip-core[:,:3]
        core[:,:3]+=shift
        yes=torch.ones(2,dtype=torch.bool)
        obs=dict(face_deg=torch.zeros(2),right_contact=~yes,contact_goal_reached=~yes)
        baseline=e.home.clone()
        # Vary the policy drastically; the passive handoff targets must never drift.
        for i in range(12):
            latent=torch.full((2,7),(-1.)**i*8.)
            cmd=g.command(baseline,q,e.home,yes if i==0 else ~yes,yes,latent,core,tip+shift,local,obs)
            passive=[j for j in range(56) if j not in arm]
            self.assertTrue(torch.equal(cmd[:,passive],e.home[:,passive]))
            self.assertLessEqual(float((cmd-baseline).abs().max()),.012001)
            torch.testing.assert_close(cmd[0],cmd[1])
            baseline=cmd

    def test_transfer_preserves_right_channel_mapping_and_resets_only_legacy_optimizer(self):
        actor=StateExpert()
        with torch.no_grad():actor.actor[-1].bias.copy_(torch.arange(28))
        ids=[20,3,16,1,22,10,5]
        model=ArmExpert(ids); opt=Mock()
        legacy={'schema':runner.LEGACY_SCHEMA,'learned_action_dim':28,'software':{'old':'sha'}}
        guided=dict(legacy,schema=runner.GUIDED_SCHEMA,learned_action_dim=7,software={'new':'sha'},
                    arm_indices=list(range(7)),action_mode='bounded_residual_on_index_pad_ik')
        ck=dict(contract=legacy,policy=actor.state_dict(),optimizer={'old':True},iteration=21)
        req=dict(role='train',transfer=dict(iteration=21,software=legacy['software']))
        restore_expert(model,opt,ck,guided,req)
        obs=torch.zeros(2,198)
        torch.testing.assert_close(model.distribution(obs).mean,actor.distribution(obs).mean[:,ids])
        opt.load_state_dict.assert_not_called()
        with self.assertRaises(ValueError):restore_expert(model,opt,dict(ck,iteration=20),guided,req)
        with self.assertRaises(ValueError):restore_expert(model,opt,dict(ck,contract={}),guided,req)
        restore_expert(model,opt,dict(ck,contract=guided),guided,{'role':'validate'})
        opt.load_state_dict.assert_called_once_with(ck['optimizer'])


class Continuation(unittest.TestCase):
    def test_busy_can_retry_and_worker_failure_does_not_retry(self):
        @contextmanager
        def acquire():
            if not acquire.ready:
                acquire.ready=True
                raise RuntimeError('GPU 0 正被使用；test')
            yield [123,456]
        acquire.ready=False
        with patch.object(runner,'gpu',acquire),patch.object(runner.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'worker'):
                with runner.available_gpu() as handles:
                    self.assertEqual(handles,[123,456])
                    raise RuntimeError('worker failed')
            sleep.assert_called_once_with(60)

    def test_continuation_uses_parent_then_validates_first_guided_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);old=home/'old.pt';old.write_bytes(b'old')
            new=home/'new.pt';new.write_bytes(b'new')
            origin=dict(iteration=21,checkpoint=str(old),sha=runner.digest(old),software={})
            cfg=dict(schema=runner.GUIDED_SCHEMA,lanes=32,iterations=23,validate_every=10,
                     validation_seeds=16,rng=327,source_sha256={},origin=origin)
            observed=[]
            def wave(slot,req,fd):
                runner.validate_request(req);observed.append(req);slot.mkdir(parents=True)
                return [],dict(output_checkpoint=str(new))
            # Stop the test at the second wave; confirm only iteration 22 imported legacy weights.
            def mock_wave(slot,req,fd):
                if req['iteration']==23:
                    self.assertNotIn('transfer',req)
                    self.assertEqual(req['checkpoint'],str(new))
                    raise InterruptedError
                return wave(slot,req,fd)
            with patch.object(runner,'wave',mock_wave),patch.object(runner,'fingerprint',return_value={}), \
                 patch.object(runner,'stats',return_value={}),patch.object(runner,'panel',return_value=dict(unsafe=1,successes=0)) as val:
                with self.assertRaises(InterruptedError):runner.campaign(home,cfg,0)
            self.assertEqual(observed[0]['transfer'],origin)
            self.assertEqual(observed[0]['seeds'][0],45_000_000+21*32)
            self.assertEqual(val.call_args.args[2:4],(22,str(new)))
            self.assertEqual(json.loads((home/'state.json').read_text())['iteration'],22)
