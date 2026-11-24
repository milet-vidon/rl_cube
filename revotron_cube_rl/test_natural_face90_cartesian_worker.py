"""Exercise worker handoff, temporal observations, PPO checkpointing and validation on CPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import natural_face90_cartesian as campaign
import natural_face90_cartesian_worker as worker


HERE = Path(__file__).resolve().parent


class Fixture:
    def __init__(self, n, placo_receipt):
        self.n = n
        self.placo_receipt = placo_receipt
        self.home = torch.zeros(n, 56)
        self.ids = range(56)
        self.names = json.loads((HERE / 'config.json').read_text())['joint_names']
        self.core = torch.tensor([0., 0., 0., 0., 0., 0., 1.], dtype=torch.float64).repeat(n, 1)
        self.pad = torch.tensor([.029, 0., -.073], dtype=torch.float64).repeat(n, 1)
        self.stage = object()
        self.views = {'Core': SimpleNamespace(get_transforms=lambda: self.core)}
        self.robot = SimpleNamespace(get_dof_positions=lambda: self.home,
                                     shared_metatype=SimpleNamespace(link_names=['right_index_DIP_link']))
        self.qpad = torch.tensor(json.loads((HERE / 'face_contact_calibration_v1.json').read_text())['point_local'],
                                 dtype=torch.float64)

    def scripted_init(self, phase, e, req, out):
        phase.e, phase.req, phase.n, phase.device, phase.out = e, req, e.n, e.home.device, out
        phase.monitor = SimpleNamespace(last_step=10)
        phase.software = {}
        phase.phase_names = ('scripted',)
        phase.retract = torch.zeros(e.n, dtype=torch.bool)
        phase.action_phase = torch.zeros(e.n, dtype=torch.long)
        phase.controller = SimpleNamespace(radial=torch.tensor([[1.,0.]], dtype=torch.float64).repeat(e.n,1),
                                           detour_done=~phase.retract, phase=phase.action_phase + 2)

    def make_guide(self, controller, geometry):
        n = self.n
        flags = torch.zeros(n, dtype=torch.bool)
        count = torch.zeros(n, dtype=torch.long)
        return SimpleNamespace(
            placo=SimpleNamespace(receipt=self.placo_receipt, close=Mock()),
            meshes=[None, None, self.qpad], arm=list(range(28,35)), aligned=flags, drive_contact=flags,
            recovering=flags, recovery_reason=count, streak=count, hover_frames=count, anchor_frames=count,
            handoff_error=count.double(), azimuth=count.double(), receipt=lambda: {},
            command=lambda target,q,handoff,fresh,learned,latent,*a: self.capture(target, latent),
        )

    def capture(self, target, latent):
        self.last_latent = latent.clone()
        return target.clone()

    def measurements(self):
        zero = torch.zeros(self.n, dtype=torch.bool)
        obs = {k: zero.clone() for k in ('wrong_contact','right_contact','index_contact','other_digit_contact',
                                         'contact_goal_reached','turn_started')}
        obs.update({k: ~zero for k in ('natural_ready','support','static_contact','structure_ok','allowed_layer_state')})
        obs.update({k: zero.double() for k in ('face_deg','progress','right_total_impulse','hold_fraction',
                                              'actual_retraction_m')})
        return obs

    def create(self, req, folder):
        with patch.object(worker.recipe.ScriptedPostPhase, '__init__',
                          lambda phase,e,req,out: self.scripted_init(phase,e,req,out)), \
             patch.object(worker.control,'CartesianGuide',self.make_guide), \
             patch('natural_face90_recipe_frames.contact_frame',
                   return_value=('right_index_DIP_link',[0.,0.,0.],[1.,0.,0.,0.])), \
             patch('natural_face90_turn_contacts.install',
                   return_value=SimpleNamespace(pad_position_world=lambda:self.pad)):
            phase = worker.CartesianPhase(self, req, folder)
        phase.obs = self.measurements()
        return phase


class CartesianWorkerTest(unittest.TestCase):
    def test_training_checkpoint_reopens_through_normal_validation_initialization(self):
        from natural_face90_placo_client import self_check
        fixture = Fixture(32, self_check()['receipt'])
        req = dict(recipe_schema='natural_face90_recipe_cartesian_ppo_v1', policy_seed=19, target_deg=15,
                   learn=True, iteration=1, role='train', software={}, seeds=list(range(fixture.n)),
                   pad_orientation_geometry={}, geometry_sha256='geo', scene_sha256='scene',
                   pad_preshape_offset_deg=-6, contact_point_offset_deg=7, checkpoint=None, checkpoint_sha=None,
                   placo=fixture.placo_receipt)
        wrong_guide = fixture.make_guide(None, {})
        with patch.object(worker.control, 'CartesianGuide', return_value=wrong_guide), \
             self.assertRaisesRegex(ValueError, 'Placo runtime'):
            worker.cartesian_guide(None, dict(req, placo=dict(req['placo'], urdf_sha256='0'*64)))
        wrong_guide.placo.close.assert_called_once()
        target = fixture.home
        steps = torch.ones(fixture.n, dtype=torch.long)
        active = torch.ones_like(steps, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, 'fingerprint', return_value={}), \
             patch.object(worker.recipe.ScriptedPostPhase, 'propose', lambda self,target,*a: target.clone()):
            train = Path(tmp)/'train'
            train.mkdir()
            phase = fixture.create(req, train)
            self.assertEqual(phase.contract['learned_action_dim'], 3)
            first_lanes = active.clone()
            first_lanes[-8:] = False
            phase.propose(target, first_lanes, steps)
            self.assertEqual(fixture.last_latent.shape, (32,3))
            self.assertEqual(phase.pending['obs'].shape, (32,19))
            torch.testing.assert_close(phase.pending['valid'], first_lanes)
            torch.testing.assert_close(phase.pending['obs'][:,3:6], torch.zeros(32,3))
            torch.testing.assert_close(phase.model.actor(phase.pending['obs']), torch.zeros(32,3))
            fixture.pad[:,2] += .0004
            fixture.core[:,0] += .18
            fixture.pad[:,0] += .18
            phase.monitor.last_step += 1
            # Physical terminal readback is re-read by reward, logger and bootstrap code.
            obs, critic, _, _ = phase.state(target, steps.float())
            torch.testing.assert_close(obs[first_lanes,3:6], torch.tensor([0.,0.,.4]).expand(24,3))
            torch.testing.assert_close(obs[~first_lanes,3:6], torch.zeros(8,3))
            torch.testing.assert_close(critic, obs)
            torch.testing.assert_close(phase.state(target,steps.float())[0], obs)
            phase.after_step(first_lanes, active, ~active, ['']*32, steps)
            steps += 1
            phase.propose(target, active, steps)
            torch.testing.assert_close(phase.pending['valid'], active)
            torch.testing.assert_close(phase.pending['obs'][~first_lanes,3:6], torch.zeros(8,3))
            phase.monitor.last_step += 1
            unsafe = torch.arange(32) % 3 == 0
            phase.after_step(active, ~active, unsafe, ['post_horizon']*32, steps)
            learn = phase.finish_learning()
            phase.guide.placo.close.assert_called_once()
            self.assertTrue(learn['actor_updated'])
            self.assertEqual(learn['real_transitions'], 56)
            self.assertEqual(learn['cartesian'], worker.cartesian_receipt(req))
            self.assertEqual(learn['placo'], phase.guide.placo.receipt)
            self.assertEqual(learn['pad_preshape']['orientation'],learn['pad_orientation'])
            checkpoint = Path(learn['output_checkpoint'])
            val_req = dict(req, role='validate', learn=False, checkpoint=str(checkpoint),
                           checkpoint_sha=learn['output_checkpoint_sha'])
            val = Path(tmp)/'validate'
            val.mkdir()
            forward = fixture.create(val_req,val)
            for value, restored in zip(phase.model.parameters(), forward.model.parameters(), strict=True):
                torch.testing.assert_close(value,restored)
            forward.propose(target, active, steps)
            frozen = forward.finish_learning()
            forward.guide.placo.close.assert_called_once()
            self.assertFalse(frozen['actor_updated'])
            self.assertTrue(frozen['deterministic'])
            self.assertFalse((val/'checkpoint.pt').exists())

            ck = torch.load(checkpoint, weights_only=True, map_location='cpu')
            for wrong in (dict(ck, iteration=0), dict(ck, contract=dict(ck['contract'], learned_action_dim=7)),
                          dict(ck, contract=dict(ck['contract'], placo={'backend':'other'}))):
                with self.assertRaisesRegex(ValueError, 'matching Cartesian'):
                    worker.restore_cartesian(forward.model, None, wrong, forward.contract, val_req)

            resume_dir = Path(tmp)/'resume'
            resume_dir.mkdir()
            resumed = fixture.create(dict(val_req, role='train', learn=True, iteration=2), resume_dir)
            self.assertEqual(resumed.algorithm.optimizer.state_dict()['param_groups'], ck['optimizer']['param_groups'])
            torch.testing.assert_close(resumed.algorithm.optimizer.state_dict()['state'], ck['optimizer']['state'])
            resumed.propose(target, active, steps)
            resumed.monitor.last_step += 1
            resumed.after_step(active, ~active, unsafe, ['post_horizon']*32, steps)
            updated = resumed.finish_learning()
            child = torch.load(updated['output_checkpoint'], weights_only=True, map_location='cpu')
            self.assertTrue(updated['actor_updated'])
            self.assertEqual(child['prior_checkpoint_sha'], learn['output_checkpoint_sha'])
            self.assertEqual(child['iteration'], 2)


if __name__ == '__main__':
    unittest.main()
