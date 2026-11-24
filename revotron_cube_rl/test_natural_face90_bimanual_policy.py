"""CPU checks for physical observations, reward attribution, and the existing PPO update."""
import unittest

import torch

from natural_face90_bimanual_policy import (
    ACTION_AXES, ACTION_DIM, ACTOR_DIM, BimanualExpert, BimanualReward, INPUT_FIELDS, REWARD_FIELDS,
    pack_observation,
)
from natural_face90_recipe_policy import PPO


def geometry(index_gap, thumb_gap):
    index = torch.tensor([[.045, 0., -.03]])
    thumb = torch.tensor([[-.045, 0., -.03]])
    return dict(goal_index_local=index, goal_thumb_local=thumb,
                index_local=index + torch.tensor([[index_gap, 0., 0.]]),
                thumb_local=thumb - torch.tensor([[thumb_gap, 0., 0.]]),
                palm_opposition_cos=torch.tensor([.96]), hand_q=torch.tensor([[.3, .4, .5]]))


def contact(thumb=False, index=False, bilateral=False, supported=True):
    b = lambda v: torch.tensor([v], dtype=torch.bool)
    ns = lambda value: torch.tensor([value], dtype=torch.float64)
    return dict(thumb=b(thumb), index=b(index), bilateral=b(bilateral), supported=b(supported),
                left_static=b(supported), overlap=torch.tensor([8 if bilateral else 0]),
                thumb_Ns=ns(.02 if thumb else 0.), index_Ns=ns(.03 if index else 0.),
                left_static_Ns=ns(.6 if supported else 0.), left_target_Ns=ns(0.),
                other_target_Ns=ns(0.))


class BimanualPolicyTest(unittest.TestCase):
    def setUp(self):
        self.reward = BimanualReward(1, 'cpu')
        self.far = geometry(.09, .09)
        self.near = geometry(.001, .001)

    def step(self, before, after, contacts, credit=0., unsafe=False, success=False, valid=True):
        masks = [torch.tensor([b]) for b in (valid, unsafe, success)]
        return self.reward.update(*masks, before, after, contacts, torch.tensor([credit]),
                                  15., torch.zeros(1, ACTION_DIM)).item()

    def part(self, name):
        return self.reward.totals[0, REWARD_FIELDS.index(name)].item()

    def test_geometry_and_hand_readbacks_in_core_frame(self):
        g = geometry(.01, .02)
        c = contact(True, False)
        m = dict(face_deg=torch.tensor([1.5]), wrong_contact=torch.tensor([False]),
                 structure_ok=torch.tensor([True]))
        obs = pack_observation(g, c, m, torch.tensor([.5]), 15, torch.tensor([.3]))
        self.assertEqual(obs.shape, (1, ACTOR_DIM))
        values = dict(zip(INPUT_FIELDS, obs[0].tolist(), strict=True))
        self.assertAlmostEqual(values['index_error_core_x_cm'], -1., places=6)
        self.assertAlmostEqual(values['thumb_error_core_x_cm'], 2., places=6)
        self.assertAlmostEqual(values['right_index_mcp_rad'], .4, places=6)
        self.assertAlmostEqual(values['goal_over_90'], 1./6, places=6)
        self.assertAlmostEqual(values['measured_face_over_goal'], .1, places=6)
        self.assertEqual((values['thumb_contact'], values['bilateral_contact']), (1., 0.))

    def test_worst_finger_approach_recontact_and_static_hold(self):
        one_far = geometry(.09, 0.)
        self.assertGreater(self.step(self.far, one_far, contact()), 0.)
        first = self.part('approach')
        self.assertGreater(self.step(one_far, self.near, contact()), first)
        peak = self.part('approach')
        self.assertGreater(peak, first)
        self.step(self.near, self.far, contact())
        self.step(self.far, self.near, contact())
        self.assertAlmostEqual(self.part('approach'), peak)

        self.step(self.near, self.near, contact(True, True, False), credit=2.)
        self.assertEqual(self.part('thumb_first'), .5)
        self.assertEqual(self.part('index_first'), .5)
        self.assertEqual(self.part('bilateral_first'), 0.)
        self.assertEqual(self.part('drive'), 0.)
        both = contact(True, True, True)
        self.step(self.near, self.near, both, credit=-3.)
        self.assertEqual(self.part('bilateral_first'), 2.)
        self.assertEqual(self.part('drive'), 0.)
        self.step(self.near, self.near, both, credit=.4)
        earned = self.part('drive')
        self.assertGreater(earned, 0.)
        self.step(self.near, self.near, contact(), credit=15.)
        self.step(self.near, self.near, both, credit=.2)
        self.assertLess(self.step(self.near, self.near, both, credit=.4), 0.)
        self.assertAlmostEqual(self.part('drive'), earned)
        self.assertEqual(self.part('bilateral_first'), 2.)

    def test_no_shaping_without_support_or_when_unsafe(self):
        unsupported = contact(True, True, True, False)
        self.assertLess(self.step(self.far, self.near, unsupported, credit=3.), 0.)
        both = contact(True, True, True)
        self.assertEqual(self.step(self.far, self.near, both, credit=4., valid=False), 0.)
        self.assertLess(self.step(self.far, self.near, both, credit=15., success=True, unsafe=True), -20.)
        self.assertEqual(self.step(self.far, self.near, both, credit=15., success=True), 0.)
        for name in ('approach', 'thumb_first', 'index_first', 'bilateral_first', 'drive', 'success'):
            self.assertEqual(self.part(name), 0.)

    def test_success_requires_verified_peak_and_can_follow_release(self):
        self.step(self.near, self.near, contact(), success=True)
        self.assertEqual(self.part('success'), 0.)
        self.step(self.near, self.near, contact(True, True, True), credit=15.)
        self.step(self.near, self.near, contact(), success=True)
        self.assertEqual(self.part('success'), 20.)

    def test_independent_lanes_and_ppo(self):
        torch.manual_seed(55)
        self.assertEqual(ACTION_AXES, ('cube_x','cube_y','cube_z','thumb_flex','index_flex',
                                       'thumb_opposition','face_orbit'))
        reward = BimanualReward(2, 'cpu')
        repeat = lambda d: {k:v.repeat_interleave(2, dim=0) for k,v in d.items()}
        both = repeat(contact(True, True, True))
        b = torch.tensor([True, False])
        reward.update(b, ~b, ~b, repeat(self.far), repeat(self.near), both, torch.tensor([2.,2.]),
                      2., torch.zeros(2,ACTION_DIM))
        self.assertGreater(float(reward.totals[0].sum()), 0.)
        self.assertTrue(torch.equal(reward.totals[1], torch.zeros_like(reward.totals[1])))

        model = BimanualExpert()
        obs = torch.randn(16, 2, ACTOR_DIM)
        self.assertTrue((model.sample(obs, obs, deterministic=True)[0] == 0).all())
        latent, logp, value = model.sample(obs, obs)
        self.assertEqual(latent.shape, (16, 2, ACTION_DIM))
        done = torch.zeros_like(logp, dtype=torch.bool)
        done[-1] = True
        data = dict(obs=obs, critic=obs, latent=latent, logp=logp, value=value,
                    reward=obs[...,0], valid=torch.ones_like(done), done=done)
        result = PPO(model).update(data)
        self.assertEqual(result['real_transitions'], 32)
        self.assertGreater(result['actor_max_delta'], 0.)


if __name__ == '__main__':
    unittest.main()
