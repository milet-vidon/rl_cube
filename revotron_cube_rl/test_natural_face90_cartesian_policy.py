import unittest

import torch

from natural_face90_cartesian_policy import (
    ACTION_AXES, ACTION_DIM, ACTOR_DIM, CRITIC_DIM, CartesianExpert, PadMotion,
    material_marker, pack_observation,
)
from natural_face90_recipe_policy import PPO


class CartesianPolicyTest(unittest.TestCase):
    def test_motion_is_frame_local_and_not_consumed_by_logging_reads(self):
        motion = PadMotion(2, 'cpu')
        p = torch.tensor([[.03, 0., -.052], [.02, 0., -.080]], dtype=torch.float64)
        step = torch.tensor([[.0001, -.0003, .0002], [.0002, .0003, .0005]], dtype=p.dtype)
        first = torch.tensor([True, False])
        both = torch.tensor([True, True])
        torch.testing.assert_close(motion.measure(42, p, first), torch.zeros_like(p))
        moved = motion.measure(43, p + step, first)
        torch.testing.assert_close(moved, torch.stack((step[0], torch.zeros_like(step[1]))))
        torch.testing.assert_close(motion.measure(43, p + step, both), moved)
        torch.testing.assert_close(motion.measure(44, p + 2 * step, both), step)
        torch.testing.assert_close(motion.measure(47, p + 4 * step, both), torch.zeros_like(p))

    def test_signed_nz_marker_and_normalized_features(self):
        radial = torch.tensor([[1., 0.]])
        face = torch.tensor([90.])
        marker = material_marker(radial, face)
        torch.testing.assert_close(marker, torch.tensor([[0., -.030, -.045]], dtype=marker.dtype))
        error = torch.tensor([[.003, -.005, .010]], dtype=marker.dtype)
        measured_delta = torch.tensor([[.0002, -.0003, .0004]], dtype=marker.dtype)
        yes = torch.tensor([True])
        no = ~yes
        sensors = dict(face_deg=face, support=yes, natural_ready=yes, static_contact=yes,
                       structure_ok=yes, allowed_layer_state=yes, wrong_contact=no,
                       right_total_impulse=torch.tensor([.002]), index_contact=yes,
                       other_digit_contact=no, contact_goal_reached=no)
        obs = pack_observation(marker - error, measured_delta, radial, sensors, torch.tensor([-3.]),
                               90, torch.tensor([.3]), yes, no, no)
        expected = torch.tensor([[.3, -.5, 1., .2, -.3, .4, 0., -1., 1., -3./90,
                                  1., 1., 1., 0., 1., 0., 0., 0., .3]])
        self.assertEqual(tuple(obs.shape), (1, 19))
        torch.testing.assert_close(obs, expected)

    def test_zero_mean_and_existing_ppo_update_use_three_latents(self):
        torch.manual_seed(33)
        self.assertEqual((ACTOR_DIM, CRITIC_DIM, ACTION_DIM), (19, 19, 3))
        self.assertEqual(ACTION_AXES, ('radial_out', 'positive_tangent', 'inward'))
        model = CartesianExpert()
        obs = torch.randn(12, 3, ACTOR_DIM)
        mean, _, _ = model.sample(obs, obs, deterministic=True)
        self.assertTrue((mean == 0).all())
        latent, logp, value = model.sample(obs, obs, deterministic=False)
        self.assertEqual(latent.shape, (12, 3, 3))
        self.assertEqual(logp.shape, value.shape)
        torch.testing.assert_close(logp, model.distribution(obs).log_prob(latent).sum(-1))
        done = torch.zeros_like(value, dtype=torch.bool)
        done[-1] = True
        data = dict(obs=obs, critic=obs, latent=latent, logp=logp, value=value,
                    reward=obs[..., 0], done=done, valid=torch.ones_like(done))
        update = PPO(model).update(data)
        self.assertEqual(update['real_transitions'], 36)
        self.assertGreater(update['actor_max_delta'], 0.)
        with self.assertRaises(ValueError):
            model.sample(torch.zeros(2, 198), torch.zeros(2, 391))


if __name__ == '__main__':
    unittest.main()
