"""Synthetic action/label contracts only; no dataset or physical qualification."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_phase_residual_v5 import baseline, observation
from test_zero_gated_residual_policy_v7 import frame, PARAMETERS, joint_safety
import arm_hold_increment_policy_v9 as p


def policy(names=p.EXPECTED_LEFT_NAMES):
    return p.HoldIncrementPolicy(baseline(), np.zeros(p.FEATURES), np.ones(p.FEATURES),
                                p.candidate_bounds(names), hidden_size=8, left_joint_names=names)


def force_modes(model, mode):
    model.mode_head.weight.data.zero_()
    model.mode_head.bias.data.reshape(28, 3).fill_(-1.)
    model.mode_head.bias.data.reshape(28, 3)[:, mode] = 1.


def project(value, previous):
    return p.project_left_targets(value, previous, PARAMETERS, list(range(28)), joint_safety.project_target)


class HoldIncrementContracts(unittest.TestCase):
    def test_initial_tie_is_exact_baseline(self):
        m = policy(); m.head.bias.data.fill_(100.)
        x = frame(); prev = torch.full((1, 1, 28), .123)
        out, hidden, update, d = m.forward_with_diagnostics(x, prev)
        self.assertTrue(torch.equal(out.view(torch.int32), d['baseline_proposal'].view(torch.int32)))
        self.assertEqual(torch.count_nonzero(d['predicted_class']), 0)
        self.assertEqual(torch.count_nonzero(update), 0)
        self.assertGreater(torch.count_nonzero(hidden), 0)

    def test_hold_represents_target_far_outside_moving_base_residual(self):
        m = policy(); force_modes(m, p.HOLD)
        x = frame(); prev = torch.full((1, 1, 28), .123); base = prev + .02
        with patch.object(m, 'baseline_targets', return_value=base):
            out, _, update, d = m.forward_with_diagnostics(x, prev)
        self.assertTrue(torch.equal(out.view(torch.int32), prev.view(torch.int32)))
        self.assertTrue(torch.equal(project(out, prev).view(torch.int32), project(prev, prev).view(torch.int32)))
        self.assertGreater(float((base - out)[0, 0, 21].detach()), .006)
        self.assertEqual(torch.count_nonzero(update), 0)
        self.assertTrue((d['predicted_class'] == p.HOLD).all())

    def test_update_uses_previous_only_for_named_arm_and_recovery_joints(self):
        m = policy(); force_modes(m, p.UPDATE); m.head.bias.data.fill_(.5)
        x = frame(); prev = torch.full((1, 1, 28), .1); base = prev + .02
        with patch.object(m, 'baseline_targets', return_value=base):
            out, _, update, d = m.forward_with_diagnostics(x, prev)
        mask = m.update_anchor_is_previous
        self.assertEqual(int(mask.sum()), 18)
        self.assertTrue(torch.equal(d['update_anchor'][..., mask], prev[..., mask]))
        self.assertTrue(torch.equal(d['update_anchor'][..., ~mask], base[..., ~mask]))
        self.assertTrue(torch.allclose(out, d['update_anchor'] + update))
        self.assertTrue((update.abs() <= m.residual_bound).all())

    def test_anchor_mask_tracks_explicit_permuted_names(self):
        m = policy(p.EXPECTED_LEFT_NAMES[::-1])
        expected = [n in p.PREVIOUS_ANCHOR_JOINTS for n in p.EXPECTED_LEFT_NAMES[::-1]]
        self.assertEqual(m.update_anchor_is_previous.tolist(), expected)

    def test_hold_update_tie_chooses_hold(self):
        m = policy(); m.mode_head.bias.data.reshape(28, 3)[:, 1:] = 1.
        prev = torch.full((1, 1, 28), .12)
        out, _, _, d = m.forward_with_diagnostics(frame(), prev)
        self.assertTrue((d['predicted_class'] == p.HOLD).all())
        self.assertTrue(torch.equal(out, prev))

    def test_original_public_gate_forces_base_and_advances_history(self):
        m = policy(); force_modes(m, p.HOLD)
        for x in (frame('ready', 1.), frame('squeeze', 71/84)):
            prev = torch.full((1, 1, 28), .12)
            out, hidden, _, d = m.forward_with_diagnostics(x, prev)
            self.assertTrue(torch.equal(out, d['baseline_proposal']))
            self.assertFalse(d['public_phase_active'].any())
            self.assertGreater(torch.count_nonzero(hidden), 0)
        self.assertTrue(m.residual_active(frame('squeeze', 72/84)).all())

    def test_classification_base_overlap_hold_update_and_ambiguous(self):
        base = torch.zeros(1, 1, 28); hold = torch.full_like(base, .01); labels = base.clone()
        hold[..., 0] = base[..., 0]  # Both exact: BASE wins.
        labels[..., 1] = hold[..., 1]
        labels[..., 2] = .02
        labels[..., 3] = .0000005
        labels[..., 4] = hold[..., 4] + .0000005
        original = labels.clone(); d = p.classify_branch_labels(labels, base, hold)
        self.assertEqual(d['class_targets'][0, 0, :5].tolist(), [0, 1, 2, 2, 2])
        self.assertEqual(d['classification_mask'][0, 0, :5].tolist(), [True, True, True, False, False])
        self.assertEqual(int(d['ambiguous'].sum()), 2)
        self.assertTrue(torch.equal(labels.view(torch.int32), original.view(torch.int32)))

    def test_signed_zero_is_bitwise_ambiguous_not_exact(self):
        base = torch.zeros(1, 1, 28); hold = torch.ones_like(base)
        labels = base.clone(); labels[..., 0] = -0.
        d = p.classify_branch_labels(labels, base, hold)
        self.assertTrue(d['ambiguous'][0, 0, 0])
        self.assertFalse(d['exact_base'][0, 0, 0])

    def loss_fixture(self):
        m = policy(); prev = torch.full((1, 1, 28), .1); base = prev + .02
        with patch.object(m, 'baseline_targets', return_value=base):
            out, _, _, d = m.forward_with_diagnostics(frame(), prev)
        labels = project(base, prev).detach().clone()
        labels[..., 21] = project(prev, prev)[..., 21]
        labels[..., 22] = project(prev + .003, prev)[..., 22]
        labels[..., 23] = project(prev + .0000005, prev)[..., 23]
        c = p.classify_branch_labels(labels, project(base, prev), project(prev, prev))
        return m, prev, out, d, labels, c

    def test_regression_and_ce_keep_raw_labels_and_supply_gradients(self):
        m, prev, out, d, labels, c = self.loss_fixture(); original = labels.clone()
        losses = p.supervision_losses(out, d, prev, labels, torch.ones((1, 1), dtype=torch.bool),
                                      c['class_targets'], c['classification_mask'], PARAMETERS,
                                      list(range(28)), joint_safety.project_target)
        self.assertEqual(losses['continuous_label_count'], 28)
        self.assertEqual(losses['ambiguous_continuous_label_count'], 1)
        self.assertGreater(float(losses['hard_decoded_mse_rad2'].detach()), float(losses['regression_mse_rad2'].detach()))
        (losses['regression_mse_rad2'] * 1e6 + losses['branch_cross_entropy']).backward()
        self.assertGreater(float(m.head.bias.grad.abs().sum()), 0.)
        self.assertGreater(float(m.mode_head.bias.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(labels.view(torch.int32), original.view(torch.int32)))

    def test_label_tampering_is_rejected(self):
        _, prev, out, d, labels, c = self.loss_fixture()
        for key in ('class_targets', 'classification_mask'):
            bad = deepcopy(c)
            if key == 'class_targets': bad[key][..., 21] = p.BASE
            else: bad[key][..., 23] = True
            with self.assertRaises(ValueError):
                p.supervision_losses(out, d, prev, labels, torch.ones((1, 1), dtype=torch.bool),
                                     bad['class_targets'], bad['classification_mask'], PARAMETERS,
                                     list(range(28)), joint_safety.project_target)

    def test_loading_rejects_changed_anchor_gate_bounds_and_baseline(self):
        m = policy(); original = deepcopy(m.state_dict())
        for key in ('update_anchor_is_previous', 'residual_phase_active', 'squeeze_active_progress',
                    'residual_bound', 'mean', 'std', 'training_class_confidence_threshold_rad',
                    next(k for k in original if k.startswith('baseline.'))):
            state = deepcopy(original)
            if state[key].dtype == torch.bool: state[key].view(-1)[0] = ~state[key].view(-1)[0]
            else: state[key].view(-1)[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): m.load_state_dict(state)
        with self.assertRaises(ValueError): m.load_state_dict(original, strict=False)
        m.load_state_dict(original)

    def test_signed_zero_frozen_buffer_change_rejected(self):
        m = policy(); state = deepcopy(m.state_dict())
        self.assertEqual(float(state['mean'][0]), 0.)
        state['mean'][0] = -0.
        with self.assertRaises(ValueError): m.load_state_dict(state)

    def test_false_hold_and_nonfinite_update_diagnostic_rejected(self):
        _, prev, out, d, labels, c = self.loss_fixture()
        for key in ('hold_proposal', 'update_proposal'):
            bad = dict(d); bad[key] = bad[key].clone()
            bad[key][..., 0] = .15 if key == 'hold_proposal' else float('nan')
            with self.assertRaises(ValueError):
                p.supervision_losses(out, bad, prev, labels, torch.ones((1, 1), dtype=torch.bool),
                                     c['class_targets'], c['classification_mask'], PARAMETERS,
                                     list(range(28)), joint_safety.project_target)

    def test_nonfinite_weights_input_and_invalid_history_rejected(self):
        m = policy(); state = deepcopy(m.state_dict()); state['mode_head.bias'][0] = float('nan')
        with self.assertRaises(ValueError): m.load_state_dict(state)
        x = frame(); prev = torch.zeros(1, 1, 28)
        with self.assertRaises(ValueError): m(x.double(), prev)
        with self.assertRaises(ValueError): m(x, prev, torch.zeros(1, 1, 9))
        x[..., 0] = float('nan')
        with self.assertRaises(ValueError): m(x, prev)

    def test_untrained_streaming_and_batch_hidden_are_equal_within_float_precision(self):
        torch.manual_seed(23); m = policy()
        x = frame().repeat(1, 4, 1); x[:, 1:, 0] += .1
        prev = torch.full((1, 4, 28), .12)
        batch, bh, _ = m(x, prev); hidden = None; outputs = []
        for t in range(4):
            out, hidden, _ = m(x[:, t:t+1], prev[:, t:t+1], hidden); outputs.append(out)
        self.assertTrue(torch.equal(batch, torch.cat(outputs, dim=1)))
        self.assertTrue(torch.allclose(bh, hidden, atol=1e-6, rtol=1e-6))

    def test_actor_requires_explicit_reset_and_contiguous_history(self):
        actor = p.EpisodeActor(policy(), list(range(28)))
        obs = observation(); obs.update(phase='squeeze', phase_progress=1.)
        with self.assertRaises(RuntimeError): actor.act(obs, episode_id='a', step=0)
        actor.reset('a'); self.assertEqual(len(actor.act(obs, episode_id='a', step=0)), 28)
        with self.assertRaises(RuntimeError): actor.act(obs, episode_id='a', step=2)
        actor.reset('b'); self.assertEqual(len(actor.act(obs, episode_id='b', step=0)), 28)

    def test_no_gradients_change_frozen_baseline(self):
        m, prev, out, d, labels, c = self.loss_fixture(); m.train()
        self.assertFalse(m.baseline.training)
        self.assertTrue(all(not v.requires_grad for v in m.baseline.parameters()))
        self.assertFalse(p.ACTION_CONTRACT['teacher_hidden_state_or_labels_in_actor'])
        self.assertFalse(p.ACTION_CONTRACT['contact_impulse_threshold_is_hardcoded_into_actor'])


if __name__ == '__main__':
    unittest.main()
