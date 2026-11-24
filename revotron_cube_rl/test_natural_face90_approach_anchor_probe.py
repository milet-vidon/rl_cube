import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import natural_face90_approach_anchor_probe as probe
import test_natural_face90_turn_guide as fixtures


class ApproachAnchorTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.ContactTurn()
        f.setUp()
        self.f = f
        self.guide = probe.ApproachGuide(f.c)
        g = self.guide
        self.zero = ~f.yes
        g.aligned = f.yes.clone()
        g.goal_local[:, 2] = -.07425
        g._evidence = dict(pre_contact=self.zero, pre_index_contact=self.zero, pre_other_digit_contact=self.zero,
                           pre_orientation_ok=f.yes, pre_support_ok=f.yes, pre_recovering=self.zero,
                           pre_centering=self.zero, pre_settling=self.zero, pre_aligned=g.aligned,
                           pre_normal_gap_m=f.q.new_full((2,), .030))
        self.obs = dict(wrong_contact=self.zero, contact_goal_reached=self.zero,
                        right_total_impulse=f.q.new_zeros(2))

    def tick(self, fresh=None):
        g = self.guide
        residual = torch.full_like(g.reference, .001)
        applied = g.filter_residual(residual, self.zero if fresh is None else fresh, self.f.yes, self.obs)
        reference = g.reference.clone()
        measured = reference + .005
        selected = g.select_anchor(measured, reference + .003)
        return selected, measured, reference, applied, residual

    def test_right_arm_anchor_and_residual_across_subthreshold_contact(self):
        self.obs['right_total_impulse'][1] = 1e-9
        chosen, measured, reference, applied, residual = self.tick(self.f.yes)
        torch.testing.assert_close(chosen[0], measured[0])
        torch.testing.assert_close(chosen[1], reference[1])
        torch.testing.assert_close(applied, residual)
        self.assertEqual(self.guide.approach_count.tolist(), [1, 0])
        self.obs['right_total_impulse'].zero_()
        self.assertEqual(self.tick()[0].tolist(), chosen.tolist())
        self.guide._evidence['pre_orientation_ok'] = self.zero
        chosen, _, reference, _, _ = self.tick()
        torch.testing.assert_close(chosen, reference)

    def test_hover_preserves_existing_residual_knockout_and_anchor(self):
        self.guide.aligned = self.zero
        self.guide._evidence['pre_aligned'] = self.zero
        chosen, measured, _, applied, _ = self.tick()
        torch.testing.assert_close(chosen, measured)
        torch.testing.assert_close(applied, torch.zeros_like(applied))
        self.assertEqual(self.guide.approach_count.tolist(), [0, 0])
        self.assertEqual(self.guide.anchor_frames.tolist(), [1, 1])

    def test_real_command_matches_plus7_with_only_eligible_reference_replaced(self):
        f = self.f
        f.c.radial = probe.material.material_radial(f.radial, 7)
        trial, oracle = probe.ApproachGuide(f.c), probe.material.MaterialGuide(f.c)
        tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        local = torch.cat((.030 * f.c.radial, f.q.new_full((2, 1), -.074)), dim=1)
        core = torch.cat((tip - local, f.q.new_tensor([[0., 0., 0., 1.]] * 2)), dim=1)
        _, palm, _, _ = f.c.rp.fk(f.q[:, trial.arm])
        axis = palm.transpose(1, 2)[:, :, 2]
        for guide in (trial, oracle):
            guide.axis.copy_(axis)
            guide.vectors.copy_(.15 * axis[:, None, :])
            guide.orientation.copy_(palm)
            guide.reference.copy_(f.q[:, guide.arm] + .006)
        oracle.reference[0] = f.q[0, trial.arm]
        obs = dict(f.obs, natural_ready=f.yes, right_contact=self.zero, index_contact=self.zero,
                   other_digit_contact=self.zero, right_total_impulse=f.q.new_tensor([0., 1e-9]))
        args = (f.q, f.q, f.q, self.zero, f.yes, torch.ones_like(trial.reference), core, tip, local, obs)
        actual, expected = trial.command(*args), oracle.command(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(trial.control_evidence()['pre_approach_anchor'].tolist(), [True, False])
        torch.testing.assert_close(trial.control_evidence()['pre_arm_residual_applied_max_rad'],
                                   oracle.control_evidence()['pre_arm_residual_applied_max_rad'])

    def test_advancing_z_request_and_support_required(self):
        self.guide._evidence['pre_support_ok'] = torch.tensor([True, False])
        self.guide.goal_local[0, 2] = -.080
        chosen, _, ref, _, _ = self.tick()
        torch.testing.assert_close(chosen, ref)

    def test_signed_same_interval_comparison_and_standoff_transition(self):
        base = dict(seed=7, pre_contact=False, pre_index_contact=False, pre_other_digit_contact=False,
                    pre_aligned=True, pre_orientation_ok=True, pre_support_ok=True, pre_recovering=False,
                    drive_command=False, pre_normal_gap_m=.030, requested_z_m=-.07425, pad_z_m=-.07505,
                    safe=True, pre_hover_measured_anchor=False, right_target_impulse_Ns=0.,
                    index_distal_impulse_Ns=0., other_digit_target_impulse_Ns=0.)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'frames.jsonl'
            trace.write_text('\n'.join(json.dumps(r) for r in [base, dict(base, pre_aligned=False,
                                                                          pre_hover_measured_anchor=True)]))
            frames, _ = probe.phase_motion(trace, [7], False)
            stats = probe.approach_stats(frames, [7])
        self.assertAlmostEqual(stats['requested_inward_median_mm'], .75)
        self.assertAlmostEqual(stats['measured_inward_median_mm'], -.05)
        self.assertEqual(stats['returns_to_standoff'], 1)

    def test_frozen_wave_contract_and_dispatch(self):
        req = dict(paired_validation=True, role='validate', learn=False, iteration=70,
                   seeds=probe.metrics.SEEDS, target_deg=15, contact_point_offset_deg=7,
                   checkpoint_sha=probe.metrics.R70_SHA, approach_anchor_source_sha256=probe.probe_pins(),
                   approach_anchor_schema=probe.SCHEMA)
        with patch.object(probe.points, 'validate_request'):
            probe.validate_probe(req)
            for changed in (dict(learn=True), dict(role='train'), dict(contact_point_offset_deg=-7),
                            dict(approach_anchor_source_sha256={}), dict(seeds=[1, 2])):
                with self.assertRaises(ValueError):
                    probe.validate_probe(dict(req, **changed))
        self.assertIs(probe._run.__globals__['PostPhase'], probe.ProbePhase)
        self.assertIs(probe._worker.__globals__['run'], probe._run)
        self.assertEqual(probe._wave.__globals__['SCRIPT'], probe.SCRIPT)
