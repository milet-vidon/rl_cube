"""CPU checks for attribution of physical contact and matched hover measurements."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import natural_face90_hover_ab_metrics as ab


def frame(step, **fields):
    defaults = dict(seed=101, lane=0, step=step, pre_support_ok=1., pre_aligned=0.,
                    pre_contact=0., pre_index_contact=0., pre_other_digit_contact=0.,
                    pre_recovering=0., pre_centering=0., pre_settling=0.,
                    pre_capture_remaining=0., drive_command=0., requested_z_m=-.075,
                    index_lead_m=.004, pad_z_m=-.075, right_target_impulse_Ns=0.,
                    index_distal_impulse_Ns=0., other_digit_target_impulse_Ns=0.)
    return dict(defaults, **fields)


def result(seed=101,lane=0):
    return dict(seed=seed,lane=lane,unsafe=False,end_reason='post_horizon',turn_started=False,
                success=False,distal_contact_frames=0,drive_frames=0,peak_drive_response_deg=0.)


class HoverMeasurement(unittest.TestCase):
    def test_contact_is_assigned_to_command_preceding_native_readback(self):
        rows = [frame(11), frame(12, right_target_impulse_Ns=.02, other_digit_target_impulse_Ns=.02,
                                 pad_z_m=-.052),
                frame(13, pre_contact=1., pre_other_digit_contact=1., pre_recovering=1.),
                frame(14, index_distal_impulse_Ns=.01, right_target_impulse_Ns=.01)]
        ep, samples = ab.episode(rows,result(),False)
        self.assertEqual(list(samples), [11,12])
        self.assertEqual(ep['first_right_digit_target_contact']['status'], 'early_hover_contact')
        self.assertEqual(ep['first_index_target_contact']['status'], 'contact_outside_hover')
        self.assertAlmostEqual(ep['tracking']['peak_inward_mm'],23.)
        self.assertFalse(ab.hover_command(frame(17, pre_aligned=1.),False))
        self.assertFalse(ab.hover_command(frame(17, pre_settling=1.),False))
        self.assertFalse(ab.hover_command(frame(17, requested_z_m=-.100),False))
        self.assertTrue(ab.hover_command(frame(17, requested_z_m=-.079, index_lead_m=-.020),False))

    def test_two_second_window_is_paired_to_baseline_and_requires_all_sixty_frames(self):
        pivot=dict(status='early_hover_contact',step=141)
        a={k:(-75.,-60.+(k-82)) for k in range(80,142)}
        b={k:(-75.,-75.) for k in range(82,200)}
        w=ab.precontact_window(pivot,a,b)
        self.assertEqual((w['complete'],w['start_step'],w['end_step']), (True,82,141))
        self.assertEqual(w['delta_pad_peak_to_peak_mm'],-59.)
        del b[108]
        self.assertFalse(ab.precontact_window(pivot,a,b)['complete'])
        self.assertEqual(ab.precontact_window(pivot,a,b)['B_eligible_frames'],59)

    def test_missing_phase_and_treatment_fidelity(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for arm in ('A','B'):
                p=Path(tmp)/arm
                p.mkdir()
                rows=[frame(10),frame(11, pre_aligned=1., right_target_impulse_Ns=.02)]
                if arm=='B':
                    for i,r in enumerate(rows):
                        r.update(pre_hover_residual_off=int(i==0),pre_arm_residual_baseline_max_rad=.005,
                                 pre_arm_residual_applied_max_rad=.005 if i else 0.,
                                 pre_arm_residual_change_max_rad=0. if i else .005)
                (p/'turn_contact_evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                report=dict(seeds=[101,202], rows=[result(),result(202,1)],
                            expert_learning=dict(learned_lanes=1,learned_transitions=2))
                f=p/'report.json'; f.write_text(json.dumps(report)); paths.append(f)
            c=ab.compare(*paths,[101,202])
            self.assertEqual(c['group']['A']['right_digit_first_contact_status']['contact_outside_hover'],1)
            self.assertEqual(c['group']['A']['index_first_contact_status']['did_not_reach_learned_phase'],1)
            self.assertEqual(c['group']['A']['entered_hover'],1)
            self.assertEqual(c['group']['B']['unsafe'],0)
            self.assertIsNone(c['group']['paired_precontact_2s']['median_paired_delta_pad_peak_to_peak_mm'])
            rows[1]['pre_arm_residual_applied_max_rad']=0.
            (paths[1].parent/'turn_contact_evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'intervention mismatch'):
                ab.compare(*paths,[101,202])


if __name__ == '__main__':
    unittest.main()
