import json
from pathlib import Path
import tempfile
import unittest

from natural_face90_hover_d_metrics import HALF, compare, episode, read_episodes


def sample(step, phase='approach', gap=.030, index=0., other=0., drive=False, seen=False, peak=0.):
    aligned = phase == 'approach'
    hold = phase == 'hold'
    anchor = not aligned and not seen
    c_z = -HALF + (min(.0005, -gap+.00075) if aligned else -.030)
    return dict(seed=12, lane=0, step=step, pre_aligned=aligned, pre_orientation_ok=True,
                pre_lateral_error_m=.005 if hold else .002, pre_normal_gap_m=gap, pre_support_ok=True,
                pre_contact=seen, pre_index_contact=seen, pre_other_digit_contact=False, pre_recovering=False,
                pre_centering=False, pre_settling=False, pre_capture_remaining=0, index_lead_m=.015,
                requested_z_m=-HALF-gap if hold else c_z, pre_c_requested_z_m=c_z, pre_measured_pad_z_m=-HALF-gap,
                pre_recenter_depth_cap_m=-HALF-gap, pre_approach_recenter_hold=hold, pre_approach_recenter_entry=hold,
                pre_recenter_any_right_contact=seen, pre_recenter_seen_right_contact=seen, pre_hover_residual_off=anchor,
                pre_hover_measured_anchor=anchor, pre_arm_residual_baseline_max_rad=.001,
                pre_arm_residual_applied_max_rad=0. if anchor else .001,
                pre_arm_residual_change_max_rad=.001 if anchor else 0., qualified_index_frame=bool(index),
                drive_command=drive, safe=True, index_distal_impulse_Ns=index, other_digit_target_impulse_Ns=other,
                right_target_impulse_Ns=index+other, face_deg=peak, peak_drive_response_deg=peak)


def outcome(frames, **changes):
    return dict(dict(seed=12, lane=0, unsafe=False, success=False, turn_started=False, end_reason='post_horizon',
                     distal_contact_frames=sum(bool(r['index_distal_impulse_Ns']) for r in frames),
                     qualified_index_frames=sum(r['qualified_index_frame'] for r in frames),
                     drive_frames=sum(r['drive_command'] for r in frames),
                     peak_drive_response_deg=frames[-1]['peak_drive_response_deg'] if frames else 0.), **changes)


def make_report(folder, frames):
    folder.mkdir()
    report = dict(seeds=[12], rows=[outcome(frames)], expert_learning=dict(learned_lanes=int(bool(frames)),
                                                                          learned_transitions=len(frames)))
    path = folder/'report.json'
    path.write_text(json.dumps(report))
    (folder/'turn_contact_evidence.jsonl').write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    return path


class DReadback(unittest.TestCase):
    def test_recenter_and_first_qualified_approach_contact(self):
        c = [sample(5, 'standoff'), sample(6)]
        d = [sample(5, 'standoff'), sample(6), sample(7, 'hold'), sample(8, gap=.010, index=.01),
             sample(9, seen=True, index=.02, drive=True, peak=2.2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare(make_report(root/'c', c), make_report(root/'d', d), [12])
        panel = result['group']['fixed_C_no_contact_horizon']
        self.assertEqual(panel['seeds'], [12])
        self.assertEqual(panel['D']['safe_qualified_approach_index_first'], 1)
        self.assertEqual(panel['D']['safe_drive_2deg'], 1)
        self.assertEqual(panel['safe_paired_min_approach_gap_mm']['median_D_minus_C'], -20.)
        row = result['per_seed'][0]['D']
        self.assertEqual((row['standoff_frames'], row['hold_frames'], row['anchor_frames']), (1,1,2))

    def test_mixed_early_or_unsafe_contact_is_not_credited_as_clean_approach(self):
        examples = [
            ([sample(1), sample(2, 'hold', index=.03, other=.01)], False, 'recenter', 'mixed'),
            ([sample(1, 'standoff', other=.03)], False, 'standoff', 'other_only'),
            ([sample(1, index=.03)], True, 'approach', 'index_only'),
        ]
        for frames, unsafe, phase, kind in examples:
            with self.subTest(phase=phase):
                row = episode(frames, outcome(frames, unsafe=unsafe), True)
                self.assertEqual((row['first_contact']['phase'], row['first_contact']['kind']), (phase, kind))
                self.assertFalse(row['safe_qualified_approach_index_first'])
        for frames, label in (([], 'no_learned_phase'), ([sample(1)], 'no_measured_contact')):
            self.assertEqual(episode(frames, outcome(frames), True)['first_contact']['phase'], label)

    def test_capped_depth_gate_and_contact_latch(self):
        a, h = sample(1), sample(2, 'hold', gap=.029)
        cont = sample(3, 'hold', gap=.028)
        cont.update(pre_approach_recenter_entry=False, requested_z_m=h['requested_z_m'],
                    pre_recenter_depth_cap_m=h['pre_recenter_depth_cap_m'])
        seq = [a,h,cont]
        self.assertEqual(episode(seq, outcome(seq), True)['hold_frames'], 2)
        for changes in (dict(requested_z_m=cont['requested_z_m']+.0001),
                        dict(pre_lateral_error_m=.0071), dict(pre_recenter_any_right_contact=True),
                        dict(pre_hover_measured_anchor=False), dict(pre_approach_recenter_hold=False)):
            altered = [a, h, dict(cont, **changes)]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                episode(altered, outcome(altered), True)
        with self.assertRaises(ValueError):
            episode([h], outcome([h]), True)

    def test_drive_response_requires_two_isolated_endpoints(self):
        for previous_other in (0., .02):
            prev = sample(1, index=.01, other=previous_other)
            next_row = sample(2, seen=True, drive=True, index=.01, peak=4.)
            if previous_other:
                with self.assertRaisesRegex(ValueError, 'two-endpoint'):
                    episode([prev, next_row], outcome([prev, next_row]), True)
            else:
                self.assertTrue(episode([prev, next_row], outcome([prev, next_row]), True)['safe_drive_2deg'])

    def test_order_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=make_report(Path(tmp)/'d', [sample(1),sample(3)])
            with self.assertRaisesRegex(ValueError, 'Missing or duplicated'):
                read_episodes(path, [12], True)
            with self.assertRaisesRegex(ValueError, 'lane order'):
                read_episodes(path, [99], True)
        f = sample(1, index=.02)
        with self.assertRaisesRegex(ValueError, 'counts'):
            episode([f], outcome([f], distal_contact_frames=0), True)


if __name__ == '__main__':
    unittest.main()
