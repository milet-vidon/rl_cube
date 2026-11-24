from pathlib import Path
import unittest
from unittest.mock import patch

import natural_face90_pad_preshape_probe as probe


class PreshapeCampaignTest(unittest.TestCase):
    def test_verification_joins_both_physical_receipts(self):
        path = Path('probe/attempt_abc12345/report.json')
        req = dict(pad_orientation_plan=dict(scene_sha256='scene'))
        orientation = dict(feasible_by_lane=[False])
        learning = dict(pad_orientation=orientation, pad_preshape=dict(orientation=orientation))
        with (patch.object(probe.points, 'verify_report', return_value=([], learning)),
              patch.object(probe.v7, 'read', return_value=dict(domain_receipt=dict(scene_sha256='scene'))),
              patch.object(probe.policy, 'verify_receipt') as verify):
            self.assertEqual(probe.verify_report(path, req), ([], learning))
            verify.assert_called_once_with(learning['pad_preshape'], path.parent / 'turn_contact_evidence.jsonl', req)
            learning['pad_preshape']['orientation'] = {}
            with self.assertRaisesRegex(ValueError, 'receipts differ'):
                probe.verify_report(path, req)

    def test_fallback_or_unsettled_shape_cannot_gain_preshape_credit(self):
        seeds = probe.anchor.metrics.FIXED_SEEDS[:4]
        rows = [dict(seed=seed, lane=i, safe_qualified_index_first=True, safe_drive_2deg=True)
                for i, seed in enumerate(seeds)]
        receipt = dict(ready_step_by_lane=[10, 10, None, 10], handoff_error_rad_by_lane=[.001, .016, .001, .001],
                       orientation=dict(feasible_by_lane=[True, True, True, False]))
        baseline = dict(unsafe_seeds=[])
        result = probe.paired_result(dict(per_seed=rows, unsafe_seeds=[]), baseline, baseline, receipt)
        self.assertEqual(result['qualifying_fixed_seeds'], [seeds[0]])
        self.assertFalse(result['paired_contact_gate_met'])
        self.assertTrue(result['rollout_paired_contact_gate_met'])

    def test_requires_coverage_of_each_archived_handoff(self):
        fixed = probe.anchor.metrics.FIXED_SEEDS
        seeds = list(fixed) + [66002209, 66002214]
        rows = [dict(seed=s, learned_frames=3) for s in seeds]
        previous = dict(per_seed=rows)
        with (patch.object(probe.geometry_check, 'handoffs', return_value=(rows, list(fixed))),
              patch.object(probe.geometry_check, 'ShapeCheck') as cls):
            test_pose = dict(measured=dict(tilt_deg=None), **{probe.VARIANT: dict(tilt_deg=9)})
            checker = cls.return_value
            checker.check.return_value = test_pose
            screen = probe.handoff_geometry(Path('snapshot'), {}, previous)
            self.assertEqual(screen['counts']['measured'], dict(all=0, fixed=0))
            self.assertEqual(screen['counts']['index_-6'], dict(all=13, fixed=11))
            self.assertEqual(screen['per_seed'][str(seeds[0])]['index_-6']['tilt_deg'], 9)
            # Even a failure outside the fixed contact cohort invalidates the frozen screen.
            fail = dict(test_pose, **{probe.VARIANT: dict(tilt_deg=None)})
            checker.check.side_effect = [test_pose] * 12 + [fail]
            with self.assertRaisesRegex(ValueError, 'missing a feasible'):
                probe.handoff_geometry(Path('snapshot'), {}, previous)

    def test_dispatch_and_sign_of_extension(self):
        pins = probe.source_pins()
        with patch.object(probe.wrist, 'validate_probe') as validate:
            parent = dict(schema=probe.wrist.SCHEMA)
            plan = dict(schema=probe.SCHEMA, source_sha256=pins, orientation_plan=parent,
                        index_flexion_offset_deg=-6, handoff_geometry=dict(counts={'index_-6': dict(all=13, fixed=11)}))
            req = dict(pad_preshape_offset_deg=-6, pad_orientation_plan=parent, pad_preshape_plan=plan)
            probe.validate_probe(req)
            validate.assert_called_once_with(req)
            with self.assertRaises(ValueError):
                probe.validate_probe(dict(req, pad_preshape_offset_deg=6))
        self.assertIs(probe._run.__globals__['PostPhase'], probe.policy.PadPreshapePhase)
        self.assertIs(probe._worker.__globals__['validate_request'], probe.validate_probe)
        self.assertIs(probe._wave.__globals__['verify_report'], probe.verify_report)
        self.assertEqual(probe._wave.__globals__['SCRIPT'], probe.SCRIPT)


if __name__ == '__main__':
    unittest.main()
