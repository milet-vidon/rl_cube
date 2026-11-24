from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import natural_face90_pad_orientation_probe as probe


class OrientationCampaignTest(unittest.TestCase):
    def test_paired_gate_uses_original_safety_and_same_seed_contacts(self):
        a, b, c = probe.anchor.metrics.FIXED_SEEDS[:3]
        original = dict(unsafe_seeds=[])
        previous = dict(unsafe_seeds=[c])
        rows = [dict(seed=s, safe_qualified_index_first=touch, safe_drive_2deg=drive)
                for s, touch, drive in [(a, True, True), (b, True, False), (c, False, True)]]
        candidate = dict(per_seed=rows, unsafe_seeds=[])
        self.assertEqual(probe.paired_result(candidate, original, previous)['qualifying_fixed_seeds'], [a])
        rows[1]['safe_drive_2deg'] = True
        self.assertTrue(probe.paired_result(candidate, original, previous)['paired_contact_gate_met'])
        candidate['unsafe_seeds'] = [c]
        result = probe.paired_result(candidate, original, previous)
        self.assertFalse(result['paired_contact_gate_met'])
        self.assertEqual(result['new_unsafe_seeds'], [c])
        self.assertEqual(result['new_unsafe_vs_previous'], [])

    def test_request_pins_and_validation_dispatch(self):
        with TemporaryDirectory() as directory, patch.object(probe.anchor, 'validate_probe') as validate:
            inputs = Path(directory) / 'input'
            inputs.write_text('{}')
            pins = probe.source_pins()
            plan = dict(schema=probe.SCHEMA, source_sha256=pins, geometry_sha256=probe.geometry_hash({}),
                        input_summary=str(inputs), input_summary_sha256=probe.v7.digest(inputs),
                        scene_file=str(inputs), scene_sha256=probe.v7.digest(inputs))
            req = dict(pad_orientation_geometry={}, pad_orientation_plan=plan)
            probe.validate_probe(req)
            validate.assert_called_once_with(req)
            for changed in (dict(geometry_sha256=''), dict(source_sha256={}), dict(scene_sha256=''),
                            dict(input_summary_sha256='')):
                with self.assertRaises(ValueError):
                    probe.validate_probe(dict(req, pad_orientation_plan=dict(plan, **changed)))
        self.assertIs(probe._worker.__globals__['run'], probe._run)
        self.assertIs(probe._worker.__globals__['validate_request'], probe.validate_probe)
        self.assertIs(probe._run.__globals__['PostPhase'], probe.policy.PadOrientationPhase)
        self.assertEqual(probe._wave.__globals__['SCRIPT'], probe.SCRIPT)
        self.assertIs(probe._wave.__globals__['verify_report'], probe.verify_report)


if __name__ == '__main__':
    unittest.main()
