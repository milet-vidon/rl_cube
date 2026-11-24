"""CPU regression tests for joining post-step geometry and terminal evidence."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from natural_face90_first_touch_audit import audit


class FirstTouchAuditTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        def save(name, value, *, lines=False):
            path = root / name
            data = (''.join(json.dumps(v) + '\n' for v in value) if lines else json.dumps(value)).encode()
            path.write_bytes(data)
            return hashlib.sha256(data).hexdigest()

        contact = dict(lane=0, seed=101, step=7)
        turn = dict(contact, face_deg=90., pad_x_m=0., pad_y_m=-.030, pad_z_m=-.052, safe=True,
                    pad_near_marker=False, qualified_index_frame=False, index_distal_impulse_Ns=.0002,
                    static_support_impulse_Ns=.005, other_digit_target_impulse_Ns=0.)
        post = dict(contact, safe=True, face_deg=90., natural_ready=True, structure_ok=True,
                    allowed_layer_state=True, wrong_contact=False, retract=False)
        point = dict(contact, substep=1, impulse_Ns=.0001, contact_marker_lateral_m=.019,
                     anchor_contact_distance_m=.025, cubelet='nZ')
        turn_hash = save('turn_contact_evidence.jsonl', [turn], lines=True)
        post_hash = save('post_measurements.jsonl', [post, dict(post, step=10, structure_ok=False)], lines=True)
        points = [dict(point, substep=2, impulse_Ns=.2), dict(point, impulse_Ns=.00001), point]
        native_hash = save('native_contact_geometry.jsonl', points, lines=True)
        report = dict(role='validate', seeds=[101, 102],
                      rows=[dict(seed=101, unsafe=True), dict(seed=102, unsafe=False)],
                      post_evidence_sha256=post_hash,
                      expert_learning=dict(turn_evidence_sha256=turn_hash,
                                           geometry_probe=dict(file='native_contact_geometry.jsonl', sha256=native_hash),
                                           contact_point=dict(offset_deg=7, radius_m=.03, selected_radial_xy_by_lane=[[1, 0], [0, 1]])))
        report_hash = save('report.json', report)
        sweep_hash = save('contact_point_summary.json', dict(fixed_C_no_contact_horizon_seeds=[101, 102]))
        episodes = [dict(seed=101, first_contact=dict(step=7, kind='index_only', qualified=False),
                         unsafe=True, end_reason='face_structure_violation', safe_qualified_index_first=False),
                    dict(seed=102, first_contact=dict(step=None, kind='none', qualified=False),
                         unsafe=False, end_reason='post_horizon', safe_qualified_index_first=False)]
        save('approach_anchor_summary.json', dict(schema='r70_plus7_approach_anchor_v1', updated=False,
             candidate_report=str(root / 'report.json'), candidate_report_sha256=report_hash,
             plan=dict(seeds=[101, 102], sweep_dir=str(root), sweep_summary_sha256=sweep_hash),
             candidate=dict(per_seed=episodes), new_unsafe_seeds=[101]))
        self.path = root / 'approach_anchor_summary.json'

    def test_rotating_marker_native_substeps_and_unsafe_delay(self):
        result = audit(self.path, include_native=True)
        first, skipped = result['fixed']
        self.assertAlmostEqual(first['pad_lateral_mm'], 0.)
        self.assertAlmostEqual(first['pad_gap_mm'], 7.)
        self.assertEqual(first['observed_failed_gates'], ['pad_axial'])
        self.assertAlmostEqual(first['native_lateral_mm'], 19.)
        self.assertAlmostEqual(first['native_point']['impulse_Ns'], .0001)
        self.assertEqual(first['steps_to_unsafe_end'], 3)
        self.assertEqual(result['all_unsafe_seeds'], [101])
        self.assertNotIn('frame_qualified', skipped)

    def test_rejects_changed_report_or_trace(self):
        for name in ('report.json', 'turn_contact_evidence.jsonl', 'post_measurements.jsonl',
                     'native_contact_geometry.jsonl', 'contact_point_summary.json'):
            with self.subTest(name=name):
                path = self.path.with_name(name)
                original = path.read_bytes()
                self.addCleanup(path.write_bytes, original)
                path.write_bytes(b' ' + original)
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    audit(self.path)
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
