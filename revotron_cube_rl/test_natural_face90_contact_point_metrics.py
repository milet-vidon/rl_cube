import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from natural_face90_contact_point_metrics import compare, FIXED_SEEDS, R70_SHA, SEEDS
from test_natural_face90_hover_d_metrics import sample, outcome


def fixture(folder, offset, hits=(), unsafe=(66002209, 66002214), other_first=()):
    folder.mkdir()
    turns, points, rows, selected = [], [], [], []
    rad = math.radians(offset)
    direction = [math.cos(rad), -math.sin(rad)]
    counts, contacts = {}, {}
    for lane, seed in enumerate(SEEDS):
        if seed in hits:
            frames = [sample(0, 'standoff', other=.01 if seed in other_first else 0.),
                      sample(1, index=.01), sample(2, index=.01, seen=True),
                      sample(3, index=.01, seen=True), sample(4, index=.01, seen=True, drive=True, peak=2.25)]
        elif seed in FIXED_SEEDS:
            frames = [sample(0, 'standoff'), sample(1)]
        elif seed in (66002209, 66002214):
            frames = [sample(0, 'standoff', other=.01)]
        else:
            frames = []
        previous_face = 0.
        for r in frames:
            pre = math.radians(offset + previous_face)
            r.update(seed=seed, lane=lane, pre_material_face_deg=previous_face,
                     pre_material_marker_x_m=.030 * math.cos(pre), pre_material_marker_y_m=-.030 * math.sin(pre))
            if r['index_distal_impulse_Ns']:
                phi = math.radians(offset + r['face_deg'])
                native_marker = [.030 * math.cos(phi), -.030 * math.sin(phi), -.045]
                points.append(dict(step=r['step'], substep=2, seed=seed, lane=lane, cubelet='nZ_pX', impulse_Ns=.01,
                                   point_cubelet_m=[0., 0., -.015], point_core_m=native_marker, face_deg=r['face_deg'],
                                   marker_core_m=native_marker, contact_marker_lateral_m=0.))
            previous_face = r['face_deg']
        row = outcome(frames, seed=seed, lane=lane, unsafe=seed in unsafe, target_deg=15,
                      final_face_deg=previous_face + .001, post_samples=len(frames))
        rows.append(row)
        counts[seed] = len(frames)
        contacts[seed] = sum(bool(r['index_distal_impulse_Ns']) for r in frames)
        selected.append(bool(frames))
        turns.extend(frames)
    def save_rows(name, records):
        data = ''.join(json.dumps(record) + '\n' for record in records).encode()
        (folder/name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    turns_sha = save_rows('turn_contact_evidence.jsonl', turns)
    learning = dict(input_checkpoint_sha=R70_SHA, actor_updated=False, deterministic=True,
                    turn_evidence_sha256=turns_sha, learned_transitions=len(turns), learned_lanes=sum(selected))
    if offset:
        learning['contact_point'] = dict(offset_deg=offset, radius_m=.030, selected_by_lane=selected,
            base_radial_xy_by_lane=[[1.,0.] if used else [0.,0.] for used in selected],
            selected_radial_xy_by_lane=[direction if used else [0.,0.] for used in selected])
        native_file = 'native_contact_geometry.jsonl'
        learning['geometry_probe'] = dict(schema='native_dip_contact_geometry_v1', file=native_file,
            sha256=save_rows(native_file, points), selected_seeds=SEEDS, active_frames_by_seed=counts,
            frames_by_seed=contacts, points=len(points))
    path = folder/'report.json'
    path.write_text(json.dumps(dict(role='validate', seeds=SEEDS, rows=rows, expert_learning=learning)))
    return path


class ContactPointComparison(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.c = fixture(self.root/'c', 0)

    def test_paired_drive_native_point_and_tie(self):
        plus = fixture(self.root/'plus', 7, FIXED_SEEDS[:2])
        minus = fixture(self.root/'minus', -7, FIXED_SEEDS[-2:])
        result = compare(self.c, plus, minus, SEEDS)
        self.assertEqual(result['tied_offsets_deg'], [7, -7])
        self.assertEqual(result['selected_offset_deg'], 7)
        angle = result['angles']['7']
        self.assertEqual(angle['unsafe_seeds'], [66002209, 66002214])
        self.assertEqual(angle['qualifying_fixed_seeds'], list(FIXED_SEEDS[:2]))
        first = angle['per_seed'][0]
        self.assertEqual(first['first_measured_distal_native_contact']['cubelet'], 'nZ_pX')
        self.assertTrue(first['safe_qualified_drive'])
        self.assertAlmostEqual(first['final_face_deg'], 2.251)
        self.assertEqual(first['peak_drive_response_deg'], 2.25)
        self.assertIsNone(angle['per_seed'][2]['final_face_deg'])

    def test_unsafe_seed_substitution_rejects_candidate_even_with_equal_count(self):
        newly_unsafe = 66002210
        plus = fixture(self.root/'plus', 7, FIXED_SEEDS[:3], (66002209, newly_unsafe))
        minus = fixture(self.root/'minus', -7, FIXED_SEEDS[:1])
        result = compare(self.c, plus, minus, SEEDS)
        self.assertIsNone(result['selected_offset_deg'])
        self.assertEqual(result['angles']['7']['new_unsafe_seeds'], [newly_unsafe])
        self.assertFalse(result['angles']['-7']['meets_training_gate'])

    def test_geometry_and_trace_must_match_original_inputs(self):
        plus = fixture(self.root/'plus', 7)
        minus = fixture(self.root/'minus', -7)
        with self.assertRaisesRegex(ValueError, 'offset'):
            compare(self.c, minus, plus, SEEDS)
        report = json.loads(plus.read_bytes())
        report['expert_learning']['contact_point']['selected_radial_xy_by_lane'][0] = [1.,0.]
        plus.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            compare(self.c, plus, minus, SEEDS)
        (self.c.parent/'turn_contact_evidence.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            compare(self.c, plus, minus, SEEDS)

    def test_first_clean_contact_and_drive_must_coincide_in_the_same_safe_seed(self):
        plus = fixture(self.root/'plus', 7, FIXED_SEEDS[:2], other_first=FIXED_SEEDS[:1])
        minus = fixture(self.root/'minus', -7, FIXED_SEEDS[:2])
        result = compare(self.c, plus, minus, SEEDS)
        angle = result['angles']['7']
        self.assertEqual(angle['fixed_C_no_contact_horizon']['safe_drive_2deg'], 2)
        self.assertEqual(angle['qualifying_fixed_seeds'], list(FIXED_SEEDS[1:2]))
        self.assertFalse(angle['meets_training_gate'])
        self.assertEqual(result['selected_offset_deg'], -7)


if __name__ == '__main__':
    unittest.main()
