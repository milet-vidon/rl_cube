import math
import unittest

from contact_hold_simple_rotation import build_evidence, summarize_rotation
from test_contact_hold_physical_metrics import CFG, row


def episode(deg, n):
    rows = []
    for i in range(n):
        r = row(i)
        a = math.radians(deg(i))/2
        sign = -1 if i%2 else 1
        r['cube_quaternion_wxyz'] = [sign*math.cos(a),0,sign*math.sin(a),0]
        rows.append(r)
    return rows


class RotationTest(unittest.TestCase):
    def test_timing_after_turn_and_duration_and_release(self):
        rows = episode(lambda i: 20 if i >= 15 else 0, 31)
        # 29 / 30 frames suffice for neither baseline hold nor post-turn retention.
        self.assertFalse(summarize_rotation(rows[:29], CFG)['ever_hold'])
        self.assertTrue(summarize_rotation(rows[:30], CFG)['ever_hold'])
        self.assertFalse(summarize_rotation(rows[:30], CFG)['ever_hold_and_turn'])
        p = summarize_rotation(rows, CFG)
        self.assertTrue(p['ending_hold_and_turn'])
        self.assertEqual(p['first_success'],dict(reference_step=0, threshold_step=15, confirmed_step=30))
        self.assertAlmostEqual(p['max_controlled_turn_deg'],20)
        rows += episode(lambda _:90,1)
        rows[-1].update(step=31,cube_table_contact=True)
        q = summarize_rotation(rows,CFG)
        self.assertTrue(q['ever_hold_and_turn'])
        self.assertFalse(q['ending_hold_and_turn'])
        self.assertTrue(q['final_table_contact'])

    def test_slip_and_uncontrolled_table_rotation(self):
        rows = episode(lambda i:20 if i>=20 else 0, 40)
        for i,r in enumerate(rows):
            r['cube_position'][0]+=.0006*i
        out = summarize_rotation(rows, CFG)
        self.assertEqual(out['best_supported']['frames'],40)
        self.assertEqual(out['best_controlled_hold']['frames'],17)
        self.assertFalse(out['ever_hold'])
        for r in rows[18:]:
            r.update(cube_table_contact=True)
        self.assertLess(summarize_rotation(rows,CFG)['max_controlled_turn_deg'], 1e-6)

    def test_jitter_and_subthreshold_oscillation_not_accumulated(self):
        for fn in (lambda i: 45 if i==4 else 0, lambda i: 13 if i%2 else 0):
            q=summarize_rotation(episode(fn,60), CFG)
            self.assertTrue(q['ever_hold'])
            self.assertFalse(q['ever_hold_and_turn'])
            self.assertLess(q['max_controlled_turn_deg'],15)

    def test_nonfinite_gaps_and_missing_flags(self):
        for field,val in [('step',100),('cube_position',[1,1,float('inf')]),('height_m',float('nan')),
                          ('cube_quaternion_wxyz',[0,0,0,0]),('hard_limit_hit',[False])]:
            rows=episode(lambda i:0,4)
            rows[1][field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):
                build_evidence(rows,CFG)

    def test_custom_geometry_does_not_bypass_drop_or_support(self):
        rows=episode(lambda i:0,4)
        for r in rows:r['layer_measurement']['assembled_for_grasp']=False
        rows[0]['drop']=True
        rows[1]['contact_impulse']=[0.,1.]
        rows[2]['hard_limit_hit'][55]=True
        self.assertEqual(build_evidence(rows,CFG)['support'],[False]*4)
        self.assertEqual(build_evidence(rows,CFG,assembly_check=lambda r: True)['start_by_end'],[1,2,3,3])


if __name__=='__main__':unittest.main()
