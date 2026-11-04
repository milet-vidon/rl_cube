import copy
import math
import unittest

from left_index_continuation import ContinuedIndexSupport as Original, SETTINGS as ORIGINAL_SETTINGS
from left_index_continuation_v20 import ContinuedIndexSupport, initial_state, index_envelope_values, SETTINGS, SCHEMA
from left_index_mpr_envelope_v20 import factored_values


def fixture():
    old = -.0004
    original = {'settings': dict(ORIGINAL_SETTINGS), 'reference_target_rad': .006787104745954275,
                'last_applied_offset_rad': old, 'next_offset_rad': old,
                'last_nominal_target_rad': .006787104745954275 + old,
                'last_applied_target_rad': .006787104745954275 + old,
                'reference_basis': 'last nominal center target minus its recorded applied offset'}
    new = dict(original, settings=dict(SETTINGS), source_center_settings=dict(ORIGINAL_SETTINGS), schema=SCHEMA)
    return original, new


class ContinuationTest(unittest.TestCase):
    def test_only_lower_bound_changed(self):
        self.assertEqual(SETTINGS, dict(ORIGINAL_SETTINGS, lower_offset_rad=-.023))

    def test_identical_before_old_saturation_and_bounded_after(self):
        old, new = fixture(); a, b = Original(old), ContinuedIndexSupport(new)
        for _ in range(200):
            self.assertEqual(a.target, b.target)
            self.assertEqual(a.update(0., False), b.update(0., False))
        for _ in range(1000):
            previous = b.offset; b.update(0., False)
            self.assertLessEqual(abs(b.offset - previous), .00005 + 1e-12)
            self.assertGreaterEqual(b.offset, -.023)
        self.assertEqual(b.offset, -.023)

    def test_existing_impulse_hold_and_release(self):
        _, new = fixture(); a = ContinuedIndexSupport(new)
        a.offset = -.02
        a.update(.001, True); self.assertEqual(a.offset, -.02)
        a.update(.003, True); self.assertAlmostEqual(a.offset, -.01995)
        a.update(.001, False); self.assertAlmostEqual(a.offset, -.02)

    def test_exact_center_handover_no_offset_doubling(self):
        old, _ = fixture()
        name = ORIGINAL_SETTINGS['joint']; target = old['last_nominal_target_rad']
        center = {'finger_contact_feedback': ORIGINAL_SETTINGS}
        row = {'nominal_target': [target], 'index_contact_feedback': {'offset_rad': -.0004, 'next_offset_rad': -.0004}}
        new = initial_state(center, row, [name], [target])
        self.assertEqual(ContinuedIndexSupport(new).target, target)
        self.assertEqual(new['source_center_settings'], ORIGINAL_SETTINGS)

    def test_reject_changed_center_or_fabricated_handover(self):
        _, new = fixture()
        for field, value in [('source_center_settings', SETTINGS), ('last_applied_offset_rad', -.02),
                             ('next_offset_rad', -.01), ('reference_target_rad', math.nan),
                             ('last_applied_target_rad', .2)]:
            bad = copy.deepcopy(new); bad[field] = value
            with self.assertRaises(ValueError): ContinuedIndexSupport(bad)

    def test_finite_and_typed_measurements(self):
        _, new = fixture(); a = ContinuedIndexSupport(new)
        for impulse, on in [(math.nan, True), (math.inf, True), (-1., True), (0., 1)]:
            with self.assertRaises(ValueError): a.update(impulse, on)

    def test_envelope_includes_actual_target_and_extremes(self):
        _, new = fixture(); actual = .006636961828917265
        values = index_envelope_values(actual, new)
        for x in [actual, new['reference_target_rad'], new['reference_target_rad'] - .023,
                  new['reference_target_rad'] - .012, new['last_applied_target_rad']]:
            self.assertIn(x, values)
        self.assertLessEqual(max(b-a for a,b in zip(values,values[1:])), .001 + 1e-12)

    def test_factorization_covers_all_pairs(self):
        indices, rings = [-.02, -.01, 0.], [0., .002, .004]
        dynamic, static = factored_values(indices, rings, 0., 0.)
        self.assertEqual(set(static), {(i,r) for i in indices for r in rings})
        self.assertEqual({i for i,r in dynamic}, set(indices))
        self.assertEqual({r for i,r in dynamic}, set(rings))
        with self.assertRaises(ValueError): factored_values(indices, rings, .1, 0.)

    def test_factorization_equals_full_boolean_pairwise_contract(self):
        # Exhaustive synthetic positions exercise failures in right/index,
        # right/ring, index/ring, and separate rotating-layer predicates.
        indices, rings, path = [-.02, -.01, 0.], [0., .002, .004], [0., .01, .03]
        dynamic, static = factored_values(indices, rings, 0., 0.)
        for threshold in [-.1, 0., .001, .003, .01, .05, .1]:
            def independent(i, r, p):
                return abs(i-p) >= threshold and abs(r-p) >= threshold and abs(i+r) >= threshold and abs(i+p*.3) >= threshold and abs(r-p*.3) >= threshold
            full = all(independent(i,r,p) for i in indices for r in rings for p in path)
            factored = all(independent(i,r,p) for i,r in dynamic for p in path) and all(independent(i,r,path[0]) for i,r in static)
            self.assertEqual(full, factored)


if __name__ == '__main__': unittest.main()
