import unittest
from types import SimpleNamespace
from unittest.mock import patch
from matched_cube_material import pair_coefficients, matched_cube_material

class MatchedMaterialTests(unittest.TestCase):
    def test_external_pair_coefficients_match_baseline_average(self):
        for internal in ((0,0),(.4,.3),(.7,.65)):
            values=pair_coefficients(internal)
            for side,other in (('left',(.5,.5)),('table',(.8,.7))):
                for i,cube in enumerate((.9,.8)):
                    self.assertAlmostEqual(max(values[side][i],internal[i]),(cube+other[i])/2.)
    def test_unmatchable_and_invalid_coefficients_rejected(self):
        for internal in ((.8,.5),(.7,.7),(-1,0),(.2,.3),(float('nan'),0)):
            with self.assertRaises(ValueError): pair_coefficients(internal)
    def test_wrapper_restored_on_failed_material_check(self):
        original=lambda *a: SimpleNamespace(model={})
        module=SimpleNamespace(DynamicCubeHandle=original)
        with patch('matched_cube_material.apply_matched_materials', side_effect=ValueError('mismatch')):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                with matched_cube_material((.4,.3),module): module.DynamicCubeHandle(None)
        self.assertIs(module.DynamicCubeHandle,original)
    def test_disabled_noop(self):
        module=SimpleNamespace(DynamicCubeHandle=object()); original=module.DynamicCubeHandle
        with matched_cube_material(None,module): self.assertIs(module.DynamicCubeHandle,original)

if __name__=='__main__':unittest.main()
