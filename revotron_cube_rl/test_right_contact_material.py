import unittest
from right_contact_material import validate_friction,is_right_hand_path
class RightMaterialTests(unittest.TestCase):
    def test_valid_coefficients(self):self.assertEqual(validate_friction((1.2,1.)),(1.2,1.))
    def test_invalid_coefficients(self):
        for v in ((float('nan'),1.),(1.,float('inf')),(-1.,0.),(1.,2.),(2.1,1.),(1.,)):
            with self.assertRaises(ValueError):validate_friction(v)
    def test_scope_excludes_left_arms_and_cube(self):
        self.assertTrue(is_right_hand_path('/World/Robot/right_index_DIP_link/collision'))
        for p in ('/World/Robot/left_index_DIP_link/collision','/World/Robot/Link7_R/collision','/World/Cube/nZ','/World/Robot/rightish'):
            self.assertFalse(is_right_hand_path(p))
if __name__=='__main__':unittest.main()
