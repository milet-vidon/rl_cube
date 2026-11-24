import unittest
import numpy as np
from types import SimpleNamespace
from opposed_cube_clearance import CubeClearance

class CubeClearanceTests(unittest.TestCase):
    def test_margin_stricter_than_final_collision_gate(self):
        self.assertGreater(CubeClearance.minimum_distance('right_index_DIP_rubber_link','target_nZ_layer'),-.002)
        self.assertGreater(CubeClearance.minimum_distance('right_index_DIP_rubber_link','stationary_layers'),.002)
        self.assertGreater(CubeClearance.minimum_distance('right_index_PIP_link','target_nZ_layer'),.002)
    def test_penalizes_observed_rejected_penetration(self):
        scene=SimpleNamespace(C=np.eye(3),center=np.zeros(3),distance=lambda a,b:-.002024654 if b[0]=='target_nZ_layer' else .01)
        terms=CubeClearance(scene).residual([('right_index_DIP_rubber_link',)])
        self.assertEqual(terms[0],0.)
        self.assertGreater(terms[1],0.)
    def test_swept_hand_geometry_checked_in_cube_frame(self):
        seen=[]
        def distance(a,b):
            seen.append((b[0],a[2][:3,3].copy()))
            return .01
        scene=SimpleNamespace(C=np.eye(3),center=np.array([2.,3.,4.]),turn_degrees=90.,distance=distance)
        transform=np.eye(4);transform[:3,3]=[3.,3.,4.]
        CubeClearance(scene).residual([('right_index_DIP_link',None,transform,np.zeros(3),.01)])
        self.assertEqual(len(seen),5)
        np.testing.assert_allclose(seen[-1][1],[2.,2.,4.],atol=1e-12)
        self.assertEqual(seen[-1][0],'stationary_layers')

if __name__=='__main__':unittest.main()
