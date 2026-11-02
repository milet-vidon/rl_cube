import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from cube_state import layer_delta

def q(rotation):
    x=rotation.as_quat();return x[[3,0,1,2]]

class LayerTests(unittest.TestCase):
    def test_whole_cube_rotation_is_not_layer_turn(self):
        identity=q(R.identity());global_q=q(R.from_euler('xyz',[45,20,90],degrees=True))
        result=layer_delta(identity,identity,global_q,global_q,[0,0,1])
        self.assertAlmostEqual(result['signed_angle_deg'],0.)
    def test_signed_turn_under_arbitrary_core_pose(self):
        initial=R.from_euler('xyz',[20,30,5],degrees=True);core=R.from_euler('xyz',[50,-20,80],degrees=True)
        for angle in [-90,90]:
            result=layer_delta(q(initial),q(initial),q(core),q(core*R.from_euler('z',angle,degrees=True)),[0,0,1])
            self.assertAlmostEqual(result['signed_angle_deg'],angle)
            self.assertAlmostEqual(result['off_axis_error_deg'],0.)
    def test_wrong_axis_is_rejected_by_swing(self):
        identity=q(R.identity());result=layer_delta(identity,identity,identity,q(R.from_euler('x',90,degrees=True)),[0,0,1])
        self.assertAlmostEqual(result['off_axis_error_deg'],90.)

if __name__=='__main__':unittest.main()
