import itertools,unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from layer_turn_metrics import measure_layers

def scene(turn=0.,core=None,one=False):
    c=core or R.identity();names=['_'.join(a for a in x if a)for x in itertools.product(['','pX','nX'],['','pY','nY'],['','pZ','nZ'])if any(x)]
    result={'Core':{'quaternion_wxyz':c.as_quat()[[3,0,1,2]].tolist(),'position_m':[.4,.2,1.2],'angular_velocity_rad_s':[0.,0.,0.]}}
    for name in names:
        q=c*R.from_rotvec(np.array([0,0,-np.radians(turn if 'nZ'in name.split('_')and(not one or name=='nZ')else 0)]))
        result[name]={'quaternion_wxyz':q.as_quat()[[3,0,1,2]].tolist(),'position_m':[.4,.2,1.2],'angular_velocity_rad_s':[0.,0.,0.]}
    return result

class LayerMetricsTests(unittest.TestCase):
    def test_whole_cube_rotation_does_not_count(self):
        m=measure_layers(scene(core=R.from_euler('xyz',[90,24,-31],degrees=True)));self.assertFalse(m['geometry_qualified']);self.assertAlmostEqual(m['face_angle_deg'],0.)
    def test_complete_face_with_global_pose_change(self):
        self.assertTrue(measure_layers(scene(90,core=R.from_euler('xyz',[55,13,70],degrees=True)))['geometry_qualified'])
    def test_center_only_not_a_face_turn(self):self.assertFalse(measure_layers(scene(90,one=True))['geometry_qualified'])
    def test_opposite_direction_fails(self):self.assertFalse(measure_layers(scene(-90))['geometry_qualified'])
    def test_separated_pieces_fail(self):
        s=scene(90);s['nZ']['position_m'][0]+=.01;self.assertFalse(measure_layers(s)['geometry_qualified'])
    def test_other_layer_must_stay_still(self):
        s=scene(90);s['pZ']['quaternion_wxyz']=R.from_euler('z',12,degrees=True).as_quat()[[3,0,1,2]].tolist();self.assertFalse(measure_layers(s)['geometry_qualified'])

if __name__=='__main__':unittest.main()
