import unittest
import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation
from test_contact_hold_physical_metrics import CFG, row
from contact_hold_size_face_score import face_samples, summarize_face

def track(group_turn=True,global_turn=False,disrupt=False,drop=False):
    names=['_'.join(f for f in faces if f) for faces in product(('','pX','nX'),('','pY','nY'),('','pZ','nZ')) if any(faces)]
    rows=[]
    for i in range(70):
        r=row(i);deg=min(20,max(0,i-15)*2)
        core=Rotation.from_euler('y',deg if global_turn else 0,degrees=True)
        def q(rot): return rot.as_quat()[[3,0,1,2]].tolist()
        r['cube_quaternion_wxyz']=q(core)
        r['cubelets']={'Core':{'quaternion_wxyz':q(core)}}
        for n in names:
            local=Rotation.from_euler('z',deg if group_turn and 'pZ' in n.split('_') and not(disrupt and n=='pZ') else 0,degrees=True)
            r['cubelets'][n]={'quaternion_wxyz':q(core*local)}
        r['layer_measurement'].update(maximum_joint_origin_separation_m=0,assembled_for_grasp=not(group_turn and deg>5))
        if i>24 and drop:r.update(contact_impulse=[0.,0.],cube_table_contact=True)
        rows.append(r)
    return rows

class FaceTest(unittest.TestCase):
    def test_face_in_moving_frame(self):
        rows=track(global_turn=True); angle,ok=face_samples(rows,'pZ')
        self.assertTrue(np.all(ok));self.assertAlmostEqual(angle[-1],20.)
        r=summarize_face(rows,CFG);self.assertTrue(r['qualified']);self.assertEqual(r['best_face'],'pZ')
    def test_tumble_split_layer_and_release_do_not_pass(self):
        for kw in ({'group_turn':False,'global_turn':True},{'disrupt':True},{'drop':True}):
            with self.subTest(kw=kw):self.assertFalse(summarize_face(track(**kw),CFG)['qualified'])

if __name__=='__main__':unittest.main()
