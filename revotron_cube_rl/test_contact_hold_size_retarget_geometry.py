import json, unittest
import numpy as np
from contact_hold_size_retarget_geometry import Hand, FINGERS, SOURCE, build


class GraspCalibrationTest(unittest.TestCase):
    def test_urdf_frames_and_source_face_distances(self):
        g=json.loads(SOURCE.read_text()); hand=Hand(g);q=g['joint_targets']
        inv=np.linalg.inv(hand.k.fk_link(q,'left_palm_link'))
        for f in FINGERS:
            a=g['contact_anchors'][f];fk=hand.fk(f,q)[a['link']]
            np.testing.assert_allclose(fk,inv@hand.k.fk_link(q,a['link']),atol=1e-12)
            seen=hand.features(f,q,hand.c0,.057)['dist']
            self.assertAlmostEqual(seen,g['cube_contact_distances_m'][f],places=6)

    def test_widened_open_and_close_are_distinct(self):
        grasp,r=build();src=json.loads(SOURCE.read_text())
        self.assertEqual(grasp['support_joint_targets'],src['support_joint_targets'])
        self.assertEqual(grasp['cube_rotation_palm'],src['cube_rotation_palm'])
        self.assertGreater(r['retarget_90_open']['span_min_mm'],r['original_90_open']['span_max_mm']+30)
        for finger in FINGERS:
            self.assertEqual(grasp['contact_anchors'][finger]['point_local'],src['contact_anchors'][finger]['point_local'])
            self.assertGreater(r['retarget_90_open']['fingers'][finger]['distal_mesh_gap_mm'],4)
            self.assertLess(abs(r['retarget_90_closed']['fingers'][finger]['distal_mesh_gap_mm']),2)
        self.assertGreater(r['closure_min_cross_finger']['min_gap_mm'],1)
        self.assertGreater(r['closure_table_min_mm'],2)
        self.assertFalse(grasp['physics_verified'])

if __name__=='__main__':
    unittest.main()
