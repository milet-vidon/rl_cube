"""Exercise the production raw-contact kernel AND rejection checks on Warp's CPU target.

No Isaac process, CUDA runtime allocation, or project scene is required. The synthetic
raw view follows the PhysX (count, start, other-actor ID, signed normal impulse) API.
"""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from contact_hold_cuda_scene_reduce import GpuContacts


META = ((0,1,0,1,9), (0,1,0,0,2), (0,0,1,-1,-1), (1,1,0,0,-1), (1,0,1,-1,-1))


def frame(kind, *, counts=(4,3), starts=(0,4), ids=(0,1,1,2,3,4,4,4),
          force=(1.,-2.,1e-8,4.,2.,1.,4.,1.), actor_path=None):
    reducer = GpuContacts.__new__(GpuContacts)
    reducer.cache = dict(enumerate(META))
    reducer.keys = torch.arange(5, dtype=torch.int64)
    reducer.meta = torch.tensor(META, dtype=torch.int32)
    reducer.missing = torch.empty(8, dtype=torch.int64)
    reducer.state = torch.zeros(5, dtype=torch.int32)
    reducer.stream = None
    raw = (torch.tensor(force, dtype=torch.float32), None,None,None,
           torch.tensor(counts, dtype=torch.int32), torch.tensor(starts, dtype=torch.int32),
           torch.tensor(ids, dtype=torch.int64))
    view = NS(max_contact_data_count=8, get_raw_contact_data=lambda dt: raw)
    reducer.views = [(kind, view, wp.array([0,1], dtype=wp.int32, device='cpu'))]
    scene = NS(n=2, cross=0, contacts=torch.zeros(2,13,dtype=torch.float64),
               feature_contacts=torch.zeros(2,10,dtype=torch.float64),
               hand_impulse=torch.zeros(2,2,dtype=torch.float64))
    reducer.acc = [torch.zeros_like(a) for a in (scene.contacts,scene.feature_contacts,scene.hand_impulse)]
    seen = []
    def lookup(ids):
        seen.extend(int(i) for i in ids.numpy())
        return [actor_path] if actor_path is not None else []
    scene.sv_warp_contact = NS(get_other_actor_paths_from_ids=lookup)
    native_launch = wp.launch
    def cpu_launch(*args, **kwargs):
        kwargs.update(device='cpu',stream=None)
        return native_launch(*args,**kwargs)
    def update():
        # Change only the execution device, leaving the kernel, retries, metadata
        # resolution, fail-closed ordering, and accumulator commits unchanged.
        with patch('contact_hold_cuda_scene_reduce.wp.launch',side_effect=cpu_launch), patch.object(
                torch.Tensor,'cuda',lambda t: t):
            reducer.update(scene)
    return scene, update, seen


class ContactReduction(unittest.TestCase):
    def test_cube_channels_and_feature_threshold(self):
        scene, update, _ = frame(1)
        update()
        contact=np.zeros((2,13));contact[0,[2,9,12]]=[2+float(np.float32(1e-8)),1,4];contact[1,12]=5
        feature=np.zeros((2,10));feature[0,[2,9]]=[2,1]
        np.testing.assert_array_equal(scene.contacts,contact)
        np.testing.assert_array_equal(scene.feature_contacts,feature)
        np.testing.assert_array_equal(scene.hand_impulse,[[contact[0,2],1],[2,0]])

    def test_robot_robot_pairs_count_half_but_robot_table_does_not(self):
        scene, update, _ = frame(0)
        update()
        expected=np.zeros((2,13));expected[:,[10,11]]=[[1.5+float(np.float32(1e-8))*.5,4],[1,5]]
        np.testing.assert_array_equal(scene.contacts,expected)
        self.assertEqual(scene.hand_impulse.count_nonzero(),0)

    def test_range_overflow_negative_count_and_exactly_full_buffer_reject(self):
        for kwargs in ({'starts':(7,4)}, {'starts':(-1,4)}, {'counts':(-1,3)}, {'counts':(4,4)}):
            with self.subTest(kwargs=kwargs):
                scene, update, _ = frame(1,**kwargs)
                with self.assertRaisesRegex(RuntimeError,'buffer exhausted'):update()
                self.assertEqual(scene.contacts.count_nonzero(),0)

    def test_nonfinite_or_cross_lane_is_rejected_without_committing(self):
        for kwargs, msg in (({'force':(float('nan'),2.,1e-8,4.,2.,1.,4.,1.)},'Nonfinite'),
                            ({'force':(float('inf'),2.,1e-8,4.,2.,1.,4.,1.)},'Nonfinite'),
                            ({'ids':(4,1,1,2,3,4,4,4)},'Cross-lane')):
            with self.subTest(msg=msg,kwargs=kwargs):
                scene, update, _ = frame(1,**kwargs)
                with self.assertRaisesRegex(ValueError,msg):update()
                self.assertEqual(scene.contacts.count_nonzero(),0)
                self.assertEqual(scene.cross,int(msg=='Cross-lane'))

    def test_missing_id_is_resolved_once_and_the_entire_frame_commits_once(self):
        # High-bit actor handle stored as signed int64 on GPU; Isaac consumes uint64.
        scene, update, looked_up = frame(1,ids=(-1,-1,1,2,3,4,4,4),
                                         actor_path='/World/envs/env_0000/Robot/left_index_distal')
        update()
        self.assertEqual(looked_up,[2**64-1])
        self.assertEqual(scene.contacts[0,1],3.)
        self.assertEqual(scene.contacts[0,12],4.)
        self.assertEqual(scene.hand_impulse[0,0],3.+float(np.float32(1e-8)))

    def test_unmapped_id_and_foreign_path_are_rejected(self):
        for path,msg in ((None,'length mismatch'),('/Outside/Robot/left_index_link','Unmapped contact actor'),
                         ('/World/envs/env_0001/Robot/left_index_link','Cross-lane')):
            with self.subTest(path=path):
                scene, update, _ = frame(1,ids=(9,1,1,2,3,4,4,4),actor_path=path)
                with self.assertRaisesRegex(ValueError,msg):update()
                self.assertEqual(scene.contacts.count_nonzero(),0)


if __name__=='__main__':
    unittest.main()
