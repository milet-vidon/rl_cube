"""CPU raw-packet regressions; no simulator is started."""
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from natural_face90_recipe_frames import PAD_BODY
from natural_face90_turn_contacts import install
from natural_face90_v5_contacts import PairRecorder
from natural_face90_v5_metrics import TargetContacts, actor_class


class Fixture:
    def __init__(self):
        names = ['_'.join(i for i in c if i) for c in product(('', 'pX', 'nX'),
                 ('', 'pY', 'nY'), ('', 'pZ', 'nZ')) if any(c)]
        paths = [f'/World/envs/env_{lane:04d}/Cube/{name}' for lane in range(2) for name in names]
        view = SimpleNamespace(sensor_paths=paths, sensor_count=52, max_contact_data_count=200,
                               get_raw_contact_data=lambda dt:self.packet)
        self.original = SimpleNamespace(clear=Mock(), update=Mock(), validate=Mock(), cache={})
        body = torch.tensor([[[1., 2., 3., 0., 0., 0., 1.]],
                             [[4., 5., 6., 0., 0., 1., 0.]]])
        robot=SimpleNamespace(shared_metatype=SimpleNamespace(link_names=[PAD_BODY]),
                              get_link_transforms=lambda:body)
        self.scene = s = SimpleNamespace(n=2, dec=2, home=torch.zeros(2,56), robot=robot,
            contacts=torch.zeros(2,13), reducer=self.original,
            contact_views=[('cube',view,[0]*26+[1]*26),('robot',SimpleNamespace(sensor_paths=[]),[])])
        self.raw = raw = TargetContacts(s, names)
        s.reducer = raw
        self.pair = s.reducer = PairRecorder(s)
        self.monitor = SimpleNamespace(r=raw)
        self.tracker = install(s, self.monitor, 0, torch.tensor([.02, 0., .03]))
        self.ids = {
            11:(0,PAD_BODY), 12:(0,'right_middle_DIP_link'), 13:(0,'right_index_PIP_link'),
            14:(0,'left_thumb_DIP_link'), 21:(1,PAD_BODY), 22:(1,'right_thumb_CMR_link'),
        }
        self.paths = {k:f'/World/envs/env_{lane:04d}/Robot/{name}' for k,(lane,name) in self.ids.items()}
        # Pre-resolve the original reducer just as the native first-ID lookup would.
        self.original.cache = dict(self.paths)
        raw.cache = {k:actor_class(p) for k,p in self.paths.items()}
        raw.keys = torch.tensor(sorted(raw.cache))
        raw.code = torch.tensor([raw.cache[k] for k in sorted(raw.cache)])
        self.sensors = {(i//26, p.rsplit('/',1)[-1]):i for i,p in enumerate(paths)}
        self.clear()

    def clear(self):
        self.scene.reducer.clear(self.scene)

    def advance(self, contacts):
        forces=torch.full((200,),float('nan'));actors=torch.full((200,),999,dtype=torch.long)
        counts=torch.zeros(52,dtype=torch.int32);starts=torch.zeros_like(counts);index=3
        # Reverse the sensor order and leave unfilled gaps between groups.
        for (lane,name),records in reversed(list(contacts.items())):
            s=self.sensors[lane,name];counts[s]=len(records);starts[s]=index
            for actor,force in records:
                forces[index]=force;actors[index]=actor;index+=1
            index+=2
        self.packet=(forces,None,None,None,counts,starts,actors)
        with patch('natural_face90_turn_contacts._actor_paths',
                   side_effect=lambda scene,ids:[self.paths[k] for k in ids]):
            self.scene.reducer.update(self.scene)


class IndexAttribution(unittest.TestCase):
    def test_distal_and_target_must_be_the_same_contact(self):
        f=Fixture()
        self.assertIs(f.scene.reducer.original,f.pair)
        f.advance({(0,'nZ'):[(11,-.2),(12,.3),(13,.4),(14,.5)],
                   (1,'nZ'):[(22,.8)], (1,'pZ'):[(21,.6)], (0,'pZ'):[(11,.7)]})
        f.advance({(1,'pX_nZ'):[(21,.1)], (0,'pY_nZ'):[(11,.05)]})
        f.scene.reducer.validate()
        torch.testing.assert_close(f.tracker.index_impulse,torch.tensor([.25,.1]).double())
        torch.testing.assert_close(f.tracker.other_digit_impulse,torch.tensor([.7,.8]).double())
        self.assertEqual(f.raw.acc.shape,(2,5))
        torch.testing.assert_close(f.raw.acc[:,1],torch.tensor([.7,.6]).double())
        self.assertEqual(f.raw.count,2)
        self.assertEqual(f.original.update.call_count,2)
        f.original.validate.assert_called_once()
        self.assertEqual(f.pair.substep,2)
        f.clear()
        torch.testing.assert_close(f.tracker.index_impulse,torch.zeros(2).double())
        torch.testing.assert_close(f.tracker.other_digit_impulse,torch.zeros(2).double())
        f.advance({(0,'nZ'):[(13,.1)]})
        f.advance({})
        f.scene.reducer.validate()
        self.assertEqual(f.tracker.index_impulse.tolist(),[0.,0.])

    def test_original_faults_remain_fatal(self):
        f=Fixture()
        with self.assertRaisesRegex(ValueError,'Cross-lane'):
            f.advance({(1,'nZ'):[(11,.3)]})
        f=Fixture();f.advance({})
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            f.scene.reducer.validate()
        f=Fixture();f.advance({});f.advance({})
        f.tracker.index_impulse[1]=1.
        with self.assertRaisesRegex(ValueError,'exceeds'):
            f.scene.reducer.validate()
        f=Fixture();f.paths[11]=f.paths[21]
        with self.assertRaisesRegex(ValueError,'identity or lane'):
            f.advance({(0,'nZ'):[(11,.3)]})

    def test_measured_pad_pose_remains_independent_of_impulse(self):
        f=Fixture()
        expected=torch.tensor([[1.02,2.,3.03],[3.98,5.,6.03]],dtype=torch.double)
        torch.testing.assert_close(f.tracker.pad_position_world(),expected)
        self.assertEqual(f.tracker.index_impulse.tolist(),[0.,0.])
        with self.assertRaisesRegex(ValueError,'PairRecorder'):
            install(f.scene,f.monitor,0,torch.zeros(3))


if __name__ == '__main__':
    unittest.main()
