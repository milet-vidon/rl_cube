"""CPU fixtures for native contact attribution and human demonstration recording."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from natural_face90_demo_record import COLUMNS, FRAMES, META, SUMMARY, Recorder


NAMES = ['_'.join(v for v in (a,b,c) if v) for a in ('','pX','nX')
         for b in ('','pY','nY') for c in ('','pZ','nZ') if a or b or c]
ROOT = '/World/envs/env_0000'
Q0 = torch.zeros(1,56)


class Sensor:
    def __init__(self):
        self.sensor_paths = [ROOT+'/Cube/'+n for n in NAMES]
        self.sensor_count = len(NAMES)
        self.max_contact_data_count = 128
        self.packet = []
        self.dt_args = []
    def get_raw_contact_data(self, dt):
        self.dt_args.append(dt)
        counts = torch.zeros(26,dtype=torch.long)
        starts = torch.zeros_like(counts)
        force = torch.zeros(128,dtype=torch.double)
        ids = torch.zeros(128,dtype=torch.long)
        j = 2
        for name,data in self.packet:
            i = NAMES.index(name)
            counts[i], starts[i] = len(data), j
            for k,impulse in data:
                ids[j],force[j] = k,impulse
                j += 1
            j += 3
        return force,None,None,None,counts,starts,ids


class Reducer:
    def __init__(self):
        self.calls = []
    def clear(self,e):
        self.calls.append('clear')
    def update(self,e):
        self.calls.append('update')
    def validate(self):
        self.calls.append('validate')


class FakeEnv:
    def __init__(self):
        self.dt,self.dec,self.n = 1/240,8,1
        self.names = [f'joint_{i}' for i in range(56)]
        self.ids = list(reversed(range(56)))
        self.contacts = torch.zeros(1,13)
        self.cfg = {'success':{'contact_impulse_min':.001}}
        self.domain_receipt = {'edge_m':.09}
        self.sensor = Sensor()
        self.base = self.reducer = Reducer()
        self.contact_views = [('cube',self.sensor,[0]*26)]
        self.physical = {}
        self.views = {}
        for name in ['Core',*NAMES]:
            pose = torch.tensor([[.3,.4,1.,0.,0.,0.,1.]],dtype=torch.float64)
            self.physical[name] = pose
            self.views[name] = SimpleNamespace(get_transforms=lambda p=pose:p.clone())
        self.robot = SimpleNamespace(get_dof_positions=lambda:torch.arange(56.).unsqueeze(0),
                                     get_dof_velocities=lambda:Q0,
                                     shared_metatype=SimpleNamespace(link_names=['right_thumb_DIP_link','right_index_DIP_link']),
                                     get_link_transforms=lambda:torch.tensor([[[0.,1.,2.,0,0,0,1.],[0.,2.,1.,0,0,0,1.]]]))
    def frame(self, packets):
        self.reducer.clear(self)
        for data in packets:
            self.sensor.packet = data
            self.reducer.update(self)
        self.reducer.validate()


def paths(_, actor_ids):
    actors = ['excluded','right_thumb_DIP_link','right_index_DIP_link','right_index_MCP_link',
              'right_middle_MCP_link','right_palm_link','left_index_DIP_link','right_thumb_PIP_link']
    return [ROOT+'/Robot/'+actors[i] for i in actor_ids]


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.env = FakeEnv()
        self.rec = Recorder(self.folder,self.env,{'seed':66002220})
        self.addCleanup(self.rec.close)
        patcher = patch('natural_face90_demo_record._actor_paths',side_effect=paths)
        self.resolver = patcher.start()
        self.addCleanup(patcher.stop)

    def test_native_two_finger_attribution_preserves_other_contacts_and_all_steps(self):
        e,r = self.env,self.rec
        r.begin_handoff(0,Q0)
        e.frame([[('nZ',[(1,.02),(2,.015),(3,-.005),(4,.002),(5,.004),(7,.001)]),
                  ('pZ',[(6,.3),(1,.07)])]] + [[]]*7)
        # A rigid 25-degree motion of the whole cube has zero relative face angle.
        def z_quat(degree):
            angle = math.radians(degree/2)
            return torch.tensor([0,0,math.sin(angle),math.cos(angle)],dtype=torch.float64)
        for pose in e.physical.values():
            pose[0,3:] = z_quat(25)
        a = r.record(0,'manual',{'turn':1},Q0,Q0)
        self.assertAlmostEqual(a['angle_deg'],0.)
        self.assertAlmostEqual(a['core_rotation_since_handoff_deg'],25.)
        self.assertAlmostEqual(a['thumb_Ns'],.021)
        self.assertAlmostEqual(a['index_Ns'],.02)
        self.assertAlmostEqual(a['other_Ns'],.002)
        self.assertEqual(a['thumb_index_same_substeps'],1)
        self.assertEqual(a['bodies']['right_index_DIP_link'], [0.,2.,1.,0.,0.,0.,1.])
        self.assertAlmostEqual(a['impulse_substeps_Ns'][0][1][COLUMNS.index('left_hand')],.3)
        self.assertEqual(a['q_rad'],list(range(55,-1,-1)))
        self.assertEqual(e.base.calls,['clear']+['update']*8+['validate'])
        self.assertEqual(e.sensor.dt_args,[1.]*8)
        # Only the selected nine pieces move in the second step (about outward nZ).
        for name in NAMES:
            if 'nZ' in name.split('_'):
                e.physical[name][0,3:] = z_quat(10)
        e.frame([[('nZ',[(1,.03)])],[('nZ',[(2,.03)])]] + [[]]*6)
        b = r.record(1,'manual',{},Q0,Q0)
        self.assertAlmostEqual(b['angle_deg'],15.)
        self.assertTrue(b['structure_ok'])
        self.assertEqual(b['thumb_index_same_substeps'],0)
        self.resolver.assert_called_once()
        result = r.finish('operator finished')
        self.assertIs(e.reducer,e.base)
        self.assertEqual(result['manual_thumb_index_overlap_frames'],1)
        self.assertAlmostEqual(result['final_manual_face_change_deg'],15.)
        for path in [META,SUMMARY,'demo_handoff.json']:
            data = json.loads((self.folder/path).read_bytes())
            self.assertTrue(data)
        data = (self.folder/FRAMES).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),result['frames_sha256'])
        self.assertEqual(len(data.splitlines()),2)
        self.assertIs(r.close(),result)
        with self.assertRaises(FileExistsError):
            Recorder(self.folder,e)

    def test_missing_substeps_and_duplicate_intervals_rejected_and_unsafe_marked(self):
        e,r = self.env,self.rec
        e.frame([[]]*7)
        with self.assertRaisesRegex(ValueError,'Incomplete physical interval'):
            r.record(0,'prefix',{},Q0,Q0)
        e.sensor.packet=[]
        e.reducer.update(e)
        r.record(0,'prefix',{},Q0,Q0)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            r.record(1,'prefix',{},Q0,Q0)
        r.begin_handoff(1,Q0)
        e.frame([[]]*8)
        r.record(1,'manual',{},Q0,Q0,safe=False,end_reason='support lost')
        self.assertEqual(r.finish('support lost')['status'],'failed')


if __name__=='__main__':
    unittest.main()
