"""The preflight must inspect the actual start and swept paths, not endpoints."""
import importlib.util
import types
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np


class Scene:
    def __init__(self,initial,*args):
        self.names=['Joint1_R'];self.q=initial.copy();self.fixed=[('cube_route_envelope',)]
        self.k=types.SimpleNamespace(limits={'right_index_MCP_joint':[-1.,1.]})
        self.last_pair=None
    def valid(self,x):
        self.last_pair=('moving_link','obstacle',-.01)
        return not .4<float(x[0])<.6


def module():
    spec=importlib.util.spec_from_file_location('guard_under_test',Path(__file__).with_name('safe_path_audit.py'))
    result=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'safe_collision_plan':types.SimpleNamespace(CollisionScene=Scene)}):spec.loader.exec_module(result)
    return result


def request(initial,targets):
    return {'initial':{'Joint1_R':initial,'right_index_MCP_joint':0.},'cube_position':[0,0,1],
            'cube_quaternion':[1,0,0,0],'joint_names':['Joint1_R','right_index_MCP_joint'],
            'phases':[('right_close',targets)]}


class SafePathTests(unittest.TestCase):
    def test_actual_start_to_first_target_is_checked(self):
        self.assertFalse(module().audit(request(1.,[[0.,0.]]))['passed'])
    def test_clear_endpoints_do_not_allow_swept_collision(self):
        self.assertFalse(module().audit(request(0.,[[0.,0.],[1.,0.]]))['passed'])
    def test_finger_limit_is_checked_before_physics(self):
        result=module().audit(request(0.,[[0.,1.1]]))
        self.assertFalse(result['passed']);self.assertEqual(result['pair'][1],'joint_limits')
    def test_clear_short_motion_passes_only_geometry(self):
        result=module().audit(request(0.,[[.2,0.]]))
        self.assertTrue(result['passed']);self.assertFalse(result['physics_validated'])

if __name__=='__main__':unittest.main()
