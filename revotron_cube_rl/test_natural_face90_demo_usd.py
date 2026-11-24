"""Exercise the demo's actual USD mount on CPU, including damaged references."""
import json
from pathlib import Path
import tempfile
import unittest

from pxr import Usd, UsdGeom, UsdPhysics

from natural_face90_demo_usd import check_demo_usd


class MountedDemoUsdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan = Path(__file__).with_name('runs')/'contact_hold90_long_v2/round_027/validate/wave_0000_0032/plan.json'
        cls.source = Path(json.loads(plan.read_bytes())['scene_source'])

    def test_frozen_scene_has_single_robot_and_complete_cube(self):
        self.assertEqual(check_demo_usd(self.source), dict(
            articulation_path='/World/envs/env_0000/Robot/root_joint', robot_joints=56, cube_bodies=27))

    def test_reject_disconnected_robot_and_kinematic_cube(self):
        for edit, expected in [('root_joint', 'articulation'), ('cubelet', 'dynamic')]:
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)
                (source/'config.json').write_bytes((self.source/'config.json').read_bytes())
                stage = Usd.Stage.CreateNew(str(source/'dynamic_training_scene.usda'))
                root = UsdGeom.Xform.Define(stage, '/World').GetPrim()
                root.GetReferences().AddReference(str(self.source/'dynamic_training_scene.usda'), '/World')
                if edit == 'root_joint':
                    stage.GetPrimAtPath('/World/Robot/root_joint').SetActive(False)
                else:
                    UsdPhysics.RigidBodyAPI.Get(stage,'/World/Cube/nZ').CreateKinematicEnabledAttr(True)
                stage.GetRootLayer().Save()
                with self.assertRaisesRegex(ValueError, expected):
                    check_demo_usd(source)
