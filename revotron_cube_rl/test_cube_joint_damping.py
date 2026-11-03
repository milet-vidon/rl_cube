import math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pxr import Usd,UsdPhysics
from cube_joint_damping import source_joint_parameters,apply_source_joint_damping,verify_passive_joint_damping,source_joint_damping

ROOT=Path(__file__).resolve().parent.parent/'isaac_mujoco_cube'
SOURCE=ROOT/'upstream/mujoco_cube-main/cube_3x3x3.xml'

class PassiveJointTests(unittest.TestCase):
    def stage(self):
        stage=Usd.Stage.Open(str(ROOT/'cube_model.usda'));stage.SetEditTarget(stage.GetSessionLayer());return stage
    def test_source_units_and_inventory(self):
        specs=source_joint_parameters(SOURCE)['joints'];self.assertEqual(len(specs),26)
        for spec in specs.values():
            # Same torque for a 1 rad/s relative angular velocity.
            self.assertAlmostEqual(spec['usd_damping_Nm_per_deg_s']*180/math.pi,.0005)
    def test_reject_nonpassive_source(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.xml';path.write_text(SOURCE.read_text().replace('0.0005 0.0005 0.0005','-0.0005 -0.0005 -0.0005'))
            with self.assertRaises(ValueError):source_joint_parameters(path)
    def test_migration_preserves_connectivity_frames_and_collision(self):
        stage=self.stage();before={}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):continue
            joint=UsdPhysics.Joint(prim)
            before[str(prim.GetPath())]=(joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets(),joint.GetLocalPos0Attr().Get(),joint.GetLocalPos1Attr().Get(),joint.GetLocalRot0Attr().Get(),joint.GetLocalRot1Attr().Get(),joint.GetCollisionEnabledAttr().Get())
        report=apply_source_joint_damping(stage,ROOT/'cube_model.usda');self.assertEqual(report['passive_axes'],66)
        for path,old in before.items():
            joint=UsdPhysics.Joint(stage.GetPrimAtPath(path))
            self.assertEqual(old,(joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets(),joint.GetLocalPos0Attr().Get(),joint.GetLocalPos1Attr().Get(),joint.GetLocalRot0Attr().Get(),joint.GetLocalRot1Attr().Get(),joint.GetCollisionEnabledAttr().Get()))
        self.assertTrue(verify_passive_joint_damping(stage,report)['verified_authored_parameters'])
    def test_active_target_is_rejected(self):
        stage=self.stage();report=apply_source_joint_damping(stage,ROOT/'cube_model.usda');spec=next(iter(report['joints'].values()))
        UsdPhysics.DriveAPI(stage.GetPrimAtPath(spec['usd_path']),spec['drive_axes'][0]).CreateTargetVelocityAttr(1.)
        with self.assertRaisesRegex(ValueError,'active'):verify_passive_joint_damping(stage,report)
    def test_factory_restored_after_failure(self):
        original=lambda *a:SimpleNamespace(model={});module=SimpleNamespace(DynamicCubeHandle=original)
        with patch('cube_joint_damping.apply_source_joint_damping',side_effect=ValueError('bad source')):
            with self.assertRaises(ValueError):
                with source_joint_damping(True,module):module.DynamicCubeHandle(None,None)
        self.assertIs(module.DynamicCubeHandle,original)

if __name__=='__main__':unittest.main()
