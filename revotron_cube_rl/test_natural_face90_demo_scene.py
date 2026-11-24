"""Exercise the pinned constructor with the real 90 mm USD and offline attach callbacks."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

from contact_hold90_cuda_domain import _read, _sha, _source
from natural_face90_demo_scene import single_world_type
from natural_face90_recipe_contacts import capacity_plan, private_scene_type, verify_capacity
from natural_face90_v5_worker import ARCHIVE_PLAN


class SingleWorldAttachTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('contact_hold_cuda_scene.py')
        _source(path.name)
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scene')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        self.source = Path(_read(ARCHIVE_PLAN)['scene_source'])
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, '/physicsScene')
        self.sim = SimpleNamespace(stage=stage, reset=Mock(), set_setting=Mock())
        self.prepare = Mock()
        self.native = Mock()
        self.view = Mock()
        self.view.create_articulation_view.return_value = None
        self.order = Mock()
        self.order.attach_mock(self.prepare, 'prepare')
        self.order.attach_mock(self.sim.reset, 'reset')
        self.order.attach_mock(self.native.register_replicator, 'register')
        self.order.attach_mock(self.view.create_articulation_view, 'articulation')
        # Only the Kit, CUDA allocation and native tensor/PhysX interfaces are doubles.
        # The constructor still composes the archived USD, resolves external references,
        # checks every source digest and verifies 27 free cube bodies and joint paths.
        scope = dict(Path=Path, read=_read, sha=_sha, np=np, Usd=Usd, UsdGeom=UsdGeom,
                     UsdPhysics=UsdPhysics, Gf=Gf, PhysxSchema=Mock(), physx_config=dict,
                     simulation_config=dict, sim_utils=SimpleNamespace(SimulationContext=lambda _: self.sim),
                     torch=SimpleNamespace(as_tensor=lambda data, **_: np.asarray(data), int32=np.int32,
                                           arange=lambda n, **_: np.arange(n)),
                     omni=SimpleNamespace(physx=SimpleNamespace(get_physx_replicator_interface=lambda: self.native)),
                     tensors=SimpleNamespace(create_simulation_view=lambda _: self.view),
                     prepare_single_world=self.prepare, LANE=re.compile(r'^/World/envs/env_(\d+)(/.*)$'))
        exec(compile(ast.Module(body=[init], type_ignores=[]), str(path), 'exec'), scope)
        self.base = type('ArchivedScene', (), {'__init__': scope['__init__']})
        self.addCleanup(UsdUtils.StageCache.Get().Erase, stage)

    def test_only_single_world_is_parsed_through_standard_attach(self):
        demo = single_world_type(self.base)
        with self.assertRaisesRegex(RuntimeError, 'env_0000/Robot/root_joint'):
            demo(self.source, 1)
        self.assertEqual([c[0] for c in self.order.mock_calls], ['prepare', 'reset', 'articulation'])
        self.prepare.assert_called_once_with(self.sim)
        self.native.register_replicator.assert_not_called()
        self.view.create_articulation_view.assert_called_once_with('/World/envs/env_*/Robot/root_joint')
        world = self.sim.stage.GetPrimAtPath('/World/envs/env_0000')
        self.assertTrue(world.GetPrimAtPath('Robot/root_joint').HasAPI(UsdPhysics.ArticulationRootAPI))
        cube = world.GetPrimAtPath('Cube')
        self.assertEqual(sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(cube)), 26)
        self.assertTrue(world.GetPrimAtPath('Core').HasAPI(UsdPhysics.RigidBodyAPI))

    def test_both_scoped_contact_views_receive_the_full_capacity(self):
        class EndOfReadback(Exception):
            pass
        inspected = []
        def after_contact_views(scene):
            inspected.extend(verify_capacity(scene, capacity_plan(1))['views'])
            raise EndOfReadback

        self.base.__init__.__globals__['GpuContacts'] = after_contact_views
        cfg = _read(self.source/'config.json')
        self.view.create_articulation_view.return_value = SimpleNamespace(count=1,
            prim_paths=['/World/envs/env_0000/Robot/root_joint'], shared_metatype=SimpleNamespace(dof_names=cfg['joint_names']))
        self.view.create_rigid_body_view.side_effect = lambda path: SimpleNamespace(prim_paths=[path.replace('*','0000')])
        def contact_view(patterns, filters=None, *, max_contact_data_count):
            if isinstance(patterns, str):  # Single warp contact view for actor lookups.
                return Mock()
            paths = [s.replace('*', '0000') for s in patterns]
            return SimpleNamespace(sensor_paths=paths, sensor_count=len(paths), max_contact_data_count=max_contact_data_count)
        self.view.create_rigid_contact_view.side_effect = contact_view
        with self.assertRaises(EndOfReadback):
            single_world_type(self.base)(self.source, 1)
        self.assertEqual([(r['kind'],r['records']) for r in inspected], [('robot',4096),('cube',4096)])
        self.assertEqual(inspected[1]['sensors'], 26)

    def test_original_zero_copy_failure_and_recipe_multilane_registration(self):
        demo = single_world_type(self.base)
        train = private_scene_type(self.base)
        for cls, n in ((self.base, 1), (train, 4)):
            with self.subTest(lanes=n):
                self.order.reset_mock()
                self.sim.stage = Usd.Stage.CreateInMemory()
                UsdPhysics.Scene.Define(self.sim.stage, '/physicsScene')
                self.addCleanup(UsdUtils.StageCache.Get().Erase, self.sim.stage)
                with self.assertRaisesRegex(AttributeError, 'count'):
                    cls(self.source, n)
                self.assertEqual([c[0] for c in self.order.mock_calls], ['prepare', 'register', 'reset', 'articulation'])
                sid, exclude, attach_end, _ = self.native.register_replicator.call_args.args
                self.assertIn('/World/envs', exclude(sid))
                attach_end(sid)
                self.native.replicate.assert_called_with(sid, '/World/envs/env_0000', n-1, True, False)
        self.assertIn('recipe_capacity', train.__init__.__code__.co_names)
        self.assertNotIn('demo_capacity', train.__init__.__globals__)
        self.assertNotIn('demo_capacity', self.base.__init__.__globals__)
        self.assertIsNot(demo.__init__, train.__init__)


if __name__ == '__main__':
    unittest.main()
