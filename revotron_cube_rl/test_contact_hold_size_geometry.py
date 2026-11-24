import hashlib
import tempfile
import unittest
from pathlib import Path

import contact_hold_size_geometry as geom

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class SizeGeometryTest(unittest.TestCase):
    def test_asset_is_reproducible_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cube_model.usda'
            r = geom.prepare_asset(geom.SOURCE, p)
            self.assertEqual(r, geom.verify_asset(p, r))
            self.assertEqual(r['local_bounds_m'], [[-.045]*3,[.045]*3])
            self.assertAlmostEqual(r['total_mass_kg'], .0685, places=8)
            with self.assertRaises(FileExistsError):
                geom.prepare_asset(geom.SOURCE,p)
            p.write_bytes(p.read_bytes().replace(b'convexHull', b'convexMesh', 1))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                geom.verify_asset(p,r)
            self.assertEqual(hashlib.sha256(geom.SOURCE.read_bytes()).hexdigest(),geom.SOURCE_SHA256)

    @unittest.skipIf(Usd is None, 'Requires standalone OpenUSD libraries')
    def test_openusd_composed_hulls_and_joint_pivots(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cube_model.usda'
            r = geom.prepare_asset(geom.SOURCE,p)
            stage=Usd.Stage.Open(str(p))
            self.assertTrue(stage)
            self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.)
            prims=list(stage.Traverse())
            self.assertEqual(sum(v.HasAPI(UsdPhysics.RigidBodyAPI) for v in prims),27)
            colliders=[v for v in prims if v.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertEqual(len(colliders),26)
            hist={}
            points=[]
            cache=UsdGeom.XformCache()
            for v in colliders:
                self.assertTrue(v.IsA(UsdGeom.Mesh))
                self.assertEqual(UsdPhysics.MeshCollisionAPI(v).GetApproximationAttr().Get(),'convexHull')
                local=UsdGeom.Mesh(v).GetPointsAttr().Get()
                xf=cache.GetLocalToWorldTransform(v)
                world=[xf.Transform(p) for p in local]
                points.extend(world)
                # 26 distinct cells: positioned by actual geometry, not centered overlapping hulls.
                center=tuple(round((min(p[i] for p in world)+max(p[i] for p in world))/.06) for i in range(3))
                self.assertNotIn(center,hist)
                hist[center]=True
            self.assertEqual(set(hist),{(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)}-{(0,0,0)})
            for i in range(3):
                self.assertAlmostEqual(min(p[i] for p in points),-.045)
                self.assertAlmostEqual(max(p[i] for p in points),.045)
            joints=[UsdPhysics.Joint(v) for v in prims if v.IsA(UsdPhysics.Joint)]
            self.assertEqual(len(joints),26)
            self.assertEqual({str(x.GetBody1Rel().GetTargets()[0]) for x in joints},{str(v.GetParent().GetPath()) for v in colliders})
            for j in joints:
                self.assertEqual(str(j.GetBody0Rel().GetTargets()[0]),'/World/Core')
                self.assertEqual(tuple(j.GetLocalPos0Attr().Get()),(0,0,0))
                self.assertEqual(tuple(j.GetLocalPos1Attr().Get()),(0,0,0))
            core=UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/Core'))
            for v in core.GetDiagonalInertiaAttr().Get():
                self.assertAlmostEqual(v,r['central_inertia_kg_m2'],places=10)
            self.assertAlmostEqual(sum(UsdPhysics.MassAPI(v).GetMassAttr().Get() for v in prims if v.HasAPI(UsdPhysics.MassAPI)),.0685,places=8)

if __name__=='__main__':
    unittest.main()
