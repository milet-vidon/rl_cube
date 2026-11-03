"""Versioned runtime override for the supplied fixture-backed cube asset."""
from pxr import UsdPhysics, UsdGeom, Gf, PhysxSchema

def spawn_dynamic_cube(stage, path, position):
    existing={str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)}
    stage.GetRootLayer().subLayerPaths.append(str(path))
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()) not in existing:
            prim.SetActive(False)
    core=stage.GetPrimAtPath('/World/Core')
    if not core or not core.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Cube core missing')
    source_kinematic=UsdPhysics.RigidBodyAPI(core).GetKinematicEnabledAttr().Get()
    UsdPhysics.RigidBodyAPI(core).CreateKinematicEnabledAttr(False)
    for name in ('/World/Core','/World/Cube'):
        xform=UsdGeom.Xformable(stage.GetPrimAtPath(name))
        translation=next((op for op in xform.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate),None)
        if translation is None:translation=xform.AddTranslateOp()
        translation.Set(Gf.Vec3d(*position))
    bodies=[]
    for p in stage.Traverse():
        if (str(p.GetPath()).startswith('/World/Cube/') or p==core) and p.HasAPI(UsdPhysics.RigidBodyAPI):
            if UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get():raise ValueError('Unexpected fixed body '+str(p.GetPath()))
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(False)
            bodies.append(str(p.GetPath()))
    if len(bodies)!=27:raise ValueError(f'Expected core + 26 cubelets, got {len(bodies)}')
    return {'runtime_model_version':'dynamic_core_v2','source_core_kinematic':source_kinematic,'runtime_core_kinematic':False,'bodies':bodies,'active_physics_scenes':[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]}
