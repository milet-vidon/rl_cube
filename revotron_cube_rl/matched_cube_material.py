"""Reduce cube/cube friction while matching baseline left/cube and table/cube pairs.

This is an uncalibrated simulation development option, not a physical calibration.
Only applies to the recorded average-combine baseline (.9,.8), default (.5,.5),
and table (.8,.7). Right-only material, if requested, remains a separate override.
"""
from contextlib import contextmanager
from right_contact_material import validate_friction

def pair_coefficients(internal):
    internal = validate_friction(internal)
    left = (.7, .65)
    table = (.85, .75)
    if any(v > bound for v, bound in zip(internal, left)):
        raise ValueError('Internal friction exceeds matched external pair coefficients')
    return {'internal': internal, 'left': left, 'table': table}

def apply_matched_materials(stage, internal):
    from pxr import UsdPhysics, UsdShade, PhysxSchema
    coefficients = pair_coefficients(internal)
    def values(prim):
        api = UsdPhysics.MaterialAPI(prim)
        return [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()]
    def require(prim, expected):
        actual = values(prim)
        if any(abs(a-b) > 1e-6 for a,b in zip(actual, expected)):
            raise ValueError('Unexpected baseline material: ' + str(prim.GetPath()))
        mode = PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get()
        if mode not in (None, 'average'):
            raise ValueError('Baseline material must use average friction combination')
    cube = stage.GetPrimAtPath('/World/Material')
    default = stage.GetPrimAtPath('/physicsScene/defaultMaterial')
    table = stage.GetPrimAtPath('/World/Table/geometry/material')
    for prim, expected in ((cube,(.9,.8)), (default,(.5,.5)), (table,(.8,.7))):
        if not prim: raise ValueError('Required baseline material is absent')
        require(prim, expected)
    left = [p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Robot/left_') and p.HasAPI(UsdPhysics.CollisionAPI)]
    if not left: raise ValueError('No left hand colliders')
    for prim in left:
        material,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if material and material.GetPath() != default.GetPath():
            raise ValueError('Left hand has an unexpected explicit material')
    def set_material(prim, friction, mode):
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr(friction[0]); api.CreateDynamicFrictionAttr(friction[1])
        api.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr(mode)
    skin = UsdShade.Material.Define(stage, '/World/MatchedLeftHandMaterial')
    set_material(skin.GetPrim(), coefficients['left'], 'max')
    bound = []
    for prim in left:
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(skin, UsdShade.Tokens.strongerThanDescendants, 'physics')
        material,_ = binding.ComputeBoundMaterial('physics')
        if material.GetPath() != skin.GetPath(): raise ValueError('Matched left material binding failed')
        bound.append(str(prim.GetPath()))
    set_material(table, coefficients['table'], 'max')
    set_material(cube, coefficients['internal'], 'average')
    report = {'coefficients': coefficients, 'left_colliders': bound,
              'baseline_pair_coefficients': {'left_cube':[.7,.65], 'table_cube':[.85,.75]},
              'combine_modes': {'left':'max','table':'max','cube':'average'},
              'basis': 'Match baseline averaged left/cube and table/cube Coulomb coefficients; internal cube friction varies. Not a calibration or guarantee of identical trajectories.',
              'verified_authored_values': {'cube':values(cube), 'left':values(skin.GetPrim()), 'table':values(table)}}
    print('MATCHED_CUBE_MATERIAL', coefficients, 'left_colliders', len(bound), flush=True)
    return report

@contextmanager
def matched_cube_material(internal, module=None):
    if internal is None:
        yield
        return
    pair_coefficients(internal)
    if module is None:
        import articulated_cube_env as module
    original = module.DynamicCubeHandle
    def build(stage, *args, **kwargs):
        handle = original(stage, *args, **kwargs)
        report = apply_matched_materials(stage, internal)
        handle.model.update(matched_contact_materials=report, runtime_friction=list(internal))
        return handle
    module.DynamicCubeHandle = build
    try:
        yield
    finally:
        module.DynamicCubeHandle = original
