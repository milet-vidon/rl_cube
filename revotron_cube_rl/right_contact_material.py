"""Right-hand-only friction comparison, applied before the first PhysX reset."""
import math

def validate_friction(values):
    values=tuple(float(v) for v in values)
    if len(values)!=2 or not all(math.isfinite(v) and 0<=v<=2. for v in values) or values[0]<values[1]:
        raise ValueError('Use finite static/dynamic friction with 0 <= dynamic <= static <= 2')
    return values

def is_right_hand_path(path):
    return str(path).startswith('/World/Robot/right_')

def apply_right_material(stage,friction):
    from pxr import UsdShade,UsdPhysics,PhysxSchema
    static,dynamic=validate_friction(friction)
    left=[p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Robot/left_') and p.HasAPI(UsdPhysics.CollisionAPI)]
    def material_path(prim):
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        return str(material.GetPath()) if material else None
    before={str(p.GetPath()):material_path(p) for p in left}
    material=UsdShade.Material.Define(stage,'/World/RightHandContactMaterial')
    physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(static);physics.CreateDynamicFrictionAttr(dynamic);physics.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('max')
    bound=[]
    for prim in stage.Traverse():
        if not is_right_hand_path(prim.GetPath()) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        if material_path(prim)!=str(material.GetPath()):raise RuntimeError('Right hand material binding failed')
        bound.append(str(prim.GetPath()))
    if not bound:raise RuntimeError('No right-hand collision shapes')
    after={str(p.GetPath()):material_path(p) for p in left}
    if before!=after:raise RuntimeError('Left-hand contact material changed unexpectedly')
    print('RIGHT_HAND_MATERIAL_BOUND',len(bound),'friction',static,dynamic,'left_bindings_unchanged',len(before),flush=True)
    return {'runtime_model_version':'dynamic_grasp_right_material_v5','left_material_bindings':before,'runtime_right_hand_friction':[static,dynamic],'right_hand_friction_combine_mode':'max',
            'right_hand_material_colliders':bound,'left_material_bindings_unchanged':True,
            'friction_basis':'uncalibrated right-hand-only simulation comparison; cube/cube and left-hand materials unchanged'}
