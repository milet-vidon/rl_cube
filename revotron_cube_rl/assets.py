"""Stage-1 free rigid cube with the original mujoco_cube geometry/texture.
The 26 pieces are one compound body: internal layers deliberately locked for pickup.
No constraint to the robot/world is created. Later unlocked-cube stages are separate.
"""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf
ROOT=Path(__file__).resolve().parent

def build_locked_cube():
    dst=ROOT/'locked_cube.usda'
    stage=Usd.Stage.CreateNew(str(dst))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    body=UsdGeom.Xform.Define(stage,'/Cube');stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.0685)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(False)
    mat=UsdShade.Material.Define(stage,'/Cube/StickerMaterial')
    physics=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(.9);physics.CreateDynamicFrictionAttr(.8);physics.CreateRestitutionAttr(0.)
    shader=UsdShade.Shader.Define(stage,'/Cube/StickerMaterial/Shader');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.7)
    tex=UsdShade.Shader.Define(stage,'/Cube/StickerMaterial/Texture');tex.CreateIdAttr('UsdUVTexture')
    cube_root=Path(os.environ.get('ISAAC_MUJOCO_CUBE_ROOT', str(ROOT.parent/'isaac_mujoco_cube'))).expanduser().resolve()
    source_root=cube_root/'upstream'/'mujoco_cube-main'
    sticker=(source_root/'assets'/'sticker.png').resolve()
    xml_path=(source_root/'cube_3x3x3.xml').resolve()
    if not sticker.is_file() or not xml_path.is_file():
        raise FileNotFoundError(f'mujoco_cube assets not found under {source_root}; set ISAAC_MUJOCO_CUBE_ROOT for this host.')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(sticker.as_posix())
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    reader=UsdShade.Shader.Define(stage,'/Cube/StickerMaterial/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    root=ET.parse(xml_path).getroot()
    meshes={m.get('name'):m for m in root.findall('asset/mesh')}
    for b in root.findall('worldbody/body/body'):
        geom=b.find('geom');m=meshes[geom.get('mesh')]
        center=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
        v=np.fromstring(m.get('vertex'),sep=' ').reshape(-1,3)+center
        f=np.fromstring(m.get('face'),sep=' ',dtype=int).reshape(-1,3)
        shape=UsdGeom.Mesh.Define(stage,'/Cube/'+b.get('name'))
        shape.CreatePointsAttr([Gf.Vec3f(*p) for p in v]);shape.CreateFaceVertexCountsAttr([3]*len(f));shape.CreateFaceVertexIndicesAttr(f.flatten().tolist());shape.CreateSubdivisionSchemeAttr('none')
        uv=np.fromstring(m.get('texcoord'),sep=' ').reshape(-1,2)
        UsdGeom.PrimvarsAPI(shape).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*p) for p in uv])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        pc=PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim());pc.CreateContactOffsetAttr(.0005);pc.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
        UsdShade.MaterialBindingAPI(shape.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    stage.GetRootLayer().Save()
    return dst

def fixed_frame(stage,robot_root,name):
    matches=[p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_root)) if p.GetName()==name]
    assert len(matches)==1,(name,len(matches))
    frame=matches[0];body=frame
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
    assert body
    cache=UsdGeom.XformCache();t=Gf.Transform(cache.GetLocalToWorldTransform(frame)*cache.GetLocalToWorldTransform(body).GetInverse());q=t.GetRotation().GetQuat()
    return body.GetName(),list(t.GetTranslation()),[q.GetReal(),*q.GetImaginary()]
