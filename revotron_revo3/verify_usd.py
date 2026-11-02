from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
from isaacsim import SimulationApp
app=SimulationApp({'headless':True},experience=r'E:\IsaacLab\repo\apps\isaaclab.python.headless.kit')
from pxr import Usd,UsdPhysics,UsdGeom,UsdLux,Gf,Sdf
path=ROOT/'source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd'
s=Usd.Stage.Open(str(path))
prims=list(s.Traverse())
joints=[p.GetName() for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
expected=json.loads((ROOT/'joint_mapping_56.json').read_text())['joint_names']
assert len(joints)==56 and set(joints)==set(expected),(len(joints),set(expected)-set(joints))
data={'revolute_joints':len(joints),'joint_names_match':True,'articulation_roots':[str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)],'meshes':sum(p.IsA(UsdGeom.Mesh) for p in prims),'validation':'USD structure and references; no robot physics rollout or control validation'}
assert data['articulation_roots']
(ROOT/'usd_audit.json').write_text(json.dumps(data,indent=2))
view=Usd.Stage.CreateNew(str(ROOT/'view_scene.usda'))
UsdGeom.SetStageUpAxis(view,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(view,1)
root=UsdGeom.Xform.Define(view,'/World');view.SetDefaultPrim(root.GetPrim())
robot=UsdGeom.Xform.Define(view,'/World/Robot')
robot.GetPrim().GetReferences().AddReference('source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd')
light=UsdLux.DomeLight.Define(view,'/World/Light');light.CreateIntensityAttr(800)
sun=UsdLux.DistantLight.Define(view,'/World/Sun');sun.CreateIntensityAttr(1800)
sun.AddRotateXYZOp().Set(Gf.Vec3f(-35,25,-30))
cam=UsdGeom.Camera.Define(view,'/World/OverviewCamera')
cam.CreateFocalLengthAttr(35)
eye=Gf.Vec3d(3,-3,1.4);target=Gf.Vec3d(0,0,-.3)
matrix=Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse()
cam.AddTransformOp().Set(matrix)
view.GetRootLayer().Save()
print('USD_AUDIT',json.dumps(data),flush=True)
app.close()
