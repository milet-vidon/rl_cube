"""Native PhysX port of kevinzakka/mujoco_cube; external contact-only bench test.
No cube motor, cube pose writes, face-membership rewiring, or cube torque injection.
Core is fixed. Kinematic external four-wall fixtures turn +Z then +X.
"""
import argparse, json, math, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--no-contact', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--steps', type=int, default=1440)
parser.add_argument('--hz', type=int, default=240)
parser.add_argument('--iterations', type=int, default=32)
parser.add_argument('--solver', choices=['TGS','PGS'], default='TGS')
parser.add_argument('--tag', default='contact')
args = parser.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'sync_loads': True, 'width': 640, 'height': 480,
                     'extra_args': ['--/app/asyncRendering=false']},
                    experience=r'E:\IsaacLab\repo\apps\isaaclab.python.headless.kit')
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade, PhysicsSchemaTools
import omni.usd, omni.physx, carb

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
scene = UsdPhysics.Scene.Define(stage, '/World/Physics')
scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
scene.CreateGravityMagnitudeAttr(9.81)
ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
ps.CreateEnableGPUDynamicsAttr(args.gpu)
ps.CreateBroadphaseTypeAttr('GPU' if args.gpu else 'MBP')
ps.CreateSolverTypeAttr(args.solver)
ps.CreateTimeStepsPerSecondAttr(args.hz)
carb.settings.get_settings().set_bool('/physics/updateToUsd', True)

mat = UsdShade.Material.Define(stage, '/World/Material')
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(0.0); pm.CreateDynamicFrictionAttr(0.0)
pm.CreateRestitutionAttr(0.0)

def collider(prim):
    if not args.no_contact or not str(prim.GetPath()).startswith('/World/Fixture'):
        UsdPhysics.CollisionAPI.Apply(prim)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateContactOffsetAttr(0.0001)
    pc.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')

def body(path, pos=(0,0,0), mass=0.0685/27, kinematic=False):
    x = UsdGeom.Xform.Define(stage,path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddOrientOp().Set(Gf.Quatf(1))
    rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    rb.CreateKinematicEnabledAttr(kinematic)
    ma = UsdPhysics.MassAPI.Apply(x.GetPrim()); ma.CreateMassAttr(mass)
    ph = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
    ph.CreateSolverPositionIterationCountAttr(args.iterations)
    ph.CreateSolverVelocityIterationCountAttr(4)
    ph.CreateSleepThresholdAttr(0)
    ph.CreateAngularDampingAttr(0.2)
    return x

body('/World/Core',kinematic=True)
src = ET.parse(ROOT/'upstream/mujoco_cube-main/cube_3x3x3.xml').getroot()
meshes = {m.get('name'):m for m in src.findall('asset/mesh')}
names=[]; initial={}; local_centers={}
for b in src.findall('worldbody/body/body'):
    name=b.get('name'); names.append(name)
    path='/World/Cube/'+name
    UsdGeom.Xform.Define(stage,'/World/Cube')
    x=body(path)
    g=b.find('geom'); mesh=meshes[g.get('mesh')]
    center=np.fromstring(g.get('pos','0 0 0'),sep=' ')
    local_centers[name]=center
    v=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3)+center
    f=np.fromstring(mesh.get('face'),sep=' ',dtype=int).reshape(-1,3)
    shape=UsdGeom.Mesh.Define(stage,path+'/Mesh')
    shape.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
    shape.CreateFaceVertexCountsAttr([3]*len(f)); shape.CreateFaceVertexIndicesAttr(f.flatten().tolist())
    shape.CreateSubdivisionSchemeAttr('none')
    collider(shape.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    # Preserve original UV atlas on the visual mesh.
    uv=np.fromstring(mesh.get('texcoord'),sep=' ').reshape(-1,2)
    UsdGeom.PrimvarsAPI(shape).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*p) for p in uv])
    joint=b.find('joint')
    jp='/World/Joints/'+name
    if joint.get('type')=='hinge':
        j=UsdPhysics.RevoluteJoint.Define(stage,jp)
        axis=np.fromstring(joint.get('axis'),sep=' ')
        j.CreateAxisAttr('XYZ'[int(np.argmax(np.abs(axis)))])
    else:
        j=UsdPhysics.SphericalJoint.Define(stage,jp)
    j.CreateBody0Rel().SetTargets(['/World/Core'])
    j.CreateBody1Rel().SetTargets([path])
    j.CreateLocalPos0Attr(Gf.Vec3f(0)); j.CreateLocalPos1Attr(Gf.Vec3f(0))
    j.CreateCollisionEnabledAttr(False)
    initial[name]=center.tolist()

# Same sticker atlas as upstream.
visual=UsdShade.Material.Define(stage,'/World/Stickers')
shader=UsdShade.Shader.Define(stage,'/World/Stickers/Preview')
shader.CreateIdAttr('UsdPreviewSurface')
shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(0.7)
texture=UsdShade.Shader.Define(stage,'/World/Stickers/Texture')
texture.CreateIdAttr('UsdUVTexture')
texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('upstream/mujoco_cube-main/assets/sticker.png')
reader=UsdShade.Shader.Define(stage,'/World/Stickers/UV')
reader.CreateIdAttr('UsdPrimvarReader_float2'); reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
for name in names:
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/World/Cube/'+name+'/Mesh')).Bind(visual)

if not args.no_contact and not args.gpu:
    stage.GetRootLayer().Export(str(ROOT/'cube_model.usda'))

fixtures=[]
for k in range(4):
    x=body('/World/Fixture'+str(k),pos=(0,0,1),kinematic=True)
    c=UsdGeom.Cube.Define(stage,str(x.GetPath())+'/Collider')
    c.CreateSizeAttr(1)
    c.AddScaleOp().Set(Gf.Vec3f(0.003,0.062,0.017))
    c.CreateDisplayColorAttr([Gf.Vec3f(0.4,0.4,0.4)])
    collider(c.GetPrim()); fixtures.append(x)

if not args.no_contact and not args.gpu:
    stage.GetRootLayer().Export(str(ROOT/'cube_contact_scene.usda'))
phys=omni.physx.get_physx_interface()
si=omni.physx.get_physx_simulation_interface()
contact_count=[0]
fixture_contacts=[0]
def on_contact(headers,data):
    contact_count[0]+=len(headers)
    for h in headers:
        if any('Fixture' in str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0,h.actor1)):
            fixture_contacts[0]+=1
for name in names:
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath('/World/Cube/'+name)).CreateThresholdAttr(0)
sub=si.subscribe_contact_report_events(on_contact)
phys.start_simulation()
dt=1/args.hz
t=0
def step():
    global t
    si.simulate(dt,t); si.fetch_results()
    phys.update_transformations(False,True,True,False)
    t+=dt

def poses():
    cache=UsdGeom.XformCache()
    return {n:np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Cube/'+n)).Transform(Gf.Vec3d(*local_centers[n]))) for n in names}

def orientations():
    cache=UsdGeom.XformCache()
    return {n:cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Cube/'+n)).ExtractRotationQuat().GetNormalized() for n in names}

def quaternion_list(q):
    return [float(q.GetReal()),*map(float,q.GetImaginary())]

def fixture(axis,angle,withdraw=False,distance=None):
    # +Z face, or +X face: local face coordinates mapped to world.
    base=Gf.Rotation(Gf.Vec3d(0,1,0),90) if axis=='X' else Gf.Rotation(Gf.Vec3d(0,0,1),0)
    turn=Gf.Rotation(Gf.Vec3d(0,0,1),angle)
    for k,x in enumerate(fixtures):
        r=Gf.Rotation(Gf.Vec3d(0,0,1),90*k)
        p=Gf.Vec3d(0.0301,0,distance if distance is not None else (0.019 if not withdraw else 0.12))
        p=base.TransformDir(turn.TransformDir(r.TransformDir(p)))
        q=(r*turn*base).GetQuat().GetNormalized()
        ops=x.GetOrderedXformOps()
        ops[0].Set(p); ops[1].Set(Gf.Quatf(q))

def rotation(axis,deg):
    v=Gf.Vec3d(1,0,0) if axis=='X' else Gf.Vec3d(0,0,1)
    return Gf.Rotation(v,deg)

results=[]; frames=[]
for phase,axis in enumerate(['Z','X']):
    before=poses()
    before_q=orientations()
    idx=2 if axis=='Z' else 0
    selected=[n for n,p in before.items() if p[idx]>0.012]
    count_start=contact_count[0]
    fixture_start=fixture_contacts[0]
    fixture(axis,0,True)
    for _ in range(round(args.hz/8)): step()
    for i in range(args.hz):
        fixture(axis,0,distance=0.12+(0.019-0.12)*(i+1)/args.hz)
        step()
    for _ in range(round(args.hz/4)): step()
    for i in range(args.steps):
        progress=(i+1)/args.steps
        angle=90*(3*progress**2-2*progress**3)
        fixture(axis,angle); step()
        if i%round(args.hz/10)==0:
            frames.append(dict(time=t,phase=axis,positions={n:p.tolist() for n,p in poses().items()},quaternions={n:quaternion_list(q) for n,q in orientations().items()}))
    for _ in range(round(args.hz/2)): step()
    after=poses()
    target={n:np.array(rotation(axis,90).TransformDir(Gf.Vec3d(*p))) if n in selected else p for n,p in before.items()}
    errors={n:float(np.linalg.norm(after[n]-target[n])) for n in names}
    after_q=orientations()
    orientation_errors={}
    for n in names:
        expected=rotation(axis,90).GetQuat()*before_q[n] if n in selected else before_q[n]
        delta=(after_q[n]*expected.GetInverse()).GetNormalized()
        orientation_errors[n]=math.degrees(2*math.acos(min(1,abs(delta.GetReal()))))
    # Signed angle of selected off-axis cubies, excluding the center cubelet.
    a,b=(0,1) if axis=='Z' else (1,2)
    angles=[]
    for n in selected:
        p=before[n]; q=after[n]
        if math.hypot(p[a],p[b])>.01:
            angles.append(math.degrees(math.atan2(p[a]*q[b]-p[b]*q[a],p[a]*q[a]+p[b]*q[b])))
    result=dict(axis=axis,selected=selected,mean_angle_deg=float(np.mean(angles)),max_position_error_m=max(errors.values()),max_orientation_error_deg=max(orientation_errors.values()),fixture_contact_headers=fixture_contacts[0]-fixture_start,contact_headers=contact_count[0]-count_start,errors_m=errors,orientation_errors_deg=orientation_errors)
    for i in range(args.hz):
        fixture(axis,90,distance=0.019+(0.12-0.019)*(i+1)/args.hz)
        step()
    for _ in range(round(args.hz/4)): step()
    released=poses()
    result['max_released_position_error_m']=max(float(np.linalg.norm(released[n]-target[n])) for n in names)
    result['passed']=len(selected)==9 and abs(result['mean_angle_deg']-90)<5 and max(errors.values())<0.002 and max(orientation_errors.values())<5 and result['max_released_position_error_m']<0.002 and result['fixture_contact_headers']>0
    results.append(result)
    print('PHASE_RESULT',json.dumps(result),flush=True)
    if not result['passed'] and not args.no_contact:
        break  # A corrupted first face is not a valid start for a consecutive-face test.

out=dict(engine='Isaac Sim 5.1 PhysX',gpu=args.gpu,hz=args.hz,solver=args.solver,iterations=args.iterations,core_fixed=True,gravity=9.81,collision_disabled_control=args.no_contact,cube_drive_count=0,physics_steps=round(t/dt),results=results,passed=len(results)==2 and all(r['passed'] for r in results),frames=frames)
(ROOT/(args.tag+'_result.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
phys.reset_simulation()
app.close()
