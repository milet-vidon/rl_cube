"""Opt-in source-derived Core inertia; no forces, drives, collision or pose edits."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,math,xml.etree.ElementTree as ET

def source_core_parameters(path,usd_mass):
    path=Path(path);root=ET.parse(path).getroot();body=root.find('./worldbody/body[@name="core"]')
    if body is None or body.find('inertial') is not None:raise ValueError('Expected source core inertia inferred from one sphere')
    geoms=body.findall('geom')
    if len(geoms)!=1:raise ValueError('Expected one source core geom')
    geom=geoms[0];defaults=root.find('./default/default[@class="core"]/geom')
    if defaults is None or geom.get('class')!='core':raise ValueError('Unsupported source core defaults')
    attrs=defaults.attrib|geom.attrib
    if attrs.get('type','sphere')!='sphere' or 'mesh' in attrs:raise ValueError('Source core is not an analytic sphere')
    if attrs.get('pos','0 0 0').split()!=['0','0','0']:raise ValueError('Source core sphere is not centered')
    size=[float(v) for v in attrs['size'].split()];radius=size[0];source_mass=float(attrs['mass']);mass=float(usd_mass)
    if not all(math.isfinite(v) and v>0 for v in (radius,mass,source_mass)):raise ValueError('Nonpositive or nonfinite core mass/radius')
    if abs(source_mass-mass)>source_mass*1e-4:raise ValueError('USD/source core masses differ')
    inertia=.4*mass*radius**2
    return {'source_mjcf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_core_mass_kg':source_mass,'runtime_core_mass_kg':mass,'sphere_radius_m':radius,'diagonal_inertia_kg_m2':[inertia]*3,'basis':'I=2/5*m*r^2; runtime mass unchanged; centered source sphere','scope':'core inertia only; source joint armature/damping/frictionloss not migrated'}

def apply_source_core_inertia(stage,cube_path):
    from pxr import UsdPhysics,Gf
    prim=stage.GetPrimAtPath('/World/Core');api=UsdPhysics.MassAPI(prim)
    source=Path(cube_path).parent/'upstream/mujoco_cube-main/cube_3x3x3.xml'
    report=source_core_parameters(source,api.GetMassAttr().Get())
    report['previous_authored_diagonal_inertia']=list(api.GetDiagonalInertiaAttr().Get())
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*report['diagonal_inertia_kg_m2']))
    api.CreateCenterOfMassAttr(Gf.Vec3f(0.,0.,0.));api.CreatePrincipalAxesAttr(Gf.Quatf(1.,0.,0.,0.))
    print('SOURCE_CORE_INERTIA',report['diagonal_inertia_kg_m2'],flush=True)
    return report

@contextmanager
def source_core_inertia(enabled,module=None):
    if not enabled:
        yield
        return
    if module is None:
        import articulated_cube_env as module
    original=module.DynamicCubeHandle
    def build(stage,path,*args,**kwargs):
        handle=original(stage,path,*args,**kwargs)
        handle.model['core_inertia_override']=apply_source_core_inertia(stage,path)
        return handle
    module.DynamicCubeHandle=build
    try:yield
    finally:module.DynamicCubeHandle=original

def verify_core_inertia(env):
    from isaac_compat import tensor
    expected=env.cube.model['core_inertia_override']['diagonal_inertia_kg_m2']
    matrix=tensor(env.cube.core.data.body_inertia).reshape(-1,9)[0].tolist()
    for i,want in zip((0,4,8),expected):
        if abs(matrix[i]-want)>1e-5*want:raise ValueError('Core inertia override did not reach PhysX')
    if max(abs(matrix[i]) for i in (1,2,3,5,6,7))>max(expected)*1e-5:raise ValueError('Unexpected off-diagonal core inertia')
    return {'expected_diagonal_inertia_kg_m2':expected,'actual_inertia_kg_m2':matrix,'verified':True}
