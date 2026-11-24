"""Opt-in linear passive relative damping approximation from the source MJCF, with zero actuation targets.

Ordinary USD spherical joints ignore drives. Replace only unlimited spherical
joints by D6 joints with locked translation and free rotation before PhysX reset.
Only the linear coefficient is applied. MJCF damping triples are linear,
quadratic and cubic velocity coefficients, not xyz axes. Higher-order damping,
armature and Coulomb friction are not migrated. No cube turn is commanded.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib, math, xml.etree.ElementTree as ET

def source_joint_parameters(path):
    path=Path(path); root=ET.parse(path).getroot()
    default=root.find('./default/default[@class="cubelet"]/joint')
    core=root.find('./worldbody/body[@name="core"]')
    if default is None or core is None or core.get('childclass')!='cubelet':
        raise ValueError('Unsupported source cube joint inheritance')
    result={}
    for body in core.findall('body'):
        joints=body.findall('joint')
        if len(joints)!=1:raise ValueError('Expected one joint per cubelet')
        attrs=default.attrib|joints[0].attrib
        if attrs.get('type') not in ('hinge','ball'):raise ValueError('Unsupported source joint type')
        values=[float(v) for v in attrs['damping'].split()]
        if len(values) not in (1,3) or any(not math.isfinite(v) or v<0 for v in values) or values[0]<=0:
            raise ValueError('Expected positive linear and nonnegative finite polynomial damping')
        coefficients=(values+[0.,0.])[:3]
        name=body.get('name')
        if name in result or attrs['name']!=name:raise ValueError('Unexpected source joint naming')
        result[name]={'type':attrs['type'],'source_polynomial_coefficients_abc':coefficients,'omitted_quadratic_and_cubic_coefficients':coefficients[1:],'damping_Nm_per_rad_s':values[0],
                      'usd_damping_Nm_per_deg_s':values[0]*math.pi/180}
    if sum(j['type']=='hinge' for j in result.values())!=6 or sum(j['type']=='ball' for j in result.values())!=20:
        raise ValueError('Expected 6 hinge and 20 ball joints')
    return {'source_mjcf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_damping_interpretation':'linear_quadratic_cubic_velocity_coefficients','joints':result}

def apply_source_joint_damping(stage,cube_path):
    from pxr import UsdPhysics
    source=Path(cube_path).parent/'upstream/mujoco_cube-main/cube_3x3x3.xml'
    report=source_joint_parameters(source); selected=[]; names=set()
    # Validate the complete migration before changing any prim.
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim);bodies=joint.GetBody1Rel().GetTargets()
        if not any(str(b).startswith('/World/Cube/') for b in bodies):continue
        if len(bodies)!=1 or list(map(str,joint.GetBody0Rel().GetTargets()))!=['/World/Core']:
            raise ValueError('Unexpected cube joint connectivity')
        name=bodies[0].name
        if name in names or name not in report['joints']:raise ValueError('Unexpected cubelet joint')
        source_type=report['joints'][name]['type']
        if not prim.IsA(UsdPhysics.RevoluteJoint if source_type=='hinge' else UsdPhysics.SphericalJoint):
            raise ValueError('Source/USD joint type mismatch')
        if UsdPhysics.DriveAPI.GetAll(prim) or UsdPhysics.LimitAPI.GetAll(prim):
            raise ValueError('Existing cube drives or D6 limits require a separate migration')
        if source_type=='ball':
            sphere=UsdPhysics.SphericalJoint(prim)
            if sphere.GetConeAngle0LimitAttr().Get()>=0 or sphere.GetConeAngle1LimitAttr().Get()>=0:
                raise ValueError('Cannot replace a limited spherical joint')
        selected.append((prim,name,source_type));names.add(name)
    if names!=set(report['joints']):raise ValueError('Cube/source joint inventory mismatch')
    count=0
    for prim,name,source_type in selected:
        if source_type=='ball':
            prim.SetTypeName('PhysicsJoint')
            for axis in ('transX','transY','transZ'):
                limit=UsdPhysics.LimitAPI.Apply(prim,axis)
                limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
            axes=('rotX','rotY','rotZ')
        else:axes=('angular',)
        for axis in axes:
            drive=UsdPhysics.DriveAPI.Apply(prim,axis)
            drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(report['joints'][name]['usd_damping_Nm_per_deg_s'])
            drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
            count+=1
        report['joints'][name].update(usd_path=str(prim.GetPath()),drive_axes=list(axes))
    report.update(converted_spherical_joints=20,passive_axes=count,
                  scope='linear passive relative damping only; zero stiffness and velocity targets; nonlinear damping/armature/frictionloss not migrated')
    verify_passive_joint_damping(stage,report)
    print('SOURCE_JOINT_DAMPING',count,'axes',flush=True)
    return report

def verify_passive_joint_damping(stage,report):
    from pxr import UsdPhysics
    for spec in report['joints'].values():
        prim=stage.GetPrimAtPath(spec['usd_path'])
        for axis in spec['drive_axes']:
            drive=UsdPhysics.DriveAPI(prim,axis)
            if drive.GetTypeAttr().Get()!='force' or drive.GetStiffnessAttr().Get()!=0 or drive.GetTargetVelocityAttr().Get()!=0 or drive.GetTargetPositionAttr().Get()!=0:
                raise ValueError('Unexpected active cube joint drive')
            expected=spec['usd_damping_Nm_per_deg_s']
            if abs(drive.GetDampingAttr().Get()-expected)>expected*1e-5:
                raise ValueError('Passive damping coefficient mismatch')
        if spec['type']=='ball':
            for axis in ('transX','transY','transZ'):
                limit=UsdPhysics.LimitAPI(prim,axis)
                if not limit.GetLowAttr().Get()>limit.GetHighAttr().Get():raise ValueError('D6 translation is not locked')
            if any(UsdPhysics.LimitAPI(prim,a) for a in ('rotX','rotY','rotZ')):
                raise ValueError('D6 rotation unexpectedly limited')
    return {'verified_authored_parameters':True,'passive_axes':report['passive_axes'],
            'physics_effect_requires_simulation_validation':True}

@contextmanager
def source_joint_damping(enabled,module=None):
    if not enabled:
        yield
        return
    if module is None:
        import articulated_cube_env as module
    original=module.DynamicCubeHandle
    def build(stage,path,*args,**kwargs):
        handle=original(stage,path,*args,**kwargs)
        handle.model['passive_joint_damping']=apply_source_joint_damping(stage,path)
        return handle
    module.DynamicCubeHandle=build
    try:yield
    finally:module.DynamicCubeHandle=original
