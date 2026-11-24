"""90 mm physical scene, calibrated GPU reset IK and no-finger-count teacher.

This module is only used by the new training lineage. Specializations compile
new functions with private globals and verify the original module hashes; the
57 mm modules, caches and assets are never patched, even in this process.
The digit-level impulses still drive fingertip feedback. Only aggregate real
left-hand/cube support gates grasp progression, retention and completion.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import inspect
import json
import textwrap
import numpy as np

ROOT=Path(__file__).resolve().parent
EDGE=.090
HALF=.045
RESET_Z=.047
CONTACT_IMPULSE=1e-5
ASSET_SHA='10d8aacdb44dca20311478b40905f05914c4646391020e2c90f7437c78d83807'
GRASP_SHA='a7be0f7f3c68da937a2ad08a52c1ed947e3d8a8302fcbbd37c700505d444cb7b'
SOURCE_SHA={
    'contact_hold_cuda_scene.py':'bae964c1f91c257159011b2597444eaa15e4165be696802bdb5f70e18ce91a13',
    'contact_hold_cuda_teacher.py':'4fa87aab69f1804e47adde3b4630338cbe4b1b32f4088b47a34113df20ec9168',
    'contact_hold_cuda_ik.py':'4f48fe8bab302febdcf9f7ad15b96352297f4329c111bdc1573564ce7588bfff'}
FAILURES90={0:'',1:'lost_primary_support',2:'contact_or_settle_timeout',3:'primary_support_lost_during_tilt',
            4:'tilt_hold_timeout',5:'planner_rejected: post-lift IK, limits or time envelope',
            6:'planner_rejected: Post-turn correction exceeded bounded pose envelope',
            7:'planner_rejected: post-turn IK violates wrist pose envelope'}

def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _read(p):return json.loads(Path(p).read_bytes())
def _require(b,msg):
    if not b:raise ValueError(msg)

def _source(name):
    _require(_sha(ROOT/name)==SOURCE_SHA[name], 'GPU domain dependency changed: '+name)

def left_supported(hand_impulse):
    """Raw per-hand impulses from the cube sensors, including palm and wrist."""
    if hand_impulse.ndim!=2 or hand_impulse.shape[1]!=2:raise ValueError('Expected actual [N,2] hand/cube impulses')
    return hand_impulse[:,0]>CONTACT_IMPULSE

def height_above_table(center_z,table_z):return center_z-(table_z+HALF)

def _grasp(path):
    _require(_sha(path)==GRASP_SHA, 'Expected the calibrated 90 mm grasp')
    g=_read(path)
    _require(g['kind']=='90mm_left_grasp_retarget_kinematic' and g['main_hand']=='left'
             and set(g['contact_anchors'])=={'thumb','index','middle','ring','little'}
             and g['arm_settle_tolerance_rad']==.02,'Wrong grasp profile')
    return g

def verify_domain(scene_source,grasp):
    """Verify geometry/calibration only; never inspect demonstrations or rollout rows."""
    from contact_hold_size_geometry import verify_asset
    src=Path(scene_source).resolve()
    manifest=_read(src/'artifact_manifest.json')['sha256']
    for name in ('dynamic_training_scene.usda','cube_asset.json','size_geometry.json','size_runtime.json',
                 'retarget_grasp90.json','retarget_geometry.json','size_cube_model.usda','config.json','asset_contract.json','runtime_robot_asset.json'):
        _require(manifest.get(name)==_sha(src/name), 'Changed archived 90 mm scene dependency: '+name)
    _grasp(grasp)
    rt=_read(src/'size_runtime.json');domain=_read(src/'size_geometry.json')
    _require(domain['edge_m']==EDGE and domain['asset_sha256']==ASSET_SHA and _sha(src/'size_cube_model.usda')==ASSET_SHA,
             'Wrong collider geometry')
    verify_asset(src/'size_cube_model.usda',domain)
    _require(rt['verified'] and rt['edge_m']==EDGE and len(rt['body_properties'])==27,'Missing measured 90 mm reference')
    _require(_read(src/'cube_asset.json')['source_sha256']==ASSET_SHA and _sha(src/'retarget_grasp90.json')==GRASP_SHA and
             _read(src/'retarget_geometry.json')['grasp_sha256']==GRASP_SHA,'Mixed assets or grasps')
    cfg=_read(src/'config.json')
    _require(cfg['success']['contact_impulse_min']==CONTACT_IMPULSE,'Changed support threshold')
    return dict(edge_m=EDGE, cube_asset_sha256=ASSET_SHA, grasp_sha256=GRASP_SHA, scene_sha256=manifest['dynamic_training_scene.usda'],
                scene_manifest_sha256=_sha(src/'artifact_manifest.json'),scene_source=str(src),source_sha256=SOURCE_SHA,
                reset_clearance_m=.002,loss_grace_frames=20,hand_contact='raw left-hand/cube impulse; includes palm, no digit count')

def _specialize(fn, changes, new_globals):
    orig=inspect.unwrap(fn)
    s=textwrap.dedent(inspect.getsource(orig))
    for old,new in changes:
        _require(s.count(old)==1, 'Unexpected pinned implementation: '+repr(old))
        s=s.replace(old,new)
    ns={**orig.__globals__,**new_globals};exec(compile(s,'<90mm:'+orig.__qualname__+'>','exec'),ns)
    return ns[orig.__name__]

def _mass_readback(scene,source):
    """Read all native lanes, detecting an incorrectly cloned mass distribution."""
    ref=_read(Path(source)/'size_runtime.json')['body_properties']
    _require(len(scene.views)==len(ref)==27,'Wrong number of replicated bodies')
    maxerr=0.; maxcom=0.
    for n,view in scene.views.items():
        p='/World/Core' if n=='Core' else '/World/Cube/'+n
        b=ref[p]
        def arr(v):
            if hasattr(v,'detach'):v=v.detach().cpu().numpy()
            return np.asarray(v)
        m=arr(view.get_masses()).reshape(scene.n)
        com=arr(view.get_coms()).reshape(scene.n,7)[:,:3]
        iner=np.linalg.eigvalsh(arr(view.get_inertias()).reshape(scene.n,3,3))
        expected=np.array(b['principal_inertia_kg_m2'])
        _require(np.allclose(m,b['mass_kg'],rtol=2e-5,atol=0) and np.allclose(com,b['local_com_m'],rtol=1e-4,atol=2e-6)
                 and np.allclose(iner,expected,rtol=2e-3,atol=0), 'GPU 90mm mass / inertia / COM mismatch: '+p)
        maxerr=max(maxerr,float(np.max(abs(iner/expected-1))));maxcom=max(maxcom,float(np.max(abs(com-b['local_com_m']))))
    return dict(verified=True,body_count=27,lanes=scene.n,max_relative_inertia_error=maxerr,max_com_error_m=maxcom)

# The old home puts the wrist housings in the swept volume of the largest cube.
# Translate both complete wrists before placing the cube, preserving orientation.
PALM_BACK_M=.020

def reset_config(config):
    from kinematics import RobotKinematics
    robot=RobotKinematics()
    initial=config['home']
    q=initial.copy()
    for side in ('left','right'):
        link=side+'_palm_link'
        start=robot.fk_link(initial,link)
        goal=start[:3,3]+np.array([-PALM_BACK_M,0.,0.])
        q=robot.fit_arm(q,side,goal,start[:3,:3])
        pose=robot.fk_link(q,link)
        _require(np.max(abs(pose[:3,3]-goal))<1e-4 and np.max(abs(pose[:3,:3]-start[:3,:3]))<1e-4,
                 'Could not initialize a separated wrist pose')
    return dict(config,home=q)

def _check_reset(s, poses, seeds):
    import torch
    world=s.views['Core'].get_transforms()[:,:3]
    delta=world-poses[:,:3]
    if not delta.isfinite().all():raise ValueError('Nonfinite physical 90mm reset pose')
    h=s.hand_impulse
    if not h.isfinite().all():raise ValueError('Nonfinite hand impulse at reset')
    bad=(delta.abs().amax(1)>.015)|(h>0).any(1)
    if bad.any():
        lane=int(torch.where(bad)[0][0])
        raise ValueError(f'90mm reset invalid in {int(bad.sum())} lanes; first lane {lane} seed {seeds[lane]} '
                         f'xyz error m={delta[lane].tolist()} L/R wrist or hand impulse Ns={h[lane].tolist()}')
    return {'max_xyz_error_m':delta.abs().amax(0).tolist(),'max_left_right_impulse_Ns':h.amax(0).tolist(),
            'lanes':len(seeds),'tolerance_m':.015}

def open_scene(scene_source,n,grasp):
    receipt=verify_domain(scene_source,grasp); _source('contact_hold_cuda_scene.py')
    from contact_hold_cuda_scene import Scene
    reset=_specialize(Scene.reset,[("self.cfg['table_z']+.0305","self.cfg['table_z']+RESET_Z"),
        ("if (xyz-v[:,:3]+self.offset).abs().max()>.015:raise ValueError('Reset poses do not match lane offsets')",
         "self.domain_receipt['reset_measurements']=check_reset(self,v,seeds)")],{'RESET_Z':RESET_Z,'check_reset':_check_reset})
    subclass=type('Physical90Scene',(Scene,),{'reset':reset})
    scene=subclass(Path(scene_source),n)
    scene.cfg=reset_config(scene.cfg)
    scene.home=scene.home.new_tensor([scene.cfg['home'][n] for n in scene.names]).expand(n,-1).contiguous()
    receipt['reset_palm_translation_m']={side:[-PALM_BACK_M,0.,0.] for side in ('left','right')}
    receipt['replicated_mass_properties']=_mass_readback(scene,scene_source)
    scene.domain_receipt=receipt
    print('GPU_90MM_DOMAIN',json.dumps(receipt),flush=True)
    return scene

@lru_cache(2)
def _ik_builder(grasp_path):
    _source('contact_hold_cuda_ik.py')
    from contact_hold_cuda_ik import plan_reset,constants
    # Strip the original LRU decorator: cache the new calibration *separately*.
    unwrapped=_specialize(constants,[('@lru_cache(1)',''),("Path(__file__).with_name('opposition_grasp_v20.json')",'profile_path'),
                            ('g,c=p.grasp,p.cfg','g,c=p.grasp,reset_config(p.cfg)')],
                           {'profile_path':Path(grasp_path),'reset_config':reset_config})
    c=unwrapped()
    return _specialize(plan_reset,[('model,fingers,ai,lim,hz=constants()','model,fingers,ai,lim,hz=physical_constants')],{'physical_constants':c})

def make_plan(core,grasp):
    _grasp(grasp)
    return _ik_builder(str(Path(grasp).resolve()))(core)

@lru_cache(2)
def _teacher_class(grasp_path):
    _source('contact_hold_cuda_teacher.py')
    from contact_hold_cuda_teacher import BatchedTeacher
    globals90=dict(left_supported=left_supported,HALF=HALF,profile_path=Path(grasp_path))
    init=_specialize(BatchedTeacher.__init__,[
        ("Path(__file__).with_name('opposition_grasp_v20.json')",'profile_path'),
        ('self.accepted=torch.zeros_like(self.phase,dtype=torch.bool)',
         'self.accepted=torch.zeros_like(self.phase,dtype=torch.bool)\n    self.ever_touched=self.accepted.clone()')],globals90)
    turn=_specialize(BatchedTeacher._turn_step,[
        ('vel,goal,display):','vel,goal,display,hand_impulse):'),
        ('touches=contacts[ii,:5]>1e-5;retained=touches[:,0]&(touches[:,1:].sum(1)>=3)','retained=left_supported(hand_impulse)[ii]'),
        ('&touches.all(1)&','&retained&'),("-.0285>=.08",'-HALF>=.08'),
        ('self.lost[ii]>=10','self.lost[ii]>=20')],globals90)
    action=_specialize(BatchedTeacher.action,[
        ('target,contacts,velocity):','target,contacts,velocity,hand_impulse):'),
        ('self._turn_step(late,core,contacts,velocity,goal,display)','self._turn_step(late,core,contacts,velocity,goal,display,hand_impulse)'),
        ('touches=contacts[:,:5]>1e-5; all5=touches.all(1); topology=touches[:,0]&(touches[:,1:].sum(1)>=3)',
         'support=left_supported(hand_impulse)\n    self.ever_touched |= active & ((hand_impulse>0).any(1) | (contacts[:,:10]>0).any(1))'),
        ('torch.where(all5,self.streak+1,0)','torch.where(support,self.streak+1,0)'),
        ('torch.where(topology,0,self.lost+1)','torch.where(support,0,self.lost+1)'),
        ('(self.lost>=10)','(self.lost>=20)'),("-.0285>=.08",'-HALF>=.08'),
        ('<self.tol)&(v[:,self.arm]', '<torch.where((before<4)&~self.ever_touched,.08,self.tol))&(v[:,self.arm]')],globals90)
    return type('Physical90Teacher',(BatchedTeacher,),{'__init__':init,'_turn_step':turn,'action':action})

def make_teacher(plan,cfg,candidate,grasp,gain_writer=None):
    _grasp(grasp)
    _require(cfg['success']['contact_impulse_min']==CONTACT_IMPULSE,'Unexpected support threshold')
    return _teacher_class(str(Path(grasp).resolve()))(plan,cfg,candidate,gain_writer=gain_writer)
