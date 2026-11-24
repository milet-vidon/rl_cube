"""Batched reset IK for the v20 pickup teacher, in scene-local coordinates.

The native CUDA kernels are compiled by the pinned PyTorch environment (NVRTC,
no external build). Only constant URDF transforms and grasp calibration are
prepared on the host. Each reset pose is independently solved, time-scaled,
and checked on the GPU; infeasible lanes never provide training labels.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from contact_hold_cuda_ik_kernel import SOLVE, RETIME

NAMES = ('ready','preshape','lower','approach','squeeze','primary_lift','support','contact','hold')
OFFSETS = (0,13,26,39,52,65,90,114,114)
INTERVALS = (12,12,12,12,12,24,24,0,0)

@lru_cache(1)
def constants():
    from opposition_teacher import OppositionPlanner
    p=OppositionPlanner(Path(__file__).with_name('opposition_grasp_v20.json'))
    g,c=p.grasp,p.cfg
    if not g['primary_lift_before_support']:raise ValueError('CUDA planner expects primary-first v20')
    names=c['joint_names'];arm_names=[f'Joint{i}_{s}' for s in 'LR' for i in range(1,8)]
    ai=[names.index(n) for n in arm_names]
    chain=[]
    for si,side in enumerate(('left','right')):
        link=side+'_palm_link';joints=[]
        while link in p.k.by_child:
            n,link,_,origin,axis=p.k.by_child[link];joints.append((n,origin,axis))
        t=np.eye(4);ids=[]
        for n,o,a in reversed(joints):
            t=t@o
            if a is None:continue
            if not np.array_equal(a[:2],[0.,0.]) or abs(a[2])!=1.:raise ValueError(f'Unsupported wrist axis: {n}')
            chain.extend([*t[:3,:3].ravel(),*t[:3,3],a[2]]);t=np.eye(4);ids.append(n)
        if ids!=arm_names[si*7:si*7+7]:
            # At this point only seven of this arm's eight transforms were added.
            raise ValueError('Unexpected arm chain order')
        chain.extend([*t[:3,:3].ravel(),*t[:3,3],0.])
    lim=np.array([p.k.limits[n] for n in names]);arm_lim=lim[ai]+[.002,-.002]
    home=np.array([c['home'][n] for n in names])
    base=np.array([[0.,0,1],[-1.,0,0],[0,-1.,0]])
    main=base@Rotation.from_euler('z',g['frame_roll_deg'],degrees=True).as_matrix()@np.array(g['cube_rotation_palm']).T
    model=np.r_[chain,arm_lim.ravel(),home[ai],main.ravel(),-main@np.array(g['center_palm']),
                np.array(g['support_rotation']).ravel(),g['support_center_offset'],g['lift_forward_m'],g['lift_height_m'],0.,c['robot_z']]
    f0=c['home'].copy();f1=f0|{n:v for n,v in p.fingers.items() if n.startswith('right_')}|p.primary_open
    f2=f1|p.primary_closed
    postures=np.array([[f[n] for n in names] for f in [f0,f1,f1,f1,f2,f2,f2]])
    start=home;knots=[]
    for goal,m in zip(postures,INTERVALS):
        u=np.linspace(0.,1.,m+1)[:,None]
        knots.append((1.-u)*start+u*goal);start=goal
    return model,np.concatenate(knots),ai,lim,c['control_hz']

def _slopes(x):
    d=x[:,1:]-x[:,:-1];l,r=d[:,:-1],d[:,1:]
    mid=torch.where(l*r>0,2*l*r/(l+r),0.)
    def end(a,b):
        out=(3*a-b)*.5
        return torch.where(out*a<=0,0.,torch.where((a*b<=0)&(out.abs()>3*a.abs()),3*a,out))
    return torch.cat([end(d[:,:1],d[:,1:2]),mid,end(d[:,-1:],d[:,-2:-1])],dim=1)

def _compile(source,name,arch):
    # The pip CUDA runtime includes NVRTC but need not include a full nvcc toolkit.
    from cuda.bindings import nvrtc
    def check(result):
        if int(result[0])!=0:raise RuntimeError(f'NVRTC {name}: {result[0]}')
        return result[1] if len(result)>1 else None
    prog=check(nvrtc.nvrtcCreateProgram(source.encode(),(name+'.cu').encode(),0,[],[]))
    try:
        opts=[('--gpu-architecture=sm_'+arch).encode(),b'--std=c++17']
        result=nvrtc.nvrtcCompileProgram(prog,len(opts),opts)
        if int(result[0]):
            log=bytes(check(nvrtc.nvrtcGetProgramLogSize(prog)))
            check(nvrtc.nvrtcGetProgramLog(prog,log));raise RuntimeError(log.decode(errors='replace'))
        binary=bytes(check(nvrtc.nvrtcGetCUBINSize(prog)))
        check(nvrtc.nvrtcGetCUBIN(prog,binary));return binary
    finally:check(nvrtc.nvrtcDestroyProgram(prog))

@lru_cache(4)
def kernels(dev):
    from torch.cuda._utils import _cuda_load_module
    with torch.cuda.device(dev):
        arch=''.join(map(str,torch.cuda.get_device_capability(dev)))
        return tuple(_cuda_load_module(_compile(src,name,arch),[name])[name]
                     for src,name in [(SOLVE,'reset_ik'),(RETIME,'retime')])

class ResetPlan:
    names=NAMES
    def __init__(self,knots,slopes,lengths,bad):
        self.knots=knots;self.slopes=slopes;self.lengths=lengths;self.rejection_code=bad
        self.accepted=self.valid=bad==0
        self.rows=torch.arange(len(bad),device=bad.device)
        self.offsets=torch.tensor(OFFSETS,device=bad.device)
        self.intervals=torch.tensor(INTERVALS,device=bad.device)
    def goal(self,phase,tick):
        rows=self.rows;lens=self.lengths[rows,phase]
        z=(tick.to(torch.float64)/(lens-1)).clamp(0.,1.)
        m=self.intervals[phase];t=(z.pow(3)*(10.+z*(-15.+6.*z))*m).clamp(min=0.).minimum(m)
        k=t.long().minimum((m-1).clamp_min(0));a=self.offsets[phase]+k
        b=a+(m>0).long();u=(t-k).float().unsqueeze(1)
        out=(1-u).square()*((1+2*u)*self.knots[rows,a]+u*self.slopes[rows,a])+u.square()*((3-2*u)*self.knots[rows,b]+(u-1)*self.slopes[rows,b])
        return out,lens
    def at(self,phase,tick):return self.goal(phase,tick)[0]
    def path_for_lane(self,j):
        from scipy.interpolate import PchipInterpolator
        if not self.valid[j].item():raise ValueError(f'lane {j} failed CUDA IK')
        pts=self.knots[j].double().cpu().numpy();paths=[]
        for name,off,m,n in zip(NAMES,OFFSETS,INTERVALS,self.lengths[j].tolist()):
            if m:
                u=np.linspace(0,1,n);q=PchipInterpolator(np.arange(m+1),pts[off:off+m+1])(np.clip(u**3*(10.+u*(-15.+6.*u))*m,0,m))
            else:q=np.repeat(pts[off:off+1],n,axis=0)
            paths.append((name,q.astype(np.float32)))
        return paths

@torch.no_grad()
def plan_reset(core):
    """Return device-resident targets and per-lane rejection codes; native XYZW."""
    if core.ndim!=2 or core.shape[1]<7 or not core.is_cuda or not len(core):
        raise ValueError('plan_reset requires nonempty CUDA native XYZW reset states')
    n=len(core);dev=core.device
    model,fingers,ai,lim,hz=constants()
    def db(x):return torch.as_tensor(x,device=dev,dtype=torch.float64)
    # Match scipy Rotation.from_quat and as_euler('xyz') for a settled cube.
    raw=core[:,3:7].double();norm=raw.norm(dim=1,keepdim=True)
    finite=core[:,:7].isfinite().all(1)&(norm[:,0]>1.e-10)
    x,y,z,w=(raw/norm.clamp_min(1.e-12)).unbind(1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pose=torch.cat([core[:,:3].double(),yaw[:,None]],dim=1).nan_to_num().contiguous()
    arm=torch.zeros((n,115,14),device=dev,dtype=torch.float64)
    bad=torch.zeros(n,dtype=torch.int32,device=dev)
    lengths=torch.full((n,9),2,device=dev,dtype=torch.long);lengths[:,7]=13;lengths[:,8]=180
    solve,retime=kernels(str(dev))
    jointlim=db(lim);vel=db([.65 if j in ai else 1.5 for j in range(56)])
    acc=db([2. if j in ai else 6. for j in range(56)])
    solve(grid=((n+31)//32,1,1),block=(32,1,1),args=[pose,db(model),arm,bad,n])
    bad.masked_fill_(~finite,-1)
    pts=torch.empty((n,115,56),device=dev);slopes=torch.empty_like(pts)
    ids=torch.tensor(ai,device=dev);f=db(fingers)
    # Bound temporary fp64 interpolation storage independently of env count.
    for i in range(0,n,128):
        stop=min(i+128,n);p=f.expand(stop-i,-1,-1).clone()
        p[:,:,ids]=arm[i:stop];s=torch.empty_like(p)
        for off,m in zip(OFFSETS[:7],INTERVALS[:7]):s[:,off:off+m+1]=_slopes(p[:,off:off+m+1])
        retime(grid=(stop-i,7,1),block=(64,1,1),args=[p,s,jointlim,vel,acc,lengths[i:stop],bad[i:stop],int(hz),stop-i])
        pts[i:stop]=p;slopes[i:stop]=s
    return ResetPlan(pts,slopes,lengths,bad)
