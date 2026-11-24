"""Conservative forearm/torso routing envelope; never alters PhysX colliders.

The V3 native incidents identify Link5_R/world, not an individual world collider.
The waist AABB is therefore an extra planning exclusion zone, not a claim that
the waist mesh was the exact contacting shape. Its bounds and the Link5 capsule
enclose the checked-in URDF collision meshes in their respective local frames.
"""
import torch
import hashlib
from pathlib import Path
from contact_hold_cuda_teacher_math import Chain, mv

LOWER=(-.088712,-.172007,-.073405)
UPPER=(.159776,.172007,.382857)
RADIUS=.0745  # covers mesh radial extent and half the sample spacing in quadrature
CLEARANCE=.025
GEOMETRY_SHA={
    'urdf/revotron_bimanual_revo3_56dof.urdf':'598cc4cc94a37e35517eba4e450b3a3b334214549c7e196bf3f1332940e71325',
    'meshes/arm/collision/waist_pitch_Link.STL':'d9153093912138ad0d9415c323047a469a38224236f959741196b70dac6b1ac2',
    'meshes/arm/collision/Link5_R.STL':'2de51a340695c89b2fd58b1171dfacc4d85aa4e6cd50879e021a2d0bde8599a8',
}


def verify_geometry():
    root=Path(__file__).resolve().parent.parent/'revotron_revo3/source/revotron_system'
    for name,expected in GEOMETRY_SHA.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Forearm routing geometry changed: '+name)


class ForearmClearance:
    def __init__(self,k,names,device):
        verify_geometry()
        self.chain=Chain(k,'Link5_R',names,device)
        self.lo=torch.tensor(LOWER,dtype=torch.float64,device=device)
        self.hi=torch.tensor(UPPER,dtype=torch.float64,device=device)
        self.points=torch.zeros((33,3),dtype=torch.float64,device=device)
        self.points[:,2]=torch.linspace(-.198,.023496,33,device=device,dtype=torch.float64)

    def measure(self,arm):
        """Minimum signed envelope clearance and its 7-arm-joint gradient."""
        pos,rot,origins,axes=self.chain.fk(arm[:,:5])
        points=pos[:,None]+torch.einsum('bij,kj->bki',rot,self.points)
        center=(self.lo+self.hi)/2;half=(self.hi-self.lo)/2
        offset=points-center;d=offset.abs()-half;outside=d.clamp_min(0)
        norm=outside.norm(dim=-1);inside=d.amax(-1).clamp_max(0)
        distance=norm+inside-RADIUS
        nearest=distance.argmin(1);lane=torch.arange(len(arm),device=arm.device)
        p=points[lane,nearest];delta=d[lane,nearest];off=offset[lane,nearest]
        out=outside[lane,nearest];length=norm[lane,nearest]
        normal=out*off.sign()/length[:,None].clamp_min(1e-12)
        axis=delta.argmax(-1);internal=torch.zeros_like(normal)
        internal.scatter_(1,axis[:,None],off.gather(1,axis[:,None]).sign())
        normal=torch.where((length>1e-12)[:,None],normal,internal)
        jac=torch.cross(axes,p[:,None]-origins,dim=-1)
        gradient=torch.zeros_like(arm);gradient[:,:5]=(jac*normal[:,None]).sum(-1)
        return distance[lane,nearest],gradient

    def steer(self,arm,dq,tip,origins,axes):
        distance,g=self.measure(arm)
        # A bounded corrective velocity, before step/velocity/soft-limit projection.
        desired=(CLEARANCE+.010-distance).clamp(-.004,.004)
        missing=(desired-(g*dq).sum(1)).clamp_min(0)
        a=axes[:,:7];o=origins[:,:7]
        j=torch.cat((torch.cross(a,tip[:,None]-o,dim=-1).transpose(1,2),a.transpose(1,2)*.065),1)
        jj=j@j.transpose(1,2)+torch.eye(6,device=arm.device,dtype=arm.dtype)*1e-7
        null_g=g-mv(j.transpose(1,2),torch.linalg.solve(jj,mv(j,g).unsqueeze(-1)).squeeze(-1))
        norm=(null_g*g).sum(1)
        direction=torch.where((norm>1e-6)[:,None],null_g,g)
        correction=direction*(missing/(direction*g).sum(1).clamp_min(1e-10))[:,None]
        return dq+torch.where((distance<CLEARANCE+.025)[:,None],correction,torch.zeros_like(correction))

    def project(self,previous,proposed):
        """Do not enter the envelope or worsen an inherited close configuration."""
        before,_=self.measure(previous)
        floor=torch.minimum(before,torch.full_like(before,CLEARANCE))
        result=previous.clone();accepted=torch.zeros(len(previous),device=previous.device,dtype=torch.bool)
        for scale in (1.,.5,.25,.125,.0625):
            trial=previous+scale*(proposed-previous);after,_=self.measure(trial)
            valid=(after>=floor-1e-8)&~accepted
            result=torch.where(valid[:,None],trial,result);accepted|=valid
        return result
