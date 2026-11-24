"""Batch URDF kinematics and bounded least squares for development teacher queries."""
import math
import numpy as np
import torch


def mv(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


def skew(a):
    x, y, z = a.unbind(-1); o = x*0
    return torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(*a.shape[:-1],3,3)


def rotvec(v):
    a = v.norm(dim=-1)[...,None,None]
    h = skew(v)
    return torch.eye(3,device=v.device,dtype=v.dtype) + torch.sinc(a/math.pi)*h + .5*torch.sinc(a/(2*math.pi)).square()*(h@h)


def quat_matrix(xyzw):
    xyzw=torch.nn.functional.normalize(xyzw,dim=-1)
    h=skew(xyzw[...,:3]); w=xyzw[...,3,None,None]
    return torch.eye(3,device=h.device,dtype=h.dtype)+2*(w*h+h@h)


def log_rot(r):
    """Principal rotation vector, with a stable branch for half turns."""
    s=torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1)*.5
    c=(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    sine=s.norm(dim=-1); angle=torch.atan2(sine,c)
    v=s*(angle/sine.clamp_min(1e-14))[...,None]
    # Largest column of R+I fixes the axis near 180 degrees.
    p=r+torch.eye(3,device=r.device,dtype=r.dtype)
    i=p.diagonal(dim1=-2,dim2=-1).argmax(-1)
    axis=p.gather(-1,i[...,None,None].expand(*i.shape,3,1)).squeeze(-1)
    axis=torch.nn.functional.normalize(axis,dim=-1)
    axis=torch.where((axis*s).sum(-1,keepdim=True)<0,-axis,axis)
    return torch.where(((sine<1e-7)&(c<0))[...,None],axis*angle[...,None],v)


class Chain:
    """Collapse fixed URDF transforms once; return revolute origins and axes with FK."""
    def __init__(self, k, link, names, device, stop=None, point=None):
        items=[]
        while link!=stop and link in k.by_child:
            row=k.by_child[link];items.append(row);link=row[1]
        if stop and link!=stop:raise ValueError('Disconnected URDF chain')
        origins=[];axes=[]; joints=[]; t=np.eye(4)
        for name,_,_,origin,axis in reversed(items):
            t=t@origin
            if axis is not None:
                joints.append(names.index(name));origins.append(t);axes.append(axis);t=np.eye(4)
        if point is not None:t[:3,3]+=t[:3,:3]@point
        fn=lambda v:torch.as_tensor(np.array(v),dtype=torch.float64,device=device)
        self.ix=joints; self.r=fn(origins)[:,:3,:3]; self.p=fn(origins)[:,:3,3]; self.axis=fn(axes)
        self.h=skew(self.axis); self.h2=self.h@self.h
        self.endr=fn(t[:3,:3]); self.endp=fn(t[:3,3])

    def fk(self, x):
        h=self.h; angles=x[...,None,None]
        rj=self.r @ (torch.eye(3,dtype=x.dtype,device=x.device)+angles.sin()*h+(1-angles.cos())*self.h2)
        r=torch.eye(3,dtype=x.dtype,device=x.device);p=x.new_zeros((*x.shape[:-1],3))
        origins=[]; axes=[]
        for j in range(len(self.ix)):
            p=p+mv(r,self.p[j]);axes.append(mv(r,self.r[j]@self.axis[j]));origins.append(p)
            r=r@rj[...,j,:,:]
        return p+mv(r,self.endp),r@self.endr,torch.stack(origins,-2),torch.stack([a.expand_as(p) for a in axes],-2)


def bounded_ls(residual,x,lo,hi,iterations=50,jacobian=None):
    """Box-constrained, scaled trust region Gauss-Newton; one system per row.
    
    On a blocked joint use its KKT sign before solving for free joints. Each lane
    adjusts damping from its own actual/predicted reduction. All residuals and FK
    evaluations, including the finite-difference axis, stay on the input device.
    """
    x=torch.maximum(torch.minimum(x,hi-1e-8),lo+1e-8)
    d=x.shape[-1]; eye=torch.eye(d,dtype=x.dtype,device=x.device)
    # residual accepts [batch,probes,d]; the leading probe is the reference.
    def value(z):return residual(z[:,None])[:,0]
    damping=x.new_full((len(x),),1e-4); done=torch.zeros_like(damping,dtype=torch.bool)
    for it in range(iterations):
        if jacobian is None:
            eps=2e-6
            rr=residual(torch.cat((x[:,None],x[:,None]+eps*eye),1))
            f=rr[:,0];j=(rr[:,1:]-f[:,None]).transpose(1,2)/eps
        else:f,j=jacobian(x)
        g=(j.transpose(1,2)@f.unsqueeze(-1)).squeeze(-1)
        blocked=((x<=lo+1e-9)&(g>0))|((x>=hi-1e-9)&(g<0))
        free=(~blocked).to(x.dtype)
        jf=j*free[:,None,:]; H=jf.transpose(1,2)@jf
        diag=H.diagonal(dim1=-2,dim2=-1).clamp_min(.02)
        A=H+torch.diag_embed(damping[:,None]*diag+(1-free))
        s=torch.linalg.solve_ex(A,(-g*free).unsqueeze(-1)).result.squeeze(-1)
        cand=(x+s).clamp(lo,hi); step=cand-x
        fc=value(cand); old=f.square().sum(-1); new=fc.square().sum(-1)
        pred=old-(f+mv(j,step)).square().sum(-1); gain=(old-new)/pred.clamp_min(1e-20)
        ok=(new<old)&~done
        x=torch.where(ok[:,None],cand,x)
        damping=torch.where((gain>.75)&ok,(damping*.3).clamp_min(1e-9),
                 torch.where((gain<.25)|~ok,(damping*5).clamp_max(1e12),damping))
        done|=((g*free).abs().amax(-1)<1e-9)|(ok & ((old-new)<1e-11*old.clamp_min(1.)))|(damping>=1e11)
        if it%6==5 and bool(done.all()):break
    return x


class FingerFeedback:
    """Left fingertip arrival IK, contact freeze and 0.005-rad slew per control step."""
    def __init__(self,k,g,names,robot_z,n,device):
        self.names=names;self.grasp=g;self.fingers=('thumb','index','middle','ring','little')
        self.hand=Chain(k,'left_hand_base_link',names,device)
        self.robot_z=robot_z
        self.chains=[];self.refs=[];self.bounds=[];self.anchors=[]
        self.applied=torch.full((n,len(names)),float('nan'),device=device,dtype=torch.float64)
        self.targets=self.applied.clone()
        self.tick=torch.zeros(n,device=device,dtype=torch.long)
        tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
        for f in self.fingers:
            anchor=g['contact_anchors'][f]
            c=Chain(k,anchor['link'],names,device,stop='left_hand_base_link',point=np.asarray(anchor['point_local']))
            self.chains.append(c)
            nm=[names[i] for i in c.ix]; ref=tensor([g['joint_targets'][a] for a in nm])
            lim=tensor([k.limits[a] for a in nm]); rad=tensor([.005 if 'MPR' in a else .30 for a in nm])
            self.refs.append(ref);self.bounds.append((torch.maximum(lim[:,0]+.02,ref-rad),torch.minimum(lim[:,1]-.02,ref+rad)))
            self.anchors.append(tensor(anchor['target_cube']))

    def update(self,mask,q,goal,contacts,core):
        ii=torch.where(mask)[0]
        if not len(ii):return goal
        q=q[ii].double();r=quat_matrix(core[ii,3:7].double())
        pos=core[ii,:3].double();pos[:,2]-=self.robot_z
        handp,handr,_,_=self.hand.fk(q[:,self.hand.ix]); weight=q.new_tensor([60.,15.,60.])
        out=goal.clone()
        for f,c in enumerate(self.chains):
            ix=c.ix; fi=torch.as_tensor(ix,device=q.device)
            touching=contacts[ii,f]>1e-5; sel=torch.where(((self.tick[ii]%3)==0)&~touching)[0]
            if len(sel):
                ref=self.refs[f];lo,hi=self.bounds[f]
                # Transform the fingertip into cube coordinates, then apply the
                # same axis-dependent weights as the scalar teacher.
                R=r[sel].mT@handr[sel]; p=mv(r[sel].mT,handp[sel]-pos[sel])-self.anchors[f]
                def residual(x):
                    tip=c.fk(x)[0];e=mv(R[:,None],tip)+p[:,None]
                    return torch.cat([e*weight,.015*(x-ref)],dim=-1)
                def jac(x):
                    tip,_,o,a=c.fk(x);d=torch.linalg.cross(a,tip[:,None]-o).transpose(1,2)
                    extra=.015*torch.eye(len(ix),device=q.device,dtype=q.dtype).expand(len(x),-1,-1)
                    j=torch.cat([weight[None,:,None]*(R@d),extra],1)
                    y=torch.cat([(mv(R,tip)+p)*weight,.015*(x-ref)],1)
                    return y,j
                fit=bounded_ls(residual,q[sel][:,ix],lo,hi,iterations=30,jacobian=jac)
                self.targets[ii[sel,None],fi[None]]=fit
            prev=self.applied[ii[:,None],fi];prev=torch.where(prev.isnan(),goal[ii][:,ix].double(),prev)
            tgt=self.targets[ii[:,None],fi]
            nxt=prev+torch.where(touching[:,None]|tgt.isnan(),0.,tgt-prev).clamp(-.005,.005)
            self.applied[ii[:,None],fi]=nxt;out[ii[:,None],fi]=nxt.to(out.dtype)
        self.tick[ii]+=1
        return out
