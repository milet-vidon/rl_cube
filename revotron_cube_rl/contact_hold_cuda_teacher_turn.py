"""GPU post-lift elbow, turn, wrist and hold correction IK for the pickup teacher."""
import math
import torch
from contact_hold_cuda_teacher_math import Chain, bounded_ls, mv, rotvec, log_rot

RAD=math.pi/180.


def timed_arm(knots,seconds,hz,velocity,acceleration,lo,hi):
    """PCHIP on equal knot intervals, using the original quintic clock and retiming rule."""
    b,m,d=knots.shape; diff=knots[:,1:]-knots[:,:-1]
    a,beta=diff[:,:-1],diff[:,1:];same=(a*beta)>0
    mid=torch.where(same,2*a*beta/torch.where(same,a+beta,1.),0.)
    def end(a,b):
        s=(3*a-b)*.5
        return torch.where(s*a<=0,0.,torch.where((a*b<=0)&(s.abs()>3*a.abs()),3*a,s))
    tangents=torch.cat([end(diff[:,0],diff[:,1])[:,None],mid,end(diff[:,-1],diff[:,-2])[:,None]],1)
    s=knots.new_full((b,),seconds);ok=torch.ones(b,dtype=torch.bool,device=knots.device)
    for _ in range(8):
        n=(s*hz).ceil().long().clamp_min(2)
        # Impossible outliers are failed independently, without allocating a gigantic temporary.
        ok &= n<=18000
        n=n.clamp_max(18000)
        tick=torch.arange(int(n.max())+1,device=s.device)
        u=torch.minimum(tick[None]/n[:,None].double(),s.new_ones(()))
        t=((10*u**3-15*u**4+6*u**5)*(m-1)).clamp(0,m-1)
        i=t.long().clamp_max(m-2);z=(t-i)[...,None];lane=torch.arange(b,device=s.device)[:,None]
        x=(2*z**3-3*z**2+1)*knots[lane,i]+(z**3-2*z**2+z)*tangents[lane,i]+(-2*z**3+3*z**2)*knots[lane,i+1]+(z**3-z**2)*tangents[lane,i+1]
        ok &= ((x>=lo)&(x<=hi)).all(dim=(1,2))
        v=x.diff(dim=1)*hz;acc=v.diff(dim=1)*hz
        vm=(v.abs()/velocity).amax((1,2))
        # Ignore acceleration across the padded tail.
        am=torch.where((tick[:-2][None]<n[:,None]-1)[...,None],acc.abs()/acceleration,0.).amax((1,2)).sqrt()
        ratio=torch.maximum(vm,am); needs=(ratio>1)&ok
        if not bool(needs.any()):return x.float(),n+1,ok
        s=torch.where(needs,s*ratio*1.03,s)
    return x.float(),n+1,ok&~needs


class TurnIK:
    def __init__(self,k,names,device,hz):
        self.chain=Chain(k,'left_palm_link',names,device)
        self.ids=self.chain.ix;self.hz=hz
        self.limit=torch.as_tensor([k.limits[names[i]] for i in self.ids],device=device,dtype=torch.float64)
        self.lo=self.limit[:,0]+2*RAD;self.hi=self.limit[:,1]-2*RAD
        self.weights=self.lo.new_tensor([.3,.3,.5,.8,4,10,10]).sqrt()

    def solve(self,initial,position,rotation,elbow,wrist=None):
        """WristDirectionPlanner/ElbowOutPlanner objectives and feasibility gates."""
        ref=initial.clamp(self.lo+1e-7,self.hi-1e-7)
        def residual(x,w=wrist):
            p,r,o,_=self.chain.fk(x);s,e,z=o[...,1,:],o[...,3,:],o[...,6,:]
            arm=e-s
            com=[100*(p-position[:,None]),12*log_rot(rotation[:,None]@r.mT)]
            if w is None:
                com.extend([35*(elbow[:,None]-arm[...,1]).clamp_min(0)[...,None],.008*self.weights*(x-ref[:,None]),.015*x[...,5:6]])
            else:
                com.extend([35*(.18-arm[...,1]).clamp_min(0)[...,None],35*(.05-(e-z)[...,1]).clamp_min(0)[...,None],5*(x[...,5:6]-w[:,None,None]),.005*(x-ref[:,None])])
            return torch.cat(com,-1)
        def fit(w):
            v=bounded_ls(lambda x:residual(x,w),ref,self.lo,self.hi,iterations=100)
            p,r,o,_=self.chain.fk(v)
            ok=((p-position).norm(dim=-1)<=.002)&(log_rot(rotation@r.mT).norm(dim=-1)<=RAD)&v.isfinite().all(-1)
            return v,p,r,o,ok
        v,_,_,_,ok=fit(wrist)
        if wrist is None:return v,ok
        eligible=(wrist>=7*RAD)&~ok
        if not bool(eligible.any()):return v,ok
        for offset in (-2,2,-4,4,-6):
            alt=wrist+offset*RAD
            sel=eligible&~ok&(alt>=7*RAD)&(alt<=16*RAD)
            if not bool(sel.any()):continue
            trial,_,r,o,passed=fit(alt);s,e,z=o[:,1],o[:,3],o[:,6]
            direction=mv(r.mT,torch.nn.functional.normalize(z-e,dim=-1))
            passed &= sel & (trial[:,5]>=5*RAD)&(torch.atan2(direction[:,1],direction[:,2])<=-5*RAD)&(direction[:,2]>.3)&((e-z)[:,1]>=.05)&((e-s)[:,1]>=.16)
            v=torch.where(passed[:,None],trial,v);ok|=passed
        remaining=torch.where(eligible&~ok)[0]
        if len(remaining):
            trial,passed=self.recovery(ref[remaining],position[remaining],rotation[remaining])
            v[remaining]=torch.where(passed[:,None],trial,v[remaining]);ok[remaining]=passed
        return v,ok

    def recovery(self,ref,position,rotation):
        lo=self.lo.clone();lo[5]=max(6*RAD,float(lo[5]));ref=ref.clamp(lo+1e-7,self.hi-1e-7)
        def metrics(x):
            p,r,o,_=self.chain.fk(x);s,e,w=o[...,1,:],o[...,3,:],o[...,6,:]
            arm=e-s;d=mv(r.mT,torch.nn.functional.normalize(w-e,dim=-1))
            return p,r,arm[...,1],(e-w)[...,1],d[...,2],torch.atan2(d[...,1],d[...,2]),torch.atan2(arm[...,1],-arm[...,2])/RAD
        def residual(x):
            p,r,out,ew,dz,tilt,deg=metrics(x)
            err=[300*(.165-out),300*(.055-ew),30*(7*RAD-x[...,5]),10*(x[...,5]-16*RAD),10*(.32-dz),10*(tilt+6*RAD),36-deg]
            return torch.cat([300*(p-position[:,None]),60*log_rot(rotation[:,None]@r.mT),torch.stack(err,-1).clamp_min(0),.02*(x-ref[:,None])],-1)
        ok=torch.zeros(len(ref),dtype=torch.bool,device=ref.device);v=ref.clone()
        for w in (ref[:,5],10*RAD,8*RAD):
            x=ref.clone();x[:,5]=w
            trial=bounded_ls(residual,x,lo,self.hi,iterations=130)
            p,r,out,ew,dz,tilt,deg=metrics(trial)
            passed=(~ok)&((p-position).norm(dim=-1)<=.002)&(log_rot(rotation@r.mT).norm(dim=-1)<=RAD)&(out>=.162)&(ew>=.055)&(deg>=35.5)&(trial[:,5]>=6*RAD)&(dz>.32)&(tilt<=-6*RAD)
            v=torch.where(passed[:,None],trial,v);ok|=passed
            if bool(ok.all()):break
        return v,ok

    def workspace(self,x):
        p0,r0,_,_=self.chain.fk(x);ref=x.clamp(self.lo+1e-7,self.hi-1e-7)
        def residual(z):
            p,r,o,_=self.chain.fk(z);s,e,w=o[...,1,:],o[...,3,:],o[...,6,:]
            outside=p-p.clamp(p.new_tensor([.20,.05,-.08]),p.new_tensor([.55,.39,.30]))
            return torch.cat([2*(p-p0[:,None]),40*log_rot(r0[:,None]@r.mT),50*(.18-(e-s)[...,1:2]).clamp_min(0),
                              50*(.05-(e-w)[...,1:2]).clamp_min(0),15*(z[...,5:6]-12*RAD),100*outside,.003*(z-ref[:,None])],-1)
        q=bounded_ls(residual,ref,self.lo,self.hi,iterations=160)
        p,_,o,_=self.chain.fk(q)
        return p,((p-p0).norm(dim=-1)<=.25)&((o[:,3,1]-o[:,1,1])>=.16)

    def path(self,initial,cube_root):
        x=initial[:,self.ids].double();cube=cube_root.double()
        p,r,o,_=self.chain.fk(x); start=(o[:,3,1]-o[:,1,1]); shift=x.new_tensor([-.045,.20,.055])
        ok=torch.ones(len(x),device=x.device,dtype=torch.bool);chunks=[];lengths=[]
        def segment(knots,seconds,vel,acc):
            seq,ls,valid=timed_arm(torch.stack(knots,1),seconds,self.hz,vel,acc,self.limit[:,0],self.limit[:,1])
            chunks.append(seq);lengths.append(ls);return valid
        outward=[x]
        for k in range(1,31):
            u=k/30;x,passed=self.solve(x,p+u*shift,r,(1-u)*start+u*.18);ok &= passed;outward.append(x)
        ok &= segment(outward,4.,.35,.7)
        p,r,_,_=self.chain.fk(x);center=cube+shift;turn=[x]
        for k in range(1,46):
            a=rotvec(x.new_tensor([-math.pi/2*k/45,0.,0.]))
            x,passed=self.solve(x,center+mv(a,p-center),a@r,torch.full_like(start,.18));ok &= passed;turn.append(x)
        ok &= segment(turn,6.,.35,.7)
        # The reference uses the previous path's float32 endpoint.
        x=x.float().double(); p,r,_,_=self.chain.fk(x);end,valid=self.workspace(x);ok &= valid
        old=x[:,5];wrist=[x]
        for k in range(1,41):
            u=k/40;x,passed=self.solve(x,(1-u)*p+u*end,r,torch.full_like(start,.18),(1-u)*old+u*12*RAD);ok &= passed;wrist.append(x)
        ok &= segment(wrist,6.,.25,.5)
        return chunks,lengths,ok
