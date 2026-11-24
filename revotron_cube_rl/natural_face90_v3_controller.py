"""Bounded robot-joint policy for development simulation: R27 grasp -> present -> twist nZ.

Read-only with respect to physics. The caller owns all collision/velocity/soft-limit
and episode gates, and applies its ordinary projected 56-joint actuator command.
The 6 policy coefficients affect robot geometry/timing only, never acceptance.
"""
from pathlib import Path
import hashlib
import json
import math
import torch
from kinematics import RobotKinematics
from contact_hold_cuda_teacher_math import Chain, mv, rotvec

PARAM_NAMES=('presentation_inward','presentation_roll','pad_radius','pad_pressure','arc_overtravel','preshape')
PHASES=('present','outside','hover','engage','sweep','release')
PAD_SHA='b9ab73c9e7e2c922ffaf1270196fce8c2a4acd635dfe8842def32d98bcbe7b44'
MAX_POST_STEPS=1900
HALF_EDGE=.045
PRESENT_FRAMES=60  # two seconds at the validated 30 Hz control rate

def _unit(v):
    return v/v.norm(dim=-1,keepdim=True).clamp_min(1e-10)

def face_point(radius,radial,deg,depth):
    """Requested pad position in Core coordinates for a signed rotation about the OUTWARD nZ axis."""
    rad=deg*math.pi/180
    tangent=torch.stack((radial[:,1],-radial[:,0]),-1)
    xy=radius[:,None]*(radial*rad.cos()[:,None]+tangent*rad.sin()[:,None])
    return torch.cat((xy,(-HALF_EDGE+depth)[:,None]),-1)

def _align(src,dst):
    # Minimal rotation that aligns a single palm axis; rotation about it remains free.
    a,b=_unit(src),_unit(dst);v=torch.cross(a,b,dim=-1);sn=v.norm(dim=-1);co=(a*b).sum(-1).clamp(-1,1)
    lazy=torch.zeros_like(a);axis=a.abs().argmin(-1);lazy.scatter_(1,axis[:,None],1.)
    direction=torch.where((sn<1e-6)[:,None],_unit(torch.cross(a,lazy,dim=-1)),_unit(v))
    return direction*torch.atan2(sn,co)[:,None]

def _dls(pos,orientation,origins,axes,goal,ang_error,max_step):
    # Measured-q geometric translational/angular Jacobian for the 7 arm DOFs.
    a=axes[:,:7]; origins=origins[:,:7]
    lin=torch.cross(a,pos[:,None]-origins,dim=-1)
    J=torch.cat((lin.transpose(1,2),a.transpose(1,2)*.065),dim=1)
    dx=goal-pos
    dx=dx*(.0050/dx.norm(dim=-1,keepdim=True).clamp_min(.0050))
    da=ang_error*(.018/ang_error.norm(dim=-1,keepdim=True).clamp_min(.018))
    err=torch.cat((dx,da*.065),1)
    k=J@J.transpose(-2,-1)+torch.eye(6,device=J.device,dtype=J.dtype)*.008**2
    dq=mv(J.transpose(-2,-1),torch.linalg.solve(k,err.unsqueeze(2)).squeeze(2))
    return dq.clamp(-max_step,max_step)

class NaturalFace90Controller:
    def __init__(self,e,parameters,target_deg=15.):
        import numpy as np
        self.e=e;self.device=e.home.device;self.n=len(e.home);self.names=list(e.names)
        if len(self.names)!=56 or e.cfg['control_hz']!=30 or e.cfg['physics_hz']!=240:raise ValueError('Expected canonical 56-joint 30/240 Hz environment')
        self.deg=float(target_deg)
        if self.deg not in (15.,30.,90.):raise ValueError('Expected a staged positive nZ target: 15, 30 or 90 degrees')
        t=torch.as_tensor(parameters,dtype=torch.float64,device=self.device)
        if t.shape==(6,):t=t[None].expand(self.n,-1)
        if t.shape!=(self.n,6) or not t.isfinite().all() or (t.abs()>1).any():raise ValueError('Expected six bounded physical policy parameters per lane')
        self.p=t.clone()
        asset=Path(__file__).with_name('face_contact_calibration_v1.json');b=asset.read_bytes()
        if hashlib.sha256(b).hexdigest()!=PAD_SHA:raise ValueError('Unexpected index-pad calibration')
        pad=json.loads(b)
        if pad['source_link']!='right_index_DIP_rubber_link':raise ValueError('Unexpected contact link')
        k=RobotKinematics()
        self.lp=Chain(k,'left_palm_link',e.names,self.device)
        self.rp=Chain(k,'right_palm_link',e.names,self.device)
        self.tip=Chain(k,pad['source_link'],e.names,self.device,point=np.asarray(pad['point_local']))
        if self.tip.ix[:7]!=self.rp.ix or len(self.lp.ix)!=7 or len(self.rp.ix)!=7:raise ValueError('Unverified arm chains')
        self.on=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.phase=torch.zeros_like(self.on,dtype=torch.long);self.clock=self.phase.clone();self.streak=self.phase.clone()
        self.arc=t.new_zeros(self.n);self.seen=self.on.clone()
        self.root=t.new_tensor([0,0,float(e.cfg['robot_z'])])
        self.goal_pos=t.new_zeros((self.n,3));self.goal_R=t.new_zeros((self.n,3,3));self.radial=t.new_zeros((self.n,2))
        self.initial=e.home.clone().double();self.start_L=self.goal_pos.clone();self.standby_R=self.goal_pos.clone()
        self.start_LR=self.goal_R.clone();self.start_right_normal=self.goal_pos.clone();self.presentation_rot=self.goal_pos.clone()
        self.left_ref=self.initial[:,self.lp.ix].clone();self.left_load_bias=self.left_ref.clone()
        self.prev_q=self.initial.clone();self.motion=torch.zeros_like(self.initial)
        self.right_ref=self.initial[:,self.rp.ix].clone();self.right_load_bias=self.right_ref.clone()
        self.right_ready=self.on.clone();self.shape_streak=self.clock.clone()
        self.route_rotation=self.goal_R.clone()
        self.fIDs=[];self.fGoal=[]
        for finger,parts in {'index':{'MCP':.24,'PIP':.22,'DIP':.16},'middle':{'MCP':.66,'PIP':.66,'DIP':.42},
                             'ring':{'MCP':.70,'PIP':.68,'DIP':.41},'little':{'MCP':.70,'PIP':.68,'DIP':.42}}.items():
            for j,v in parts.items():
                self.fIDs.append(e.names.index(f'right_{finger}_{j}_joint'));self.fGoal.append(v)
        self.fGoal=t.new_tensor(self.fGoal)
    def _q(self):return self.e.robot.get_dof_positions()[:,self.e.ids].double()
    def _core(self):
        from contact_hold_cuda_teacher_math import quat_matrix
        p=self.e.core().double()
        return p[:,:3],quat_matrix(p[:,3:7])
    def begin(self,mask,target,obs=None):
        mask=mask.bool()
        if mask.shape!=self.on.shape or (mask&self.seen).any():raise ValueError('Each lane must start from its live R27 endpoint exactly once')
        if not mask.any():return
        q=self._q();c,R=self._core();pl,L,*_=self.lp.fk(q[:,self.lp.ix]);pr,_,*_=self.tip.fk(q[:,self.tip.ix]);pl=pl+self.root;pr=pr+self.root
        # Rotate the left palm about the measured cube centre, not the model world origin.
        sh=torch.zeros_like(c);inward=(.036+.012*self.p[:,0]).clamp_min(0)
        sh[:,1]=-c[:,1].sign()*torch.minimum(c[:,1].abs(),inward)
        sh[:,0]=.011;sh[:,2]=.012
        # +world X tilts the left palm toward the right at the R27 endpoint.
        rv=torch.zeros_like(c);rv[:,0]=(10.+3.5*self.p[:,1])*math.pi/180
        turn=rotvec(rv)
        self.goal_pos[mask]=(c+sh+mv(turn,pl-c))[mask]
        self.start_L[mask]=pl[mask]
        self.start_LR[mask]=L[mask];self.presentation_rot[mask]=rv[mask]
        self.left_ref[mask]=q[mask][:,self.lp.ix]
        self.left_load_bias[mask]=(target[mask][:,self.lp.ix].double()-self.left_ref[mask]).clamp(-.065,.065)
        self.prev_q[mask]=q[mask];self.motion[mask]=0.
        self.right_ref[mask]=q[mask][:,self.rp.ix]
        self.right_load_bias[mask]=(target[mask][:,self.rp.ix].double()-self.right_ref[mask]).clamp(-.050,.050)
        # Keep the right palm at its original location in the presentation phase;
        # only its direction can align. No target-layer approach before handoff.
        palmR,rotR,*_=self.rp.fk(q[:,self.rp.ix])
        self.standby_R[mask]=(palmR+self.root)[mask];self.start_right_normal[mask]=rotR[mask,:,0]
        self.route_rotation[mask]=rotR[mask]
        self.goal_R[mask]=(turn@L)[mask]
        # Pick a reachable quadrant using actual R27 endpoint measurements, then keep
        # this radial direction fixed IN THE CORE, not the world's arbitrary yaw.
        xy=mv(R.transpose(1,2),pr-c)[:,:2]; fallback=torch.zeros_like(xy);fallback[:,0]=1.
        self.radial[mask]=torch.where((xy.norm(dim=1)>.006)[:,None],_unit(xy),fallback)[mask]
        self.seen|=mask;self.on|=mask;self.initial[mask]=target[mask].double()
    def action(self,target,active,obs):
        active=active.bool()&self.on;failed=torch.zeros_like(active);why=['']*self.n
        if not active.any():return target.clone(),failed.clone(),failed,why
        def metric(name,kind):
            if name not in obs:raise ValueError('Missing independently measured metric: '+name)
            v=obs[name]
            if v.shape!=(self.n,):raise ValueError('Bad metric shape: '+name)
            return v.bool() if kind=='bool' else v.double()
        ready=metric('natural_ready','bool');right=metric('right_contact','bool');wrong=metric('wrong_contact','bool')
        face=metric('face_deg','float');structure=metric('structure_ok','bool');support=metric('support','bool')
        def fail(m,msg):
            nonlocal failed
            hit=active&m&~failed
            if hit.any():
                failed|=hit
                for jj in torch.where(hit)[0].tolist():why[jj]=msg
        fail(wrong,'wrong_target_contact')
        q=self._q(); c,R=self._core();pl,L,oL,aL=self.lp.fk(q[:,self.lp.ix]);pr,_,oR,aR=self.tip.fk(q[:,self.tip.ix]);rp,P,oP,aP=self.rp.fk(q[:,self.rp.ix])
        normal=-R[:,:,2]; rad=.030+.005*self.p[:,2];depth=.00085+.00065*self.p[:,3]
        zero=torch.zeros_like(rad); base=face_point(rad,self.radial,zero,zero)
        local=mv(R.transpose(1,2),pr+self.root-c); radial=self.radial;side_xy=.105*radial
        # Do not cut through the cube corner on a diagonal: first use the
        # current side of the face at >10 cm radial, THEN advance to front.
        outside=torch.cat((side_xy,-torch.full_like(rad[:,None],.105)),1)
        route=outside.clone()
        behind=local[:,2]>-.079
        # Stay radial while above/behind the selected face. Change axial side only
        # once the pad has cleared the largest possible cube diagonal in XY.
        go_out=behind&(local[:,:2].norm(dim=1)<.087)
        route[:,2]=torch.where(go_out,local[:,2],outside[:,2])
        hover=base.clone();hover[:,2]=-.080
        press=face_point(rad,self.radial,self.arc,depth)
        away=press.clone();away[:,2]=-.100
        ph=self.phase;engage=ph==3;sweep=ph==4;release=ph==5
        localgoal=torch.where((ph<=1)[:,None],route,torch.where((ph==2)[:,None],hover,torch.where(release[:,None],away,press)))
        goal_r=c+mv(R,localgoal)
        # Presentation and touch use different wrist directions: opposing the
        # supporting palm at standby, then placing the index pad toward the face.
        # Right finger shape changes gradually; left-hand grasp fingers hold their
        # exact last projected R27 commands throughout the post-prefix sequence.
        req=target.double().clone();cap=torch.minimum(self.e.step,self.e.vel/30).double().expand_as(req)
        lim=self.e.limits.double().expand(self.n,-1,-1)
        def assign(ids,x):
            lo=lim[:,ids,0];hi=lim[:,ids,1];baseq=target[:,ids].double()
            dest=x.clamp(lo,hi); req[:,ids]=baseq+(dest-baseq).clamp(-cap[:,ids],cap[:,ids])
        from contact_hold_cuda_teacher_math import log_rot
        at_start=ph==0
        t=((self.clock.double()+1)/PRESENT_FRAMES).clamp(0,1)
        u=torch.where(at_start,t**3*(10.-15.*t+6.*t*t),torch.ones_like(t))[:,None]
        show_pos=self.start_L+u*(self.goal_pos-self.start_L)
        show_R=rotvec(u*self.presentation_rot)@self.start_LR
        # Integrate IK on a *reference* configuration, never the loaded arm's measured
        # Cartesian error. Otherwise a delayed physical drive can wind up and oscillate
        # whilst supposedly holding the cube still. Preserve the loaded prefix command.
        ids=self.lp.ix
        ref_pos,ref_R,ref_o,ref_a=self.lp.fk(self.left_ref)
        dl=_dls(ref_pos,ref_R,ref_o,ref_a,show_pos-self.root,log_rot(show_R@ref_R.transpose(1,2)),.025)
        next_ref=(self.left_ref+dl.clamp(-cap[:,ids],cap[:,ids])).clamp(lim[:,ids,0],lim[:,ids,1])
        self.left_ref=torch.where(active[:,None],next_ref,self.left_ref)
        self.motion=torch.where(active[:,None],.4*self.motion+.6*(q-self.prev_q).clamp(-.045,.045),self.motion)
        self.prev_q=torch.where(active[:,None],q,self.prev_q)
        tracking=.12*(self.left_ref-q[:,ids]).clamp(-.040,.040)-.55*self.motion[:,ids]
        assign(ids,self.left_ref+self.left_load_bias+tracking)
        # Shape the fingers while the hand is still at standby. V2 reshaped them
        # during a long reach, making the pad IK chase a moving chain endpoint.
        fingers=(self.fGoal[None]+(.08*self.p[:,5,None])).expand(self.n,-1)
        fingers=torch.where(release[:,None],.55*fingers,fingers)
        fingers=torch.where(at_start[:,None],self.initial[:,self.fIDs],fingers)
        shaped=(q[:,self.fIDs]-fingers).abs().amax(1)<.045
        shaping=active&(ph==1)&~self.right_ready
        self.shape_streak=torch.where(shaping&shaped,self.shape_streak+1,torch.zeros_like(self.shape_streak))
        just_ready=shaping&(self.shape_streak>=10)&(self.clock>=45)
        self.right_ready |= just_ready
        self.route_rotation=torch.where(just_ready[:,None,None],P,self.route_rotation)
        wait_shape=(ph==1)&~self.right_ready
        standby_normal=_unit((1-u)*self.start_right_normal-u*self.goal_R[:,:,0])
        # Use a kinematic reference, preserving the measured load offset. Never
        # repeatedly integrate measured tracking lag into a new actuator target.
        ix=self.rp.ix
        fullref=q.clone();fullref[:,ix]=self.right_ref
        ref_tip,_,ref_o,ref_a=self.tip.fk(fullref[:,self.tip.ix])
        ref_palm,ref_P,ref_po,ref_pa=self.rp.fk(self.right_ref)
        standby=_dls(ref_palm,ref_P,ref_po,ref_pa,self.standby_R-self.root,
                     _align(ref_P[:,:,0],standby_normal),.018)
        # Align before leaving the outside waypoint; preserve roll about the normal.
        rot_error=_align(self.route_rotation[:,:,0],-normal)
        goal_rotation=rotvec(rot_error)@self.route_rotation
        route_error=log_rot(goal_rotation@ref_P.transpose(1,2))
        dr=_dls(ref_tip,ref_P,ref_o,ref_a,goal_r-self.root,route_error,.018)
        dr=torch.where((at_start|wait_shape)[:,None],standby,dr)
        # If the real arm lags or is blocked, do not accumulate a distant reference
        # that will command a lunge when it becomes free. This bounds reference
        # lead; the existing measured collision stop remains unconditional.
        new_ref=self.right_ref+dr.clamp(-cap[:,ix],cap[:,ix])
        new_ref=new_ref.clamp(q[:,ix]-.060,q[:,ix]+.060).clamp(lim[:,ix,0],lim[:,ix,1])
        self.right_ref=torch.where(active[:,None],new_ref,self.right_ref)
        tracking=.12*(self.right_ref-q[:,ix]).clamp(-.040,.040)-.55*self.motion[:,ix]
        # The unloaded right arm has no grasp load to preserve. Fade only its
        # handoff offset; retaining it forever biases the actual pad away from IK.
        bias_scale=torch.where(at_start,(1-t).clamp(0,1),torch.zeros_like(t))[:,None]
        assign(ix,self.right_ref+bias_scale*self.right_load_bias+tracking)
        # One third rad/s: reach starts only after the actual finger shape settles.
        slow=target[:,self.fIDs].double()+(fingers-target[:,self.fIDs].double()).clamp(-.011,.011)
        assign(self.fIDs,slow)
        self.clock+=active.long()
        goalhold=right&~wrong&support&structure&metric('contact_goal_reached','bool')&(face>=self.deg)&(face<=(self.deg+3.))
        enough=torch.where(engage,right&~wrong&support&structure,torch.where(sweep,goalhold,torch.zeros_like(active)))
        self.streak=torch.where(active,torch.where(enough,self.streak+1,0),self.streak)
        adv=active&sweep&right&~wrong&support&structure
        arc_cap=self.deg+4.+2.*self.p[:,4]
        self.arc=torch.where(adv,torch.minimum(self.arc+(.24+.075*self.p[:,4]),arc_cap),self.arc)
        near_out=((local-outside).norm(dim=1)<.022)&self.right_ready&(log_rot(goal_rotation@P.transpose(1,2)).norm(dim=1)<math.radians(4))
        near_hover=(local-hover).norm(dim=1)<.012
        measured_palm=pl+self.root
        delta=self.goal_pos-self.start_L
        achieved=((measured_palm-self.start_L)*_unit(delta)).sum(dim=1)
        near_present=((self.goal_pos-measured_palm).norm(dim=1)<.020)&(achieved>=.018)&(log_rot(self.goal_R@L.transpose(1,2)).norm(dim=1)<math.radians(5))
        advance=( (ph==0)&(self.clock>=PRESENT_FRAMES)&ready&near_present | (ph==1)&near_out | (ph==2)&near_hover | engage&(self.streak>=3) | sweep&(self.streak>=15) )&active
        timeouts=self.clock.new_tensor([450,420,210,180,420,140])[ph]
        fail(~advance&(self.clock>=timeouts), 'post_phase_timeout')
        self.phase+= (advance&~failed).long();self.clock=torch.where(advance,0,self.clock); self.streak=torch.where(advance,0,self.streak)
        self.on &= ~failed
        req=torch.where((active&~failed)[:,None],req,target.double())
        # A release signal corresponds to this frame's ACTUAL requested withdrawal,
        # not the previous frame that merely met the angular hold threshold.
        return req.to(target.dtype),active&release&~failed,failed,why
