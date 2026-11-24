"""Development contact-hold teacher: one batched state machine on the physics device.

Inputs use canonical joint order, lane-local xyz, and PhysX xyzw. This teacher
produces unprojected commands; collection still owns the joint safety projector
and the episode/shape/contact safety gates. It never writes object state.
"""
import json
from pathlib import Path
import torch
from kinematics import RobotKinematics
from contact_hold_cuda_teacher_math import FingerFeedback, quat_matrix, rotvec, mv, log_rot
from contact_hold_cuda_teacher_turn import TurnIK, RAD

FAILURES={0:'',1:'lost_primary_grasp',2:'contact_topology_or_settle_timeout',
          3:'primary_grasp_lost_during_tilt',4:'tilt_hold_timeout',
          5:'planner_rejected: post-lift IK, limits or time envelope',
          6:'planner_rejected: Post-turn correction exceeded bounded pose envelope',
          7:'planner_rejected: post-turn IK violates wrist pose envelope'}


class BatchedTeacher:
    def __init__(self,plan,cfg,candidate,gain_writer=None):
        self.plan=plan;self.cfg=cfg;self.c=candidate;self.writer=gain_writer
        device=plan.accepted.device;n=len(plan.accepted)
        self.phase=torch.zeros(n,device=device,dtype=torch.long) # 0..5 pickup, 6 after handoff
        self.tick=torch.zeros_like(self.phase); self.wait=self.tick.clone(); self.streak=self.tick.clone()
        self.lost=self.tick.clone();self.hold=self.tick.clone();self.failure=self.tick.clone()
        self.accepted=torch.zeros_like(self.phase,dtype=torch.bool)
        self.g=json.loads(Path(__file__).with_name('opposition_grasp_v20.json').read_text())
        self.tol=self.g.get('arm_settle_tolerance_rad',.035)
        names=cfg['joint_names']; k=RobotKinematics()
        self.fingers=FingerFeedback(k,self.g,names,cfg['robot_z'],n,device)
        self.turn=TurnIK(k,names,device,cfg['control_hz'])
        self.arm=[i for i,j in enumerate(names) if j.startswith('Joint')]
        self.left=[i for i,j in enumerate(names) if j.startswith('left_')]
        self.pair=[i for i,j in enumerate(names) if j.startswith(('left_thumb_','left_little_'))]
        self.reference=torch.eye(3,device=device,dtype=torch.float64).repeat(n,1,1)
        self.turn_lens=torch.ones((n,3),device=device,dtype=torch.long)
        self.turn_goal=torch.zeros((n,len(names)),device=device)
        self.path=None
        self.center=torch.zeros((n,3),device=device,dtype=torch.float64)
        self.corrected=self.accepted.clone();self.palm_ref=self.reference.clone();self.palm_ref_pos=self.center.clone()
        self.rx=rotvec(self.center.new_tensor([-90*RAD,0.,0.]))

    def _begin_turn(self,ii,nominal,core):
        zoff=self.cfg['robot_z']
        for js in ii.split(32):
            root=core[js,:3].double();root[:,2]-=zoff
            chunks,lens,valid=self.turn.path(nominal[js],root)
            self.failure[js]=torch.where(valid,0,5)
            selected=torch.where(valid)[0]
            if not len(selected):continue
            lanes=js[selected];ls=torch.stack(lens,1)[selected];total=ls.sum(1)
            size=int(total.max())
            if self.path is None or self.path.shape[1]<size:
                backing=nominal.new_zeros((len(nominal),size,len(self.turn.ids)))
                if self.path is not None:backing[:,:self.path.shape[1]]=self.path
                self.path=backing
            for j,chunk in enumerate(chunks):
                lane_i,local=torch.where(torch.arange(chunk.shape[1],device=ls.device)[None]<ls[:,j,None])
                offset=ls[:,:j].sum(1)
                self.path[lanes[lane_i],offset[lane_i]+local]=chunk[selected[lane_i],local]
            self.turn_goal[lanes]=nominal[lanes]
            self.reference[lanes]=quat_matrix(core[lanes,3:7].double())
            self.turn_lens[lanes]=ls
            self.phase[lanes]=6;self.tick[lanes]=0;self.lost[lanes]=0

    def _gains(self,ii,delta):
        c=self.c
        def write(sel,prop,joints,value):
            if not len(sel):return
            if self.writer is None:raise ValueError('Missing PhysX gain writer during wrist hold')
            v=torch.as_tensor(value,dtype=torch.float32,device=ii.device).expand(len(sel),len(joints)).contiguous()
            self.writer(ii[sel],prop,joints,v)
        start=torch.where(delta==0)[0]
        write(start,'stiffnesses',self.left,c['left_finger_hold_stiffness'])
        write(start,'dampings',self.left,c['left_finger_hold_damping'])
        final=c['thumb_little_final_stiffness'];count=c['thumb_little_ramp_steps']; first=c['left_finger_hold_stiffness']
        if final!=first:
            ramp=torch.where((delta>=1)&(delta<=count))[0]
            write(ramp,'stiffnesses',self.pair,first+(final-first)*delta[ramp,None]/count)

    def _turn_step(self,ii,core,contacts,vel,goal,display):
        lens=self.turn_lens[ii];length=lens.sum(1);tick=self.tick[ii];delta=tick-length
        self._gains(ii,delta)
        display[ii]=torch.where(tick<lens[:,0],6,torch.where(tick<lens[:,:2].sum(1),7,torch.where(delta<0,8,9)))
        ri=quat_matrix(core[ii,3:7].double());err=log_rot(self.rx@self.reference[ii]@ri.mT)
        angle=err.norm(dim=-1)
        fix=torch.where((delta>=0)&(delta%10==0)&(angle>2.5*RAD))[0]
        if len(fix):
            lane=ii[fix];last=length[fix]-1
            x=self.path[lane,last].double();p,r,_,_=self.turn.chain.fk(x)
            new=~self.corrected[lane];xyz=core[lane,:3].double();xyz[:,2]-=self.cfg['robot_z']
            self.center[lane]=torch.where(new[:,None],xyz,self.center[lane])
            self.palm_ref[lane]=torch.where(new[:,None,None],r,self.palm_ref[lane])
            self.palm_ref_pos[lane]=torch.where(new[:,None],p,self.palm_ref_pos[lane])
            self.corrected[lane]=True
            d=rotvec(err[fix]*torch.minimum(torch.full_like(angle[fix],.5),RAD/angle[fix])[:,None])
            center=self.center[lane];desired_p=center+mv(d,p-center);desired_r=d@r
            bounded=((desired_p-self.palm_ref_pos[lane]).norm(dim=-1)<=.04)&(log_rot(desired_r@self.palm_ref[lane].mT).norm(dim=-1)<=15*RAD)
            fitted,valid=self.turn.solve(x,desired_p,desired_r,x.new_full((len(x),),.18),x.new_full((len(x),),12*RAD))
            self.failure[lane]=torch.where(~bounded,6,torch.where(valid,0,7))
            self.path[lane,last]=torch.where((bounded&valid)[:,None],fitted,x).float()
        goal[ii]=self.turn_goal[ii]
        arm_ix=torch.as_tensor(self.turn.ids,device=ii.device)
        goal[ii[:,None],arm_ix]=self.path[ii,torch.minimum(tick,length-1)]
        touches=contacts[ii,:5]>1e-5;retained=touches[:,0]&(touches[:,1:].sum(1)>=3)
        self.lost[ii]=torch.where(retained,0,self.lost[ii]+1)
        qualified=(delta>=0)&touches.all(1)&(core[ii,2]-self.cfg['table_z']-.0285>=.08)&(contacts[ii,12]<1e-5)&(angle<=5*RAD)
        qualified &= (vel[ii,:3].norm(dim=-1)<.03)&(vel[ii,3:].norm(dim=-1)<.20)
        self.hold[ii]=torch.where(qualified,self.hold[ii]+1,0)
        self.accepted[ii]=self.hold[ii]>=90
        self.tick[ii]+=1
        failed=torch.where(self.tick[ii]>length+300,4,torch.where(self.lost[ii]>=10,3,0))
        self.failure[ii]=torch.where(self.failure[ii]!=0,self.failure[ii],failed)

    @torch.no_grad()
    def action(self,active,q,v,core,target,contacts,velocity):
        """Return (absolute raw joint goal, canonical displayed phase, failure code, accepted)."""
        pickup=active&(self.phase<6); turning=active&(self.phase==6);before=self.phase
        goal,length=self.plan.goal(before.clamp_max(5),self.tick)
        nominal=goal.clone();display=before.clone()
        late=torch.where(turning)[0]
        if len(late):self._turn_step(late,core,contacts,velocity,goal,display)
        fingers=turning|(pickup&((before==5)|((before==4)&(self.tick>=length-1))))
        goal=self.fingers.update(fingers,q,goal,contacts,core)
        touches=contacts[:,:5]>1e-5; all5=touches.all(1); topology=touches[:,0]&(touches[:,1:].sum(1)>=3)
        self.streak=torch.where(pickup,torch.where(all5,self.streak+1,0),self.streak)
        lift=pickup&(before==5);self.lost=torch.where(lift,torch.where(topology,0,self.lost+1),self.lost)
        self.failure=torch.where(lift&(self.lost>=10),1,self.failure)
        going=pickup&(self.failure==0)
        next_tick=self.tick+going.long();end=going&(next_tick>=length)
        settled=((q[:,self.arm]-goal[:,self.arm]).abs().amax(1)<self.tol)&(v[:,self.arm].abs().amax(1)<.18)
        ready=settled&((before<4)|(self.streak>=10))&((before<5)|((core[:,2]-self.cfg['table_z']-.0285>=.08)&(contacts[:,12]<1e-5)))
        advance=end&ready;stall=end&~ready
        self.wait=torch.where(advance,0,self.wait+stall.long())
        self.failure=torch.where(stall&(self.wait>120),2,self.failure)
        self.tick=torch.where(advance,0,torch.where(stall,length-1,next_tick))
        self.phase=torch.where(advance&(before<5),before+1,before)
        handoff=torch.where(advance&lift)[0]
        if len(handoff):
            self._begin_turn(handoff,nominal,core)
            display[handoff]=torch.where(self.failure[handoff]==0,6,5)
        good=active&(self.failure==0)
        return torch.where(good[:,None],goal,target),display,self.failure,self.accepted
