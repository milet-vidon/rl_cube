"""Read-only, per-lane PhysX gates for natural handover followed by an nZ-layer twist.

The raw sensor knows *which cubelet* was touched; the PhysX other-actor ID
knows *which robot link*. Material proximity and inferred fingertip positions
are never substituted for contact. Quaternions are native tensor-view XYZW.
"""
import math
import re
import numpy as np
import torch

LANE=re.compile(r'^/World/envs/env_(\d+)(/.*)$')
DIGIT=re.compile(r'^/Robot/right_(?:thumb|index|middle|ring|little)_[^/]+$')
RIGHT_ARM=re.compile(r'^/Robot/Link\d+_R$')
D=180/math.pi


def check(x,msg):
    if not bool(x):raise ValueError(msg)

def unit(q):
    d=q.norm(dim=-1,keepdim=True);check(torch.isfinite(q).all() & ((d-1).abs()<.003).all(),'Invalid live unit quaternion');return q/d

def conj(q):return torch.cat((-q[...,:3],q[...,3:]),-1)
def mul(q,r):
    a,b=torch.broadcast_tensors(q[...,:3],r[...,:3]);return torch.cat((q[...,3:]*b+r[...,3:]*a+torch.linalg.cross(a,b,dim=-1),q[...,3:]*r[...,3:]-(a*b).sum(-1,keepdim=True)),-1)
def rotate(q,p):
    a,b=torch.broadcast_tensors(q[...,:3],p);t=2*torch.linalg.cross(a,b,dim=-1);return b+q[...,3:]*t+torch.linalg.cross(a,t,dim=-1)
def degrees(q):return 2*D*torch.atan2(q[...,:3].norm(dim=-1),q[...,3].abs())
def wrap(v):return (v+180)%360-180

def body_geometry(core,pieces):
    check(core.ndim==2 and core.shape[1]==7 and pieces.shape==(len(core),26,7),'Missing complete measured cube pose');check(torch.isfinite(core).all() & torch.isfinite(pieces).all(),'Nonfinite cube state')
    qi=conj(unit(core[:,None,3:]));return rotate(qi,pieces[:,:,:3]-core[:,None,:3]),mul(qi,unit(pieces[:,:,3:]))

class Layer:
    def __init__(self,names,core,pieces):
        check(len(names)==len(set(names))==26,'Incomplete cube identities')
        canonical={ '_'.join(v for v in (a,b,c) if v) for a in ('','pX','nX') for b in ('','pY','nY') for c in ('','pZ','nZ') if a or b or c};check({frozenset(n.split('_')) for n in names}=={frozenset(n.split('_')) for n in canonical},'Unexpected cubelet identities')
        self.select=torch.tensor(['nZ' in n.split('_') for n in names],device=core.device);check(self.select.sum()==9,'Not a 9-piece layer');self.r0,self.q0=body_geometry(core,pieces)
        # The authored joints and bodies share their origin for this USD; the
        # math also supports body-origin offsets that rotate along a legal orbit.
    def measure(self,core,pieces,target):
        r,q=body_geometry(core,pieces);self.measured_pos=r;self.measured_q=q;dq=mul(q,conj(self.q0));dq=torch.where(dq[...,3:]<0,-dq,dq)
        selected=dq[:,self.select];angles=wrap(2*D*torch.atan2(-selected[...,2],selected[...,3]));reference=angles[:,:1];angles=reference+wrap(angles-reference);median=angles.median(1).values
        # Exact fit for arbitrary legal axial twists; test the off-axis remainder.
        h=angles/D*.5;zero=h*0;axial=torch.stack((zero,zero,-h.sin(),h.cos()),-1);off=degrees(mul(conj(axial),selected)).amax(1)
        m=median/D*.5;z=m*0;orient=torch.stack((z,z,-m.sin(),m.cos()),-1);expected=self.r0.clone();expected[:,self.select]=rotate(orient[:,None],self.r0[:,self.select])
        residual=(r-expected).norm(dim=-1).amax(1);rest=degrees(dq[:,~self.select]).amax(1);span=angles.amax(1)-angles.amin(1);err=wrap(angles-target).abs().amax(1)
        structure=(off<=5)&(rest<=5)&(span<=5)&(residual<=.002);assembled=structure&(degrees(dq).amax(1)<=5)
        return dict(face_deg=median,structure_ok=structure,assembled=assembled,at_goal=structure&(err<=5),face_error_deg=err,piece_off_deg=off,other_rot_deg=rest,anchor_error_m=residual,spread_deg=span)

def actor_class(path):
    m=LANE.fullmatch(str(path));check(m,'Unknown contact actor path: '+str(path)); lane=int(m[1]);rel=m[2];name=rel.rsplit('/',1)[-1]
    if DIGIT.fullmatch(rel):return lane,1 # right finger
    if rel.startswith('/Robot/right_') or RIGHT_ARM.fullmatch(rel):return lane,2 # any other right-link touch
    if rel.startswith('/Robot/left_') or name=='Link7_L':return lane,3
    return lane,0

class TargetContacts:
    """Wrap the live instance reducer only, preserving original order and validation."""
    def __init__(self,s,names):
        self.base=s.reducer; self.n=s.n;dev=s.home.device
        records=[(v,lanes) for typ,v,lanes in s.contact_views if typ=='cube'];check(len(records)==1,'Missing raw cube contact view');self.v,lanes=records[0]
        paths=list(self.v.sensor_paths);check(len(paths)==len(lanes)==self.v.sensor_count==26*s.n,'Incomplete raw cube sensor coverage'); ix=[];tg=[];seen=set()
        for path,k in zip(paths,lanes):
            m=LANE.fullmatch(str(path));check(m and int(m[1])==int(k) and m[2].startswith('/Cube/'),'Unmapped raw cube sensor');name=m[2].split('/')[-1];check(name in names and (int(k),name) not in seen,'Unexpected or duplicate cube sensor');seen.add((int(k),name));ix.append(int(k));tg.append('nZ' in name.split('_'))
        self.sid=torch.arange(len(paths),device=dev);self.lanes=torch.tensor(ix,device=dev);self.target=torch.tensor(tg,device=dev);self.cache={};self.acc=torch.zeros((s.n,5),device=dev,dtype=torch.float64);self.count=0;self.serial=0;self.capacity=int(self.v.max_contact_data_count)
    def clear(self,s):self.base.clear(s);self.acc.zero_();self.count=0;self.serial+=1
    def validate(self):self.base.validate()
    def update(self,s):
        self.base.update(s);self.count+=1
        # Reuse the list of IDs already discovered and capacity-checked by the native reducer.
        if len(self.cache)!=len(self.base.cache):
            from importlib import import_module
            wp=import_module('warp');keys=sorted(set(self.base.cache)-set(self.cache));u=np.asarray(keys,dtype=np.int64);paths=s.sv_warp_contact.get_other_actor_paths_from_ids(wp.array(u.view(np.uint64),dtype=wp.uint64,device='cpu'));check(len(paths)==len(u),'Missing native actor identities')
            self.cache.update((k,actor_class(v)) for k,v in zip(keys,paths));ks=sorted(self.cache);self.keys=torch.tensor(ks,dtype=torch.long,device=self.acc.device);self.code=torch.tensor([self.cache[k] for k in ks],dtype=torch.long,device=self.acc.device)
        raw=self.v.get_raw_contact_data(1.);force,counts,starts,other=raw[0].reshape(-1),raw[4].reshape(-1).long(),raw[5].reshape(-1).long(),raw[6].reshape(-1).view(torch.int64)
        check(len(counts)==len(self.sid) and len(starts)==len(counts),'Invalid raw contact sensor packet');total=int(counts.sum());check((counts>=0).all() and total<self.capacity,'Contact capacity exceeded')
        if not total:return
        sensor=torch.repeat_interleave(self.sid,counts,output_size=total);offsets=counts.cumsum(0)-counts;indices=torch.arange(total,device=counts.device)+starts[sensor]-offsets[sensor];check((indices>=0).all() & (indices<len(force)).all() & (indices<len(other)).all(),'Invalid raw contact span')
        keys=other[indices];where=torch.searchsorted(self.keys,keys);check((where<len(self.keys)).all(),'Unresolved raw contact actor');check((self.keys[where]==keys).all(),'Unknown raw contact actor');meta=self.code[where];lane=self.lanes[sensor];check((lane==meta[:,0]).all(),'Cross-lane raw contact');imp=force[indices].abs().double();check(torch.isfinite(imp).all(),'Bad raw impulse');target=self.target[sensor];kind=meta[:,1];right=(kind==1)|(kind==2)
        # [right-digit target, wrong right touch, any right, left rest, left target]
        for col,mask in enumerate(((kind==1)&target,right&(~target|(kind!=1)),right,(kind==3)&~target,(kind==3)&target)):
            self.acc[:,col].scatter_add_(0,lane,imp*mask)

class PoseWindows:
    """Control-frame stability, independent of a transient final 1/240-s solver velocity.

    Ring holds world core and all nine target bodies in the core frame. For every
    trailing suffix, the diagonal of the componentwise range bounds *every* pair
    of positions. Hemisphere-aligned quaternion chord diameters similarly bound
    the principal rotation between any pair; excursions returning to start count.
    The two gates use their own suffix (30 vs 90 consecutive measured frames).
    """
    def __init__(self,n,dev):
        self.history=torch.zeros((n,90,10,7),device=dev);self.index=0;self.count=torch.zeros(n,device=dev,dtype=torch.long);self.order=torch.arange(90,device=dev)
    def measure(self,active,c,pos,rot):
        check(pos.shape==(len(c),9,3) and rot.shape==(len(c),9,4),'Need all nine actual nZ poses')
        new=torch.cat((c[:,None],torch.cat((pos,rot),dim=-1)),dim=1).float()
        self.count=torch.where(active,(self.count+1).clamp(max=90),0)
        self.history[:,self.index]=new;ids=(self.index-self.order)%90;self.index=(self.index+1)%90
        # Keep per-step working memory bounded even for a large campaign wave.
        groups=[]
        for h,cnt in zip(self.history.split(16),self.count.split(16)):
            h=h[:,ids];qs=h[...,3:];qs=torch.where((qs*qs[:,:1]).sum(-1,keepdim=True)<0,-qs,qs);h=torch.cat((h[...,:3],qs),-1)
            delta=h.cummax(1).values-h.cummin(1).values;dp=delta[...,:3].norm(dim=-1);da=4*D*torch.asin((delta[...,3:].norm(dim=-1)*.5).clamp(0,1));cp=dp[:,:,0];cq=da[:,:,0];pp=dp[:,:,1:].amax(2);pq=da[:,:,1:].amax(2)
            # Tolerances in metres/degrees for the complete span, not per-step changes.
            steady=(cp<=.008)&(cq<=3.)&(pp<=.002)&(pq<=1.)&(self.order[None,:]<cnt[:,None]);result={};row=torch.arange(len(h),device=h.device)
            for stage,k in [('natural',30),('retreat',90)]:
                n=cnt.clamp(max=k);idx=(n-1).clamp_min(0);suffix=steady[:,:k].long().sum(1);result[stage+'_motion_window_frames']=n;result[stage+'_steady_suffix_frames']=suffix;result[stage+'_motion_ok']=(suffix>=k)
                for name,v in [('core_span_m',cp),('core_span_deg',cq),('piece_span_m',pp),('piece_span_deg',pq)]:result[stage+'_'+name]=torch.where(n>0,v[row,idx],0)
            groups.append(result)
        return {k:torch.cat([g[k] for g in groups]) for k in groups[0]}

class Sequence:
    """State machine: a hold on a naturally changed pose, direct finger drive, retreat then 3 s hold."""
    def __init__(self,n,dev,target=15.):
        self.target=target;self.natural_count=torch.zeros(n,dtype=torch.long,device=dev);self.hold_count=self.natural_count.clone()
        self.ready=self.natural_count.bool();self.has_contact=self.ready.clone();self.turn=self.ready.clone();self.goal_touch=self.ready.clone();self.success=self.ready.clone()
        self.last_direct=self.ready.clone();self.last_angle=self.natural_count.double();self.driven=self.natural_count.double();self.contact_angle=self.natural_count.double();self.close_dist=self.contact_angle.clone();self.touch_peak=self.contact_angle.clone();self.peak_qualified_face=self.contact_angle.clone();self.start_rel=torch.zeros((n,3),device=dev,dtype=torch.float64);self.minrel=self.start_rel.clone();self.maxrel=self.start_rel.clone();self.nlo=self.start_rel.clone();self.nhi=self.start_rel.clone()
    @staticmethod
    def steady(count,flag,pos,lo,hi):
        new=(count==0)[:,None];low=torch.where(new,pos,torch.minimum(pos,lo));high=torch.where(new,pos,torch.maximum(pos,hi));consistent=(high-low).norm(dim=-1)<=.01
        # Conservative bound on any pair of palm-relative positions in this episode window.
        valid=flag&consistent;return torch.where(valid,count+1,0),low,high
    def update(self,post,support,natural,post_quality,touch,wrong,any_right,finger_dist,retract,stable,rest_support,goal_geom,palm_rel,quality,natural_suffix,retreat_suffix):
        g=goal_geom;angle=g['face_deg'];good=post&support&g['structure_ok']&~wrong
        qualified=good&g['assembled']&natural&~any_right&stable
        self.natural_count,self.nlo,self.nhi=self.steady(self.natural_count,qualified,palm_rel,self.nlo,self.nhi)
        natural_frames=torch.minimum(self.natural_count,natural_suffix);previously_ready=self.ready.clone();self.ready|=natural_frames>=30
        direct=good&previously_ready&rest_support&touch&~retract
        first=direct&~self.has_contact&g['assembled'];self.contact_angle=torch.where(first,angle,self.contact_angle);self.has_contact|=first
        direct &= self.has_contact
        self.close_dist=torch.where(first,finger_dist,torch.where(direct,torch.minimum(self.close_dist,finger_dist),self.close_dist))
        advance=angle-self.contact_angle
        self.driven+=torch.where(direct&self.last_direct,wrap(angle-self.last_angle),0);self.last_direct=direct.clone();self.last_angle=angle.clone()
        credited=torch.minimum(advance,self.driven);self.touch_peak=torch.where(direct,torch.maximum(self.touch_peak,credited),self.touch_peak)
        self.turn|=direct&(credited>=2.)
        # The five-degree per-cubelet geometric tolerance is not a lowered skill target.
        reached=direct&self.turn&(credited>=self.target-1e-7)&(angle>=self.target-1e-7)&g['at_goal']
        self.goal_touch|=reached;self.peak_qualified_face=torch.where(reached,torch.maximum(self.peak_qualified_face,angle),self.peak_qualified_face)
        retreat_floor=torch.maximum(torch.full_like(angle,self.target),self.peak_qualified_face)-2.
        valid=good&self.goal_touch&retract&~any_right&rest_support&g['at_goal']&(advance>=self.target-2.-1e-7)&(angle>=retreat_floor-1e-7)&post_quality&stable&(finger_dist-self.close_dist>=.020)
        self.hold_count,self.minrel,self.maxrel=self.steady(self.hold_count,valid,palm_rel,self.minrel,self.maxrel);held_frames=torch.minimum(self.hold_count,retreat_suffix);self.success|=held_frames>=90
        allowed=torch.where(previously_ready&self.has_contact,g['structure_ok']&(angle>=-5)&(angle<=self.target+5),g['assembled'])
        return dict(natural_ready=self.ready.clone(),natural_now=qualified,turn_started=self.turn.clone(),success=self.success.clone(),allowed_layer_state=allowed,progress=(self.touch_peak/self.target).clamp(0,1),posture_quality=quality,hold_fraction=(held_frames/90.).clamp(0,1),natural_streak=natural_frames.clone(),hold_streak=held_frames.clone(),contact_net_deg=advance,measured_contact_rotation_deg=self.driven.clone(),contact_goal_reached=self.goal_touch.clone(),peak_qualified_face_deg=self.peak_qualified_face.clone(),retreat_floor_deg=retreat_floor,retreat_drop_deg=(self.peak_qualified_face-angle).clamp_min(0),actual_retraction_m=torch.where(self.has_contact,finger_dist-self.close_dist,0))

class NaturalFace90Observer:
    def __init__(self,scene,target_deg=15.):
        from assets import fixed_frame
        s=self.s=scene;check(target_deg in (15.,90.) and abs(s.dt*s.dec-1/30)<1e-8,'Unsupported layer target/control rate');self.target=float(target_deg);self.dev=s.home.device
        self.names=tuple(k for k in s.views if k!='Core');c,p=self.cube();self.layer=Layer(self.names,c,p);self.machine=Sequence(s.n,self.dev,self.target);self.motion=PoseWindows(s.n,self.dev);self.empty_motion=None
        names=list(s.robot.shared_metatype.link_names);self.elbows=[names.index(f'Link4_{a}') for a in 'LR'];self.wrists=[names.index(f'Link7_{a}') for a in 'LR'];self.fingers=[names.index(f'right_{f}_DIP_link') for f in ('thumb','index','middle','ring','little')];self.palms=[]
        for side in ('left','right'):
            body,off,q=fixed_frame(s.stage,'/World/envs/env_0000/Robot',side+'_palm_link');self.palms.append((names.index(body),torch.tensor(off,dtype=torch.float64,device=self.dev),unit(torch.tensor([*q[1:],q[0]],dtype=torch.float64,device=self.dev))))
        self.j6=s.names.index('Joint6_L');self.prev=None;self.last_step=-1;self.serial=0
        self.r=TargetContacts(s,self.names);s.reducer=self.r
    def cube(self):
        return self.s.views['Core'].get_transforms().double(),torch.stack([self.s.views[n].get_transforms() for n in self.names],1).double()
    def observe(self,active,post_prefix,safe,retract,step):
        s=self.s;check(step==self.last_step+1 and self.r.serial==self.serial+1 and self.r.count==s.dec,'Missing, duplicated or incomplete physical control step');self.last_step=step;self.serial=self.r.serial
        for m in (active,post_prefix,safe,retract):check(m.shape==(s.n,) and m.dtype==torch.bool and m.device==self.dev,'Bad per-lane mask')
        c,p=self.cube();g=self.layer.measure(c,p,self.target);T=s.robot.get_link_transforms().double();check(torch.isfinite(T).all(),'Nonfinite articulated link pose');palms=[]
        for ix,off,rot in self.palms:
            b=T[:,ix];q=unit(b[:,3:]);palms.append((b[:,:3]+rotate(q,off),mul(q,rot)))
        (left,lq),(right,rq)=palms; rel=rotate(conj(lq),c[:,:3]-left);nvec=left.new_tensor([1.,0.,0.]);direction=torch.nn.functional.normalize(right-left,dim=1);ln=rotate(lq,nvec);rn=rotate(rq,nvec)
        facing=(ln*direction).sum(1);rface=-(rn*direction).sum(1);dot=(ln*rn).sum(1);en=T[:,self.elbows,:3];wr=T[:,self.wrists,:3]
        y0=s.offset[:,1];left_y=left[:,1]-y0;centre_y=c[:,1]-y0
        if self.prev is None:self.prev=(left_y.clone(),lq.clone(),centre_y.clone())
        a,q0,c0=self.prev;prefix=active&~post_prefix;a=torch.where(prefix,left_y,a);q0=torch.where(prefix[:,None],lq,q0);c0=torch.where(prefix,centre_y,c0);self.prev=(a,q0,c0)
        shift=a.abs()-left_y.abs();core_shift=c0.abs()-centre_y.abs();tilt=degrees(mul(lq,conj(q0)));elbows=(en[:,0,1]-wr[:,0,1]>=.020)&(wr[:,1,1]-en[:,1,1]>=.020);clear=(left[:,1]-right[:,1]>=.03)&(en[:,0,1]-en[:,1,1]>=.04)
        posture=(facing>=.40)&(rface>=.40)&(dot<=.10)&elbows&clear; change=(shift>=.018)&(core_shift>=.010)&(tilt>=4.)&(tilt<=35.);natural=posture&change
        quality=torch.stack(((shift/.018).clamp(0,1),(tilt/4.).clamp(0,1),(facing/.4).clamp(0,1),(rface/.4).clamp(0,1),elbows.double(),clear.double()),1).mean(1)
        imp=self.r.acc;threshold=float(s.cfg['success']['contact_impulse_min']);touch=imp[:,0]>threshold;wrong=imp[:,1]>1e-7;any_right=imp[:,2]>1e-7;rest_support=imp[:,3]>threshold
        # Contact membership requires the original per-sensor impulses as well; no separation test creates contact.
        table=s.contacts[:,12]>=1e-5;height=c[:,2]-s.offset[:,2]-s.cfg['table_z']-.045;support=active&safe&~table&(height>=.02)&(s.hand_impulse[:,0]>threshold)
        vel=s.views['Core'].get_velocities().double();ang=torch.stack([s.views[n].get_velocities()[:,3:6] for n in self.names if 'nZ' in n.split('_')],1).double();check(torch.isfinite(vel).all() & torch.isfinite(ang).all(),'Nonfinite measured body speed')
        spin=(ang-vel[:,None,3:6]).norm(dim=-1).amax(1);stable=height>=.08
        # Prefix poses cannot fill either post-stage window. Inactive lanes reset their counter.
        if self.empty_motion is None or (active&post_prefix).any():
            local=c.clone();local[:,:3]-=s.offset;motion=self.motion.measure(active&post_prefix,local,self.layer.measured_pos[:,self.layer.select],self.layer.measured_q[:,self.layer.select]);
            if self.empty_motion is None:self.empty_motion={k:torch.zeros_like(v) for k,v in motion.items()}
        else:motion=self.empty_motion;self.motion.count.zero_()
        dist=(T[:,self.fingers,:3]-c[:,None,:3]).norm(dim=-1).amin(1);m=self.machine.update(active&post_prefix,support,natural,posture,touch,wrong,any_right,dist,retract,stable,rest_support,g,rel,quality,motion['natural_steady_suffix_frames'],motion['retreat_steady_suffix_frames'])
        qj=s.robot.get_dof_positions()[:,s.ids[self.j6]];check(torch.isfinite(qj).all(),'Invalid measured joint angle')
        return dict(**g,**m,**motion,core_linear_speed_mps=vel[:,:3].norm(dim=1),core_angular_speed_rad_s=vel[:,3:6].norm(dim=1),right_contact=touch,wrong_contact=wrong,support=support,target_impulse=imp[:,0].clone(),other_impulse=imp[:,1].clone(),right_total_impulse=imp[:,2].clone(),left_static_impulse=imp[:,3].clone(),left_target_impulse=imp[:,4].clone(),normalized_world_palm_turn_deg=tilt,palm_toward_midline_m=shift,core_toward_midline_m=core_shift,left_faces_right=facing,right_faces_left=rface,palm_normal_dot=dot,elbows_out=elbows,uncrossed=clear,joint6_measured_deg=qj*D,selected_relative_angular_speed=spin)
