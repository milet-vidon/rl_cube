"""Real 90 mm GPU PhysX rollouts: collect projected teacher labels OR evaluate a student from reset.

A validation process produces metrics only. Source rollouts can supply a USD
scene, never supervision. All training rows come from a fresh online query made
at the measured state just before the 56D command was sent to PhysX.
"""
import argparse
import json
import math
from pathlib import Path
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SCHEMA='contact_hold90_feedback_v1'
LABEL='projected_90mm_feedback_teacher'
FILES=('contact_hold90_cuda_sample.py','contact_hold90_cuda_domain.py','contact_hold_cuda_scene.py','contact_hold_cuda_scene_reduce.py','contact_hold_cuda_teacher.py','contact_hold_cuda_teacher_math.py','contact_hold_cuda_teacher_turn.py','contact_hold_cuda_ik.py','contact_hold_cuda_ik_kernel.py','phase_contact_policy.py','contact_hold_policy.py','joint_safety.py','kinematics.py','contact_hold_size_geometry.py','contact_hold_simple_rotation.py','contact_hold_physical_metrics.py')


def code_hashes():
    from contact_hold_dataset import sha
    return {n:sha(ROOT/n) for n in FILES}


def partitions(seeds,role,explicit=None):
    if role not in ('collection','validate'):raise ValueError('Unknown role')
    lo=10_000_000 if role=='collection' else 20_000_000
    if not 2<=len(seeds)<=2048 or len(set(seeds))!=len(seeds) or any(type(s) is not int or not lo<=s<lo+10_000_000 for s in seeds):
        raise ValueError('Use unique seeds in separated 90 mm train/evaluation ranges')
    if role=='validate':
        if explicit:raise ValueError('The entire validation panel is excluded from the loss')
        return [],[]
    # Entire lanes go to one side of the supervised split. Preserve the default
    # split across rounds/waves even if the effective wave width changes.
    val=[s for s in seeds if s%4==3] if explicit is None else list(explicit)
    if not set(val)<=set(seeds) or len(set(val))!=len(val):raise ValueError('Invalid loss-validation seed selection')
    if len(seeds)<4 and explicit is None and not val:val=[seeds[-1]]
    tr=[s for s in seeds if s not in set(val)]
    if not tr or not val:raise ValueError('Use at least one entire fit and loss-validation lane in a wave')
    return tr,val


def verify_prerequisites():
    from stage_gate import require_staged_gate
    from side_gate import require_side_gate
    from opposition_gate import require_opposition_gate
    from evidence_paths import local_path,raw_sha
    for gate in (require_staged_gate,require_side_gate,require_opposition_gate):gate()
    cert=json.loads((ROOT/'wrist_direction_gate_certificate.json').read_bytes())
    if not cert['passed'] or cert['independent_accepted']<5:raise ValueError('Original wrist reference missing')
    for n,h in (cert['source_sha256']|cert['assets_sha256']).items():
        if raw_sha(local_path(n,ROOT))!=h:raise ValueError('Changed wrist reference')
    if raw_sha(ROOT/'wrist_direction_protocol_v1.json')!=cert['protocol_sha256']:raise ValueError('Wrist protocol changed')
    for t in cert['trials']:
        d=local_path(t['run'],ROOT)
        for name,key in [('config.json','config_sha256'),('runtime_config.json','runtime_config_sha256'),('asset_contract.json','asset_contract_sha256'),(f"trajectory_{t['seed']}.json",'trajectory_sha256')]:
            if raw_sha(d/name)!=t[key]:raise ValueError('Changed prior reference '+str(d/name))


class PhaseBank:
    """Independent fixed-capacity uniform-priority reservoirs for each phase and split."""
    def __init__(self,cap,seed):
        self.cap=cap;self.rng=np.random.default_rng(seed);self.parts={};self.seen={};self.buffered={}
    def _compact(self,key):
        v=self.parts[key];a={k:np.concatenate([p[k] for p in v]) for k in v[0]}
        if len(a['priority'])>self.cap:
            ix=np.argpartition(a['priority'],self.cap-1)[:self.cap];a={k:t[ix] for k,t in a.items()}
        self.parts[key]=[a];self.buffered[key]=len(a['priority'])
    def add(self,a):
        group=a['phase']*2+a['val'].astype(np.int64)
        for k in np.unique(group):
            ix=np.flatnonzero(group==k);k=int(k);parts=self.parts.setdefault(k,[])
            parts.append(dict({name:v[ix] for name,v in a.items()},priority=self.rng.random(len(ix))))
            self.seen[k]=self.seen.get(k,0)+len(ix);self.buffered[k]=self.buffered.get(k,0)+len(ix)
            if self.buffered[k]>=self.cap*2:self._compact(k)
    def finish(self):
        if not self.parts:raise ValueError('No eligible 90 mm teacher queries')
        for k in self.parts:self._compact(k)
        blocks=[self.parts[k][0] for k in sorted(self.parts)]
        return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0] if k!='priority'}


class HoldMeter:
    """At 30 Hz test every sliding 30-frame interval against all pairwise palm distances."""
    def __init__(self,n,device):
        self.xyz=torch.zeros((n,30,3),device=device,dtype=torch.float64);self.valid=torch.zeros((n,30),device=device,dtype=torch.bool)
        self.max_sq=torch.zeros((n,30),device=device,dtype=torch.float64);self.ever=self.valid[:,0].clone();self.end=self.ever.clone();self.best=torch.zeros(n,device=device,dtype=torch.long)
    def update(self,step,acted,support,xyz):
        slot=step%30
        # Associate each pair with its earlier sample, keeping the largest
        # distance to every later sample still in the rolling window.
        sq=(self.xyz-xyz[:,None,:]).square().sum(2)
        self.max_sq=torch.maximum(self.max_sq,sq);self.max_sq[:,slot]=0.
        self.xyz[:,slot]=xyz;self.valid[:,slot]=support
        now=self.valid.all(1)&(self.max_sq.amax(1)<=.01**2+1e-14)
        ages=(slot-torch.arange(30,device=xyz.device))%30
        bad=(~self.valid)|(self.max_sq>.01**2+1e-14)
        suffix=torch.where(bad,ages[None,:],30).amin(1).clamp(max=min(step+1,30))
        self.best=torch.where(acted,torch.maximum(self.best,suffix),self.best)
        self.ever|=acted&now;self.end=torch.where(acted,now,self.end)


def local_palm_reader(e):
    from assets import fixed_frame
    from contact_hold_cuda_teacher_math import quat_matrix,mv
    body,offset,quat=fixed_frame(e.stage,'/World/envs/env_0000/Robot','left_palm_link')
    ix=list(e.robot.shared_metatype.link_names).index(body)
    offset=torch.tensor(offset,device=e.offset.device,dtype=torch.float64)
    qr=torch.tensor([quat[1],quat[2],quat[3],quat[0]],device=e.offset.device,dtype=torch.float64)
    frame=quat_matrix(qr)
    def measured(core):
        body_pose=e.robot.get_link_transforms()[:,ix].double();r=quat_matrix(body_pose[:,3:7]);world=core[:,:3].double()+e.offset.double()
        return mv((r@frame).mT,world-body_pose[:,:3]-mv(r,offset))
    return measured


def whole_score(trace,step_count):
    from contact_hold_simple_rotation import summarize_rotation
    if not step_count:return None
    # columns: supported, xyz in measured palm, xyzw, height, table, hand, drop.
    a=trace[:step_count];support=a[:,0].astype(bool);starts=[];left=0
    for i in range(len(a)):
        if not support[i]:left=i+1
        elif left<i:
            d=a[left:i,1:4]-a[i,1:4];bad=np.flatnonzero(np.einsum('ij,ij->i',d,d)>.01**2+1e-14)
            if len(bad):left+=int(bad[-1])+1
        starts.append(left)
    q=a[:,[7,4,5,6]];q=q/np.linalg.norm(q,axis=1)[:,None]
    ev=dict(start_by_end=starts,support=support.tolist(),quaternions=q,hz=30,hold_frames=30,post_frames=15,confirm_frames=3)
    # summarize_rotation consumes only step ids and the final measured state
    # when provided with a complete reconstructed window evidence sequence.
    rows=[{'step':i} for i in range(len(a))];last=rows[-1];last.update(height_m=float(a[-1,8]),
        cube_table_contact=bool(a[-1,9]>=1e-5),table_impulse=float(a[-1,9]),contact_impulse=[float(a[-1,10]),0.],drop=bool(a[-1,11]))
    return summarize_rotation(rows,{'success':{'contact_impulse_min':float(a[-1,12])}},evidence=ev)


def sample(args):
    from contact_hold90_cuda_domain import open_scene,make_plan,make_teacher,FAILURES90,ASSET_SHA,GRASP_SHA
    from contact_hold_loop_data import read,write,checkpoint
    from contact_hold_dataset import sha
    from phase_contact_policy import PHASES
    from contact_hold_policy import restored
    from joint_safety import project_target
    import time
    verify_prerequisites()
    p=read(args.plan);seeds=p['seeds'];train_seeds,val_seeds=partitions(seeds,args.role,p.get('loss_val_seeds'))
    if 'role' in p and p['role']!=args.role:raise ValueError('Role mismatch')
    if not 2<=args.steps<=3500:raise ValueError('Expected horizon 2..3500')
    parent=Path(p['parent']).resolve();source=Path(p['scene_source']).resolve();grasp=Path(p['grasp']).resolve();n=len(seeds)
    code=code_hashes();out=args.output;out.mkdir(parents=True,exist_ok=False)
    if checkpoint(parent)!=p['parent_sha'] or any(Path(k).name!=k or sha(ROOT/k)!=v for k,v in p.get('code',{}).items()):raise ValueError('Code or parent checkpoint mismatch')
    ck=torch.load(parent/'best.pt',weights_only=True,map_location='cpu');model=restored(ck).cuda().eval();meta=ck['metadata'];ids=meta['output_indices']
    for name,h in meta['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed checkpoint code: '+name)
    if set(seeds)&set(meta.get('training_seeds',[])) and args.role=='validate':raise ValueError('Validation seed previously trained')
    asset=read(source/'cube_asset.json')['source_sha256'];grasp_sha=sha(grasp)
    if asset!=ASSET_SHA or grasp_sha!=GRASP_SHA:
        raise ValueError('Scene asset is not the frozen size asset')
    if any(k in p and p[k]!=v for k,v in [('asset_sha256',asset),('grasp_sha256',grasp_sha),('scene_manifest_sha256',sha(source/'artifact_manifest.json'))]):raise ValueError('Domain pin mismatch')
    e=open_scene(source,n,grasp)
    if e.names!=meta['joint_names'] or list(ids)!=[i for i,name in enumerate(e.names) if name.startswith('left_') or name.endswith('_L')]:raise ValueError('Left 28 action mapping mismatch')
    e.reset(seeds);read_palm=local_palm_reader(e)
    if not (e.contacts[:,12]>0).all():raise ValueError('Missing reset table impulse')
    plan=make_plan(e.core(),grasp);teacher=make_teacher(plan,e.cfg,meta['sources'][0]['candidate'],grasp,e.batch_gain)
    dt=e.dec*e.dt
    if abs(dt*30-1)>1e-8 or e.cfg['physics_hz']!=240:raise ValueError('Wrong physical rate')
    target=e.home.clone();dev=target.device;lane=torch.arange(n,device=dev);seed_t=torch.tensor(seeds,device=dev)
    active=plan.accepted.clone();ends=torch.where(active,0,-1).long();prev=torch.full_like(lane,-1);start=torch.zeros_like(lane)
    arm=[i for i,k in enumerate(e.names) if k.startswith('Joint')];phase_steps=torch.zeros(n,len(PHASES),device=dev,dtype=torch.int32)
    # Some teacher-forced lanes reach later states for initial supervision; DAgger
    # lanes execute the current student from reset or from squeeze/lift onward.
    behavior=torch.tensor((0,4,5,100),device=dev)[(seed_t+seed_t//4)%4] if args.role=='collection' else torch.zeros_like(lane)
    val=torch.tensor([s in set(val_seeds) for s in seeds],device=dev);cap=torch.minimum(e.step,e.vel*dt)
    success=e.cfg['success'];min_contact=success['contact_impulse_min'];init_xy=target.new_tensor(e.cfg['cube_xyz'][:2]);feedback=e.contacts.clone();feedback[:,:10]=0
    bank=PhaseBank(int(p.get('rows_per_phase_split',4096)),seeds[0]);meter=HoldMeter(n,dev)
    peak=target.new_full((n,),-float('inf'));supported_ever=torch.zeros_like(active);returned=supported_ever.clone();steps=torch.zeros_like(lane)
    trace=[];t0=time.monotonic();table_last=torch.zeros_like(active);teacher_steps=torch.zeros_like(lane);env_hold=torch.zeros_like(lane);supported_steps=torch.zeros_like(lane)
    print('GPU90_READY',n,'role',args.role,'IK accepted',int(active.sum()),flush=True)
    for step in range(args.steps):
        q=e.robot.get_dof_positions()[:,e.ids];v=e.robot.get_dof_velocities()[:,e.ids];core=e.core();vel=e.views['Core'].get_velocities()
        raw,ph,fail,accepted=teacher.action(active,q,v,core,target,feedback,vel,e.hand_impulse)
        ends=torch.where(active&(fail!=0),fail,ends);active&=fail==0
        if not active.any():break
        acted=active.clone();ph=torch.where(acted,ph,0);raw=torch.where(acted[:,None],raw,target)
        start=torch.where(acted&(ph!=prev),step,start);prev=torch.where(acted,ph,prev);phase_steps[lane,ph]+=acted.int()
        feat=torch.log1p(e.feature_contacts.to(q.dtype)/.001) if step else q.new_zeros(n,10)
        x=torch.cat((q,v,target,core[:,:3],core[:,[6,3,4,5]],feat,torch.nn.functional.one_hot(ph,len(PHASES)).to(q.dtype),((step-start)/300.).clamp(max=1)[:,None]),1)
        if x.shape!=(n,198) or not x.isfinite().all():raise ValueError('Invalid actual 90 mm features')
        label,hit=project_target(target,raw,e.limits,e.step,e.vel,dt);policy=model(x);use=acted&(ph>=behavior)
        req=raw.clone();req[:,ids]=torch.where(use[:,None],target[:,ids]+(policy-target[:,ids]).clamp(-cap[:,ids],cap[:,ids]),raw[:,ids]);cmd,soft=project_target(target,req,e.limits,e.step,e.vel,dt)
        e.step_physics(cmd);target=cmd;post=e.robot.get_dof_positions()[:,e.ids];postcore=e.core();feedback=e.contacts
        poses=torch.stack([b.get_transforms() for b in e.views.values()],1);cube_w=postcore[:,:3]+e.offset
        if not (post.isfinite().all() & postcore.isfinite().all() & poses.isfinite().all()):raise ValueError('Nonfinite PhysX response')
        dots=(poses[:,:,3:7]*postcore[:,None,3:7]).sum(-1).abs().clamp(max=1.)
        dis=(poses[:,:,:3]-cube_w[:,None,:]).norm(dim=-1).amax(1);deformed=dots.amin(1)<math.cos(math.radians(15))
        assembled=(dots.amin(1)>=math.cos(math.radians(2.5)))&(dis<=.002)
        hard=((post<e.hard[...,0]-1e-4)|(post>e.hard[...,1]+1e-4)).any(1)
        softarm=((post[:,arm]<e.limits[:,arm,0]-1e-4)|(post[:,arm]>e.limits[:,arm,1]+1e-4)|soft[:,arm]).any(1)
        height=postcore[:,2]-e.cfg['table_z']-.045;drop=(postcore[:,2]<e.cfg['table_z']-.12)|((postcore[:,:2]-init_xy).norm(dim=1)>.45)
        table=e.contacts[:,12]>=1e-5;bad_self=e.contacts[:,10]>1e-6;bad_table=e.contacts[:,11]>1e-6
        unsafe=hard|softarm|drop|deformed|bad_self|bad_table
        supported=acted&(height>=.02)&~table&(e.hand_impulse[:,0]>min_contact)&~unsafe&assembled
        local=read_palm(postcore);meter.update(step,acted,supported,local)
        returned|=acted&supported_ever&table;supported_ever|=supported;supported_steps+=supported.long();steps+=acted.long();teacher_steps+=(acted&~use).long()
        peak=torch.where(acted,torch.maximum(peak,height),peak);table_last=torch.where(acted,table,table_last)
        if args.role=='validate':trace.append(torch.cat((supported[:,None],local,postcore[:,3:7],height[:,None],e.contacts[:,12:13],e.hand_impulse[:,:1],drop[:,None],torch.full_like(height[:,None],min_contact)),1).cpu().numpy())
        postvelocity=e.views['Core'].get_velocities()
        if not torch.isfinite(postvelocity).all():raise ValueError('Nonfinite measured cube velocity')
        stable=(e.hand_impulse>min_contact).all(1)&(height>=success['lift_m'])&(~table)&(~drop)&(postvelocity[:,:3].norm(dim=1)<success['linear_speed_max'])&(postvelocity[:,3:].norm(dim=1)<success['angular_speed_max'])
        env_hold=torch.where(acted&stable,env_hold+1,0)
        for flag,reason in ((hard,-2),(softarm,-3),(drop,-4),(deformed,-5),(bad_self,-6),(bad_table,-7),(accepted,-8),(env_hold>=round(success['hold_seconds']*30),-9)):
            ends=torch.where(active&flag,reason,ends);active&=~flag
        if step+1==args.steps:ends=torch.where(active,-10,ends);active.zero_()
        if args.role=='collection' and step>0:
            y=label[:,ids];mask=~hit[:,ids]&~soft[:,ids]&(y>=model.lower-1e-6)&(y<=model.upper+1e-6)
            ii=lane[active&mask.any(1)]
            if len(ii):bank.add({k:t.cpu().numpy() for k,t in dict(x=x[ii],y=y[ii],mask=mask[ii],val=val[ii],seed=seed_t[ii],lane=ii,step=torch.full_like(ii,step),phase=ph[ii],teacher_executed=(~use[ii])).items()})
        if step%120==0:print('GPU90_STEP',step,'active',int(active.sum()),'hold',int(meter.ever.sum()),flush=True)
        if not active.any():break
    torch.cuda.synchronize();elapsed=time.monotonic()-t0
    reason={**FAILURES90,-1:'reset_planner_rejected',-2:'hard_joint_limit',-3:'soft_arm_limit',-4:'cube_drop',-5:'layer_deviation',-6:'robot_self_contact',-7:'robot_table_contact',-8:'teacher_wrist_endpoint',-9:'environment_done',-10:'step_limit'}
    end=ends.tolist();ns=steps.tolist();hi=peak.tolist();best=meter.best.tolist();ever=meter.ever.tolist();ending=meter.end.tolist();ret=returned.tolist();tab=table_last.tolist();phasehist=phase_steps.tolist();tsteps=teacher_steps.tolist();sup=supported_steps.tolist()
    hist=np.asarray(trace) if trace else None;each=[]
    for i,s in enumerate(seeds):
        whole=whole_score(hist[:,i],ns[i]) if hist is not None and ns[i] else None
        if whole is not None and (whole['ever_hold']!=ever[i] or whole['ending_hold']!=ending[i]):raise ValueError('GPU hold metric failed CPU replay')
        each.append(dict(lane=i,seed=s,end_reason=reason[end[i]],steps=ns[i],max_height_m=hi[i] if ns[i] else None,ever_hold=ever[i],ending_hold=ending[i],
            returned_to_table_after_lift=ret[i],final_table_contact=tab[i] if ns[i] else None,ever_whole_turn=whole['ever_hold_and_turn'] if whole else (False if args.role=='validate' else None),
            best_controlled_hold_s=whole['best_controlled_hold']['seconds'] if whole else best[i]/30,hold_duration_capped_at_1s=(args.role=='collection'),physical_rotation=whole,teacher_executed_left_steps=tsteps[i],student_executed_left_steps=ns[i]-tsteps[i],supported_steps=sup[i],
            phase_steps={name:ct for name,ct in zip(PHASES,phasehist[i]) if ct}))
    wc=sum(v['ever_whole_turn'] is True for v in each) if args.role=='validate' else None
    report=dict(schema=SCHEMA,role=args.role,parent_sha=p['parent_sha'],checkpoint=p['parent_sha'],plan_sha256=sha(args.plan),cube_edge_m=.09,asset_sha256=asset,grasp_sha256=grasp_sha,
        scene_manifest_sha256=sha(source/'artifact_manifest.json'),scene_sha256=sha(source/'dynamic_training_scene.usda'),code_sha256=code,num_envs=n,seeds=seeds,
        train_seeds=train_seeds,loss_val_seeds=val_seeds,eval_seeds=seeds if args.role=='validate' else [],teacher_label_source=LABEL if args.role=='collection' else None,
        single_face_trained=False,single_face_commanded=False,eligible_success_demonstration=False,physics_device='cuda:0',action_dim=56,control_hz=30,physics_hz=240,
        solver=e.readback,domain_receipt=e.domain_receipt,cross_lane_contacts=e.cross,student_start_phase=behavior.tolist(),requested_steps=args.steps,elapsed_rollout_s=elapsed,per_lane=each,
        metrics=dict(hold_count=sum(ever),hold_rate=sum(ever)/n,ending_hold_count=sum(ending),returned_to_table_count=sum(ret),whole_turn_count=wc,
            whole_turn_rate=wc/n if wc is not None else None,mean_max_height_m=float(np.mean([h for h,ct in zip(hi,ns) if ct])) if any(ns) else None),
        physical_hold_definition='30 consecutive safe assembled frames, >=2cm above tabletop, left measured support, no table impulse and all pairwise palm-relative center distances <=1cm',samples=0)
    if args.role=='collection':
        a=bank.finish();np.savez_compressed(out/'batch.npz',**a)
        report.update(data_sha256=sha(out/'batch.npz'),samples=len(a['x']),phase_split_seen=bank.seen,
            phase_split_key='2*phase+loss_validation_bit',label_references='retargeted live teacher feedback projected from actual previous targets; not student outputs')
    elif any(tsteps):raise ValueError('Teacher executed a left action in validation')
    write(out/'report.json',report);print('GPU90_COMPLETE',args.role,n,report['metrics'],'rows',report['samples'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--role',choices=('collection','validate'),required=True);parser.add_argument('--steps',type=int,default=3500);a=parser.parse_args()
    from isaaclab.app import AppLauncher
    from isaac_compat import close_application
    app=AppLauncher(headless=True,device='cuda:0').app
    try:
        with torch.no_grad():sample(a)
    except BaseException:traceback.print_exc();raise
    finally:close_application(app)


if __name__=='__main__':main()
