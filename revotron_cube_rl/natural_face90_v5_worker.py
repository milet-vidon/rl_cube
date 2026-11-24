"""Simulated 90 mm R27 pickup followed by a phase-gated, contact-driven wrist/layer policy.

No pose writes after the single normal reset. The R27 teacher schedule and student
left commands remain frozen. Post-prefix actions are always passed through the
same canonical joint projector and measured after a real PhysX step.
"""
from pathlib import Path
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import time

ROOT = Path('/home/brainco/rl_cube/github_project/revotron_cube_rl')
CAMPAIGN = ROOT/'runs/contact_hold90_long_v2'
PARENT = CAMPAIGN/'round_027/fit'
ARCHIVE_PLAN = PARENT.parent/'validate/wave_0000_0032/plan.json'
PARENT_HASH = '7144453d2425fce0f5d4293748b4bf71cb8afbca2ec6917624842e16b2b656a7'
PLAN_HASH = '524598f6cc71bf3fb72f17c026f6f9e87f292a87eefeadd68b0ebbe59cbbabf1'
REPORT_HASH = '39b7641e2372f5d3e6d236049e588c133b6ce31eeb863a74016a6f94bfeaab9d'
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(p,obj):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def require(test,msg):
    if not test:raise ValueError(msg)
def inputs(req):
    from natural_face90_v5_clearance import verify_geometry
    verify_geometry()
    from contact_hold90_cuda_sample import verify_prerequisites,code_hashes
    from contact_hold90_cuda_domain import verify_domain
    from contact_hold_loop_data import checkpoint
    from contact_hold_policy import restored
    import torch
    require(req.get('role') in ('train','validate','smoke'),'Unknown role')
    seeds=req['seeds'];require(isinstance(seeds,list) and 2<=len(seeds)<=256
        and all(type(s) is int and 0<=s<2**31-1 for s in seeds),'Invalid fresh rollout seed block')
    params=req.get('parameters')
    require(isinstance(params,list) and len(params)==len(seeds) and all(isinstance(a,list) and len(a)==6 and all(type(x) in (int,float) and math.isfinite(x) and -1.<=x<=1. for x in a) for a in params),'Expected six normalized finite search coefficients per lane')
    require(req.get('parent_sha',PARENT_HASH)==PARENT_HASH and req['target_deg']==15,'Frozen R27 and 15 degree goal required')
    require(sha(ARCHIVE_PLAN)==PLAN_HASH and checkpoint(PARENT)==PARENT_HASH,'Changed frozen R27')
    ref=read(ARCHIVE_PLAN);cert=read(PARENT.parent/'validate/accepted_0000.json')
    require(cert['plan_sha']==PLAN_HASH and cert['report_sha']==REPORT_HASH and sha(Path(cert['dest'])/'report.json')==REPORT_HASH,'Changed R27 validation receipt')
    require(ref['parent']==str(PARENT) and ref['parent_sha']==PARENT_HASH and code_hashes()==ref['code'],'Changed plan or sampler sources')
    verify_prerequisites();scene=Path(ref['scene_source']);grasp=Path(ref['grasp']);domain=verify_domain(scene,grasp)
    historic=read(Path(cert['dest'])/'report.json')
    for k in ('scene_manifest_sha256','scene_sha256','asset_sha256','grasp_sha256'):
        actual=domain['cube_asset_sha256'] if k=='asset_sha256' else domain[k]
        require(actual==historic[k],f'Domain or calibration pin changed: {k}')
    ck=torch.load(PARENT/'best.pt',weights_only=True,map_location='cpu');meta=ck['metadata']
    for name,h in meta['source_sha256'].items():
        require(Path(name).name==name and sha(ROOT/name)==h,'Changed parent training source: '+name)
    forbidden=set(meta['training_seeds'])|set(meta['loss_val_seeds_90'])|set(ref['seeds'])
    if req['role']!='smoke':
        require(not forbidden.intersection(seeds),'Previously exposed R27 seeds cannot be used for this campaign')
        lo,hi=(41_000_000,51_000_000) if req['role']=='train' else (61_000_000,71_000_000)
        require(all(lo<=v<hi for v in seeds),'Seed outside disjoint project role block')
    # Duplicate seeds within a wave intentionally pair distinct controller parameters.
    return scene,grasp,meta,restored(ck)
def measurements(e, target, soft, arm, init_xy, postcore, poses):
    """Original sampler conditions; common device and dimensions checked explicitly."""
    import torch
    post=e.robot.get_dof_positions()[:,e.ids];cube_w=postcore[:,:3]+e.offset
    if not (post.isfinite().all() and postcore.isfinite().all() and poses.isfinite().all() and e.hand_impulse.isfinite().all() and e.contacts.isfinite().all()):raise ValueError('Nonfinite physics measurement')
    dots=(poses[:,:,3:7]*postcore[:,None,3:7]).sum(-1).abs().clamp(max=1.)
    dis=(poses[:,:,:3]-cube_w[:,None,:]).norm(dim=-1).amax(1)
    deformed=dots.amin(1)<math.cos(math.radians(15))
    assembled=(dots.amin(1)>=math.cos(math.radians(2.5)))&(dis<=.002)
    hard=((post<e.hard[...,0]-1e-4)|(post>e.hard[...,1]+1e-4)).any(1)
    softarm=((post[:,arm]<e.limits[:,arm,0]-1e-4)|(post[:,arm]>e.limits[:,arm,1]+1e-4)|soft[:,arm]).any(1)
    height=postcore[:,2]-e.cfg['table_z']-.045
    drop=(postcore[:,2]<e.cfg['table_z']-.12)|((postcore[:,:2]-init_xy).norm(dim=1)>.45)
    table=e.contacts[:,12]>=1e-5;bad_self=e.contacts[:,10]>1e-6;bad_table=e.contacts[:,11]>1e-6
    common=hard|softarm|drop|bad_self|bad_table
    physical_support=(height>=.02)&~table&(e.hand_impulse[:,0]>e.cfg['success']['contact_impulse_min'])&~common
    return (hard,softarm,drop,bad_self,bad_table),deformed,assembled,height,table,physical_support

def require_locks():
    # Same project-wide inherited FD protocol used by the R17/R27 offline evaluator.
    d=ROOT.parent/'work/teacher_optimization_20261103'
    for env,name in [('CUBE_DASH_GPU_FD','contact_hold_gpu_training.lock'),('CUBE_DASH_ISAAC_FD','isaac_gui_evaluation.lock')]:
        fd=int(os.environ[env])
        require(fd>=3 and os.path.samestat(os.fstat(fd),(d/name).stat()),'Not a project resource lease: '+name)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


def run(req,out):
    import torch
    from contact_hold90_cuda_domain import open_scene,make_plan,make_teacher,FAILURES90
    from contact_hold90_cuda_sample import local_palm_reader,HoldMeter
    from phase_contact_policy import PHASES
    from joint_safety import project_target

    source,grasp,meta,actor=inputs(req);seeds=req['seeds'];n=len(seeds)
    actor=actor.cuda().eval();ids=meta['output_indices']
    e=open_scene(source,n,grasp)
    require(e.names==meta['joint_names'] and list(ids)==[i for i,k in enumerate(e.names) if k.startswith('left_') or k.endswith('_L')],'Incorrect left action mapping')
    e.reset(seeds); require(bool((e.contacts[:,12]>0).all()),'No reset table impulse')
    read_palm=local_palm_reader(e); plan=make_plan(e.core(),grasp)
    teacher=make_teacher(plan,e.cfg,meta['sources'][0]['candidate'],grasp,e.batch_gain)
    dt=e.dt*e.dec;require(abs(dt*30-1)<1e-8 and e.cfg['physics_hz']==240,'Changed physics rate')
    dev=e.home.device;target=e.home.clone();cap=torch.minimum(e.step,e.vel*dt); arm=[i for i,k in enumerate(e.names) if k.startswith('Joint')]
    init_xy=target.new_tensor(e.cfg['cube_xyz'][:2]);original=e.cfg['success']
    lane=torch.arange(n,device=dev);prev=torch.full_like(lane,-1);start=torch.zeros_like(lane)
    prefix=plan.accepted.clone();post=torch.zeros_like(prefix);step_p=torch.zeros_like(lane);step_q=step_p.clone();env_hold=step_p.clone();meter=HoldMeter(n,dev)
    post_support_lost=step_p.clone();unsafe_seen=torch.zeros_like(prefix);admitted=torch.zeros_like(prefix);reason=['' if v else 'reset_planner_rejected' for v in prefix.tolist()]
    feedback=e.contacts.clone();feedback[:,:10]=0
    new=PostPhase(e,req,out)
    new.record_state(~prefix,-1,target,'terminal','reset_planner_rejected',~prefix)
    prefix_cap=int(req.get('prefix_steps',3500));post_cap=int(req.get('post_steps',new.max_steps))
    require(1<=prefix_cap<=3500 and 1<=post_cap<=new.max_steps,'Horizon exceeds native budget')
    print('NATURAL90_READY',n,'IK',int(prefix.sum()),'prefix cap',prefix_cap,'post cap',post_cap,flush=True)
    last_save=time.monotonic(); started=last_save
    def end(mask,why,unsafe=False):
        # First failing gate wins; accepts are processed only after safety masks.
        mask=mask&(prefix|post)
        if not mask.any():return
        ii=mask.nonzero(as_tuple=True)[0].tolist()
        new.record_state(mask,step,target,'terminal',why,prefix)
        for i in ii:reason[i]=why if isinstance(why,str) else why[i]
        prefix[mask]=False;post[mask]=False
        if unsafe:unsafe_seen[mask]=True
    with (out/'progress.jsonl').open('x') as fp:
        for step in range(prefix_cap+post_cap):
            if not (prefix|post).any():break
            # Snapshot active groups before advancing the shared physical scene exactly once.
            before_post=post.clone(); q=e.robot.get_dof_positions()[:,e.ids];v=e.robot.get_dof_velocities()[:,e.ids];core=e.core();vel=e.views['Core'].get_velocities()
            command=target.clone();accepted=torch.zeros_like(prefix)
            if prefix.any():
                raw,ph,fail,accepted=teacher.action(prefix,q,v,core,target,feedback,vel,e.hand_impulse)
                mask=prefix&(fail!=0)
                if mask.any():
                    why=[FAILURES90.get(x,'invalid_teacher_failure') for x in fail.tolist()]
                    end(mask,why)
                ph=torch.where(prefix,ph,0);raw=torch.where(prefix[:,None],raw,target)
                start=torch.where(prefix&(ph!=prev),step,start);prev=torch.where(prefix,ph,prev)
                feat=torch.log1p(e.feature_contacts.to(q.dtype)/.001) if step else q.new_zeros(n,10)
                x=torch.cat((q,v,target,core[:,:3],core[:,[6,3,4,5]],feat,torch.nn.functional.one_hot(ph,len(PHASES)).to(q.dtype),((step-start)/300.).clamp(max=1)[:,None]),1)
                require(x.shape==(n,198) and bool(torch.isfinite(x).all()),'Bad measured R27 inputs')
                pred=actor(x);req0=raw.clone()
                req0[:,ids]=torch.where(prefix[:,None],target[:,ids]+(pred-target[:,ids]).clamp(-cap[:,ids],cap[:,ids]),raw[:,ids])
                command=torch.where(prefix[:,None],req0,command)
            if before_post.any():
                post_request=new.propose(target,before_post,step_q)
                require(post_request.shape==target.shape and bool(torch.isfinite(post_request).all()),'Invalid continuation action')
                command=torch.where(before_post[:,None],post_request,command)
            was=prefix.clone()
            if not (was|before_post).any():break
            cmd,soft=project_target(target,command,e.limits,e.step,e.vel,dt)
            new.remember(step,q,v,cmd,core,was,before_post)
            e.step_physics(cmd);target=cmd;feedback=e.contacts
            core=e.core();vel=e.views['Core'].get_velocities()
            require(bool(torch.isfinite(vel).all()),'Nonfinite measured cube velocity')
            poses=torch.stack([b.get_transforms() for b in e.views.values()],1)
            flags,deformed,assembled,height,table,support=measurements(e,target,soft,arm,init_xy,core,poses)
            step_p+=was.long();step_q+=before_post.long()
            meter.update(step,was,was&support&assembled,read_palm(core))
            orig_stable=(e.hand_impulse>original['contact_impulse_min']).all(1)&(height>=original['lift_m'])&~table&~flags[2]&(vel[:,:3].norm(dim=1)<original['linear_speed_max'])&(vel[:,3:].norm(dim=1)<original['angular_speed_max'])
            env_hold=torch.where(was&orig_stable,env_hold+1,0)
            # All five original hard/collision gates are unconditional across phases.
            # On the prefix also keep the exact original cube layer gate. The post
            # observer requires an assembled cube before free twist and enforces
            # target/non-target layer integrity and center alignment during twisting.
            readback_safe=~(flags[0]|flags[1]|flags[2]|flags[3]|flags[4]|(was&deformed))
            post_support_lost=torch.where(before_post&~support,post_support_lost+1,0)
            post_return_table=before_post&table
            loss_grace=int(e.domain_receipt['loss_grace_frames'])
            post_loss=before_post&(post_support_lost>=loss_grace)
            readback_safe &= ~(post_return_table|post_loss)
            post_metrics=new.observe(was|before_post,before_post,readback_safe,step)
            new.record_progress(before_post,step,target)
            end(flags[0],'hard_joint_limit',True);end(flags[1],'soft_arm_limit',True);end(flags[2],'cube_drop',True)
            new.record_collisions(flags[3]&(was|before_post),step,target,core,was)
            end(was&deformed,'layer_deviation',True);end(flags[3],'robot_self_contact',True);end(flags[4],'robot_table_contact',True)
            end(post_return_table,'cube_returned_to_table_after_wrist',True)
            end(post_loss,'primary_support_lost_after_wrist',True)
            if before_post.any():
                end(post&post_metrics['unsafe'],post_metrics['reason'],True)
                end(post&post_metrics['failed'],post_metrics['reason'])
                end(post&post_metrics['success'],'post_turn_verified')
                end(post&(step_q>=post_cap),'post_horizon')
            # Accepted is the R27 wrist-teacher endpoint command, but no handoff
            # unless the last actual step still met its measured 30-frame hold.
            transfer=prefix&accepted&meter.end&support&assembled
            end(prefix&accepted&~transfer,'wrist_endpoint_without_supported_hold')
            if transfer.any():
                new.begin(transfer,target,core,poses)
                new.record_state(transfer,step,target,'handoff')
                prefix[transfer]=False;post[transfer]=True;admitted[transfer]=True
            end(prefix&(env_hold>=round(original['hold_seconds']*30)),'environment_done_before_wrist_endpoint')
            end(prefix&(step_p>=prefix_cap),'prefix_horizon')
            if step%120==0 or transfer.any() or time.monotonic()-last_save>90:
                event=dict(step=step,prefix=int(prefix.sum()),post=int(post.sum()),entered_post=int(admitted.sum()),closed=sum(bool(x) for x in reason),elapsed_s=round(time.monotonic()-started,2))
                fp.write(json.dumps(event)+'\n');fp.flush();save(out/'progress.json',event);print('NATURAL90_STEP',event,flush=True);last_save=time.monotonic()
    require(not (prefix|post).any() and all(reason),'Unresolved episode termination')
    new.evidence.close();new.incidents.close();new.traces.close()
    rows=[]
    for i,s in enumerate(seeds):
        row=new.row(i);row.update(seed=s,lane=i,parameters=req['parameters'][i],prefix_steps=int(step_p[i]),post_steps=int(step_q[i]),steps=int(step_p[i]+step_q[i]),prefix_endpoint_held=bool(admitted[i]),target_deg=req['target_deg'],success=(reason[i]=='post_turn_verified'),unsafe=bool(unsafe_seen[i]),end_reason=reason[i])
        if row['success']:require(row['natural_ready'] and row['turn_started'] and not row['unsafe'],'Invalid physical success evidence')
        rows.append(row)
    report=dict(schema='natural_face90_rollout_v1',role=req['role'],seeds=seeds,rows=rows,parent_sha=PARENT_HASH,source_plan_sha=PLAN_HASH,
        order='R27 student pickup from reset, measured endpoint/steady-hold gate, post policy in same continuously stepped scene',
        action_dim=56,control_hz=30,physics_hz=240,reset_count=1,simulation_device='cuda:0',physics_stepped=True,face='nZ',domain_receipt=e.domain_receipt,
        controller_evidence_sha256=sha(out/'controller_states.jsonl'),collision_evidence_sha256=sha(out/'collision_events.jsonl'),request_sha256=sha(out/'request.json'),post_evidence_sha256=sha(out/'post_measurements.jsonl'),reference_report_sha256=REPORT_HASH,accepted_model=False,simulation_only=True,cross_lane_contacts=e.cross,software=new.software,elapsed_s=time.monotonic()-started)
    require(not e.cross,'Cross lane contacts');save(out/'report.json',report);print('NATURAL90_COMPLETE',n,sum(r['success'] for r in rows),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--request',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    req=read(args.request);out=args.out
    require_locks();out.mkdir(exist_ok=False,parents=True);save(out/'request.json',req)
    # Start Isaac before importing CUDA or simulation libraries.
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    try:
        import torch
        with torch.no_grad():run(req,out)
    except BaseException as err:
        save(out/'failed.json',dict(error=repr(err)));raise
    finally:
        from isaac_compat import close_application
        close_application(app)

class PostPhase:
    """Bridge the joint controller to a separately implemented passive physics observer."""
    def __init__(self,e,req,out):
        import torch
        import natural_face90_v5_metrics as metrics
        import natural_face90_v5_controller as control
        self.e=e;self.req=req;self.n=e.n;self.device=e.home.device;self.out=out
        self.monitor=metrics.NaturalFace90Observer(e,target_deg=req['target_deg'])
        import natural_face90_v5_contacts as attribution
        import natural_face90_v5_clearance as clearance
        self.pairs=attribution.PairRecorder(e);e.reducer=self.pairs
        from collections import deque
        self.history=deque(maxlen=12)
        self.body_names=['world','Link4_R','Link5_R','Link7_R','right_index_DIP_link']
        paths=e.robot.link_paths
        require(len(paths)==e.n,'Missing measured articulation body paths')
        self.body_ids=[list(map(lambda p:p.rsplit('/',1)[-1],paths[0])).index(k) for k in self.body_names]
        require(all([row[j].rsplit('/',1)[-1] for j in self.body_ids]==self.body_names for row in paths),'Body pose ordering differs across lanes')
        self.incidents=(out/'collision_events.jsonl').open('x',buffering=1)
        self.traces=(out/'controller_states.jsonl').open('x',buffering=1)
        self.init_controller(control)
        self.last_logged_phase=torch.full_like(self.controller.phase,-1)
        self.software={Path(m.__file__).name:sha(Path(m.__file__)) for m in (metrics,control,attribution,clearance)}
        self.software[Path(__file__).name]=sha(Path(__file__))
        self.obs={};self.retract=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.started=self.retract.clone();self.ctrl_failure=self.retract.clone();self.ctrl_reason=['']*self.n
        self.cache=[self.blank() for _ in range(self.n)]
        self.evidence=(out/'post_measurements.jsonl').open('x',buffering=1)
    @staticmethod
    def blank():
        return dict(natural_ready=False,turn_started=False,progress=0.,posture_quality=0.,right_contact=0.,hold_fraction=0.,final_face_deg=0.,peak_signed_face_deg=0.,post_samples=0,target_contact_steps=0)
    def observe(self,active,post,safe,step):
        import torch
        m=self.monitor.observe(active,post,safe,self.retract,step);self.obs=m
        result={k:m[k] for k in ('natural_ready','turn_started','success','structure_ok','allowed_layer_state','wrong_contact','progress','posture_quality','right_contact','hold_fraction','face_deg')}
        require(all(v.shape==(self.n,) and bool(torch.isfinite(v).all()) for v in result.values()),'Invalid observer measurement')
        unsafe=post&(~m['structure_ok']|~m['allowed_layer_state']|m['wrong_contact'].bool())
        why=['']*self.n
        if post.any():
            m['right_forearm_clearance_m']=self.controller.clearance.measure(
                self.e.robot.get_dof_positions()[:,self.e.ids][:,self.controller.rp.ix].double())[0]
            m['right_detour_done']=self.controller.detour_done
            fields=('natural_ready','turn_started','success','natural_now','structure_ok','allowed_layer_state','wrong_contact','progress','posture_quality','right_contact','hold_fraction','face_deg','target_impulse','other_impulse','anchor_error_m','piece_off_deg','other_rot_deg','normalized_world_palm_turn_deg','palm_toward_midline_m','core_toward_midline_m','left_faces_right','right_faces_left','palm_normal_dot','elbows_out','uncrossed','joint6_measured_deg','left_static_impulse','left_target_impulse','right_total_impulse','selected_relative_angular_speed','natural_streak','hold_streak','measured_contact_rotation_deg','actual_retraction_m','contact_goal_reached')
            fields+=('right_forearm_clearance_m','right_detour_done')
            fields += ('peak_qualified_face_deg','retreat_floor_deg','retreat_drop_deg','core_linear_speed_mps','core_angular_speed_rad_s')
            fields += tuple(stage+'_'+field for stage in ('natural','retreat') for field in
                            ('motion_ok','motion_window_frames','steady_suffix_frames','core_span_m','core_span_deg','piece_span_m','piece_span_deg'))
            fields=[k for k in fields if k in m]
            # Transfer only the active postgroup's physical measurements; evidence comes solely from simulation.
            ii=post.nonzero(as_tuple=True)[0];values=torch.stack([m[k].double()[ii] for k in fields],1).cpu().tolist()
            for i,nums in zip(ii.tolist(),values):
                row=dict(zip(fields,nums));v=self.cache[i]
                v.update(natural_ready=bool(row['natural_ready']),turn_started=bool(row['turn_started']),progress=min(1.,max(0.,row['progress'])),
                  posture_quality=min(1.,max(0.,row['posture_quality'])),hold_fraction=min(1.,max(0.,row['hold_fraction'])),final_face_deg=row['face_deg'])
                v['post_samples']+=1;v['target_contact_steps']+=int(row['right_contact']>0)
                v['right_contact']=v['target_contact_steps']/v['post_samples'];v['peak_signed_face_deg']=max(v['peak_signed_face_deg'],row['face_deg'])
                if not row['structure_ok']:why[i]='face_structure_violation'
                elif not row['allowed_layer_state']:why[i]='layer_motion_outside_current_gate'
                elif row['wrong_contact']:why[i]='right_contact_on_wrong_piece_or_link'
                elif self.ctrl_reason[i]:why[i]=self.ctrl_reason[i]
                self.evidence.write(json.dumps(dict(step=step,lane=i,seed=self.req['seeds'][i],safe=bool(safe[i]),retract=bool(self.retract[i]),controller_phase=self.stage_name(i),**row),allow_nan=False)+'\n')
        return dict(unsafe=unsafe,failed=self.ctrl_failure,success=m['success'],reason=why)
    def remember(self,step,q,dq,command,core,prefix,post):
        # Small GPU ring buffer; no additional physics or per-frame disk transfer.
        self.history.append(dict(step=step,q=q.clone(),dq=dq.clone(),command=command.clone(),
            core=core.clone(),prefix=prefix.clone(),post=post.clone(),phase=self.action_phase.clone(),
            body_poses=self.body_poses()))
    def body_poses(self):
        poses=self.e.robot.get_link_transforms()[:,self.body_ids].clone()
        poses[:,:,:3]-=self.e.offset[:,None]
        return poses
    def record_progress(self,post,step,target):
        mask=post&((self.action_phase!=self.last_logged_phase)|(step%30==0))
        if mask.any():
            self.record_state(mask,step,target,'periodic')
            self.last_logged_phase[mask]=self.action_phase[mask]
    def record_state(self,mask,step,target,event,reason=None,prefix=None):
        if not mask.any():return
        q=self.e.robot.get_dof_positions()[:,self.e.ids]
        dq=self.e.robot.get_dof_velocities()[:,self.e.ids]
        core=self.e.core();bodies=self.body_poses();metrics=self.controller.diagnostics(target)
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            is_prefix=prefix is not None and bool(prefix[i])
            row=dict(event=event,step=step,lane=i,seed=self.req['seeds'][i],
                phase='prefix' if is_prefix else self.stage_name(i),
                reason=reason if isinstance(reason,str) else reason[i] if reason is not None else None,
                q=q[i].tolist(),dq=dq[i].tolist(),command=target[i].tolist(),core=core[i].tolist(),
                joint_names=self.e.names,body_names=self.body_names,body_poses=bodies[i].tolist(),
                post_controller_started=bool(self.controller.seen[i]),
                **{k:v[i].tolist() for k,v in metrics.items()})
            self.traces.write(json.dumps(row,allow_nan=False)+'\n')
    def record_collisions(self,mask,step,target,core,prefix):
        if not mask.any():return
        q=self.e.robot.get_dof_positions()[:,self.e.ids]
        vel=self.e.robot.get_dof_velocities()[:,self.e.ids]
        bodies=self.body_poses()
        distances,_=self.controller.clearance.measure(q[:,self.controller.rp.ix].double())
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            before=[]
            for h in self.history:
                if bool(h['prefix'][i]|h['post'][i]):
                    before.append(dict(step=h['step'],phase='prefix' if bool(h['prefix'][i]) else self.phase_names[int(h['phase'][i])],
                        **{k:h[k][i].tolist() for k in ('q','dq','command','core','body_poses')}))
            event=dict(step=step,lane=i,seed=self.req['seeds'][i],
                phase='prefix' if bool(prefix[i]) else self.stage_name(i),
                parameters=self.req['parameters'][i],joint_names=self.e.names,
                q=q[i].tolist(),dq=vel[i].tolist(),command=target[i].tolist(),core=core[i].tolist(),
                body_names=self.body_names,body_poses=bodies[i].tolist(),
                forearm_envelope_clearance_m=float(distances[i]),
                self_contact_impulse_Ns=float(self.e.contacts[i,10]),
                pairs=self.pairs.lane(i,float(self.e.contacts[i,10])),before_steps=before)
            self.incidents.write(json.dumps(event,allow_nan=False)+'\n')
    def row(self,i):return self.cache[i].copy()
    def init_controller(self,control):
        self.controller=control.NaturalFace90Controller(self.e,self.req['parameters'],self.req['target_deg'])
        self.max_steps=int(control.MAX_POST_STEPS)
        self.phase_names=tuple(control.PHASES);self.action_phase=self.controller.phase.clone()
    def begin(self,mask,target,core,poses):
        self.controller.begin(mask,target,self.obs)
        self.started|=mask
    def propose(self,target,active,steps):
        self.action_phase=self.controller.phase.clone()
        command,retract,failed,reasons=self.controller.action(target,active,self.obs)
        require(retract.shape==active.shape and failed.shape==active.shape and len(reasons)==len(active),'Controller status shape mismatch')
        self.retract=retract&active;self.ctrl_failure=failed&active;self.ctrl_reason=reasons
        return command
    def stage_name(self,i):
        return self.phase_names[int(self.action_phase[i])]
if __name__=='__main__':main()
