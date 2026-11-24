"""Parallel PhysX development collection from reset to failure or the wrist hold."""
import argparse
from pathlib import Path
import traceback


def sample(args):
    import time
    import numpy as np
    import torch
    from contact_hold_cuda_scene import Scene
    from contact_hold_wide_replay import Bank
    from contact_hold_policy import restored
    from contact_hold_loop_data import checkpoint,read,write,ROOT
    from contact_hold_dataset import sha
    from phase_contact_policy import PHASES
    from joint_safety import project_target
    from stage_gate import require_staged_gate
    from side_gate import require_side_gate
    from opposition_gate import require_opposition_gate
    from evidence_paths import local_path,raw_sha
    for gate in (require_staged_gate,require_side_gate,require_opposition_gate):gate()
    cert=read(ROOT/'wrist_direction_gate_certificate.json')
    if not cert['passed'] or cert['independent_accepted']<5:raise ValueError('Missing wrist gate')
    for name,h in (cert['source_sha256']|cert['assets_sha256']).items():
        if raw_sha(local_path(name,ROOT))!=h:raise ValueError('Changed wrist certificate dependency')
    if raw_sha(ROOT/'wrist_direction_protocol_v1.json')!=cert['protocol_sha256']:raise ValueError('Wrist protocol changed')
    for t in cert['trials']:
        d=local_path(t['run'],ROOT)
        for name,key in [('config.json','config_sha256'),('runtime_config.json','runtime_config_sha256'),('asset_contract.json','asset_contract_sha256'),(f"trajectory_{t['seed']}.json",'trajectory_sha256')]:
            if raw_sha(d/name)!=t[key]:raise ValueError('Wrist evidence changed')
    p=read(args.plan); out=args.output;out.mkdir(parents=True,exist_ok=False)
    seeds=p['seeds'];n=len(seeds)
    if not 2<=n<=2048 or len(set(seeds))!=n or any(s not in range(1_000_000,9_000_000) for s in seeds):
        raise ValueError('Expected unique development seeds from reserved GPU collection range')
    parent=Path(p['parent'])
    if checkpoint(parent)!=p['parent_sha'] or any(sha(ROOT/k)!=h for k,h in p['code'].items()):raise ValueError('Sampler plan changed')
    saved=torch.load(parent/'best.pt',weights_only=True,map_location='cpu')
    policy=restored(saved).cuda()
    meta=saved['metadata']
    for name,h in meta['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed checkpoint dependency '+name)
    source=Path(meta['sources'][0]['path'])
    if sha(source/'artifact_manifest.json')!=meta['sources'][0]['manifest_sha256']:raise ValueError('Changed teacher scene source')
    e=Scene(source,n)
    if e.names!=meta['joint_names']:raise ValueError('Policy joint order mismatch')
    e.reset(seeds)
    ids=meta['output_indices'];arm=[i for i,k in enumerate(e.names) if k.startswith('Joint')]
    if not (e.contacts[:,12]>0).all():raise ValueError('Missing reset table impulse')
    from contact_hold_cuda_ik import plan_reset
    from contact_hold_cuda_teacher import BatchedTeacher,FAILURES
    torch.cuda.synchronize(); t0=time.monotonic()
    plan=plan_reset(e.core())
    teacher=BatchedTeacher(plan,e.cfg,meta['sources'][0]['candidate'],gain_writer=e.batch_gain)
    torch.cuda.synchronize(); plan_elapsed=time.monotonic()-t0
    active=plan.accepted.clone(); print('PLAN_LANES',n,'accepted',int(active.sum()),'seconds',plan_elapsed,flush=True)
    target=e.home.clone(); ids=meta['output_indices']; dev=target.device
    ends=torch.where(active,0,-1).long()
    lane=torch.arange(n,device=dev); start=torch.zeros_like(lane); prev=torch.full_like(lane,-1)
    phase_steps=torch.zeros(n,len(PHASES),dtype=torch.long,device=dev)
    holds=torch.zeros_like(lane)
    # Use GLOBAL wave index so retrying a slice preserves the student/teacher schedule.
    handoff=torch.tensor((0,4,0,5),device=dev)[(lane+p.get('lane_offset',0))%4]
    cap=torch.minimum(e.step,e.vel*(e.dt*e.dec)); xyz=target.new_tensor(e.cfg['cube_xyz'])
    bank=Bank(8192,seeds[0]); success=e.cfg['success']; required=max(1,round(success['hold_seconds']*e.cfg['control_hz']))
    t0=time.monotonic(); snapshots={}; steps_executed=0
    feedback=e.contacts.clone(); feedback[:,:10]=0
    for step in range(args.steps):
        q=e.robot.get_dof_positions()[:,e.ids];v=e.robot.get_dof_velocities()[:,e.ids];core=e.core()
        velocity=e.views['Core'].get_velocities()
        raw,ph,fail,accepted=teacher.action(active,q,v,core,target,feedback,velocity)
        ends=torch.where(active&(fail!=0),fail,ends);active &= fail==0
        if not active.any():break
        ph=torch.where(active,ph,0);raw=torch.where(active[:,None],raw,target)
        start=torch.where(active&(ph!=prev),step,start);prev=torch.where(active,ph,prev)
        phase_steps[lane,ph]+=active.long()
        elapsed=((step-start)/300.).clamp(max=1)[:,None]
        feature=torch.log1p(e.feature_contacts.to(q.dtype)/.001) if step else q.new_zeros(n,10)
        x=torch.cat((q,v,target,core[:,:3],core[:,[6,3,4,5]],feature,
                     torch.nn.functional.one_hot(ph,len(PHASES)).to(q.dtype),elapsed),1)
        if x.shape!=(n,198) or not x.isfinite().all():raise ValueError('Invalid observation')
        label,hit=project_target(target,raw,e.limits,e.step,e.vel,e.dt*e.dec)
        student=policy(x); use=active&(ph>=handoff)
        request=raw.clone()
        request[:,ids]=torch.where(use[:,None],target[:,ids]+(student-target[:,ids]).clamp(-cap[:,ids],cap[:,ids]),raw[:,ids])
        cmd,soft=project_target(target,request,e.limits,e.step,e.vel,e.dt*e.dec)
        e.step_physics(cmd);target=cmd;post=e.robot.get_dof_positions()[:,e.ids];postcore=e.core()
        feedback=e.contacts
        velocity=e.views['Core'].get_velocities()
        quat=torch.stack([obj.get_transforms()[:,3:7] for obj in e.views.values()],1)
        if not (post.isfinite().all() & quat.isfinite().all() & postcore.isfinite().all() & velocity.isfinite().all()):raise ValueError('Nonfinite PhysX state')
        hard=((post<e.hard[...,0]-1e-4)|(post>e.hard[...,1]+1e-4)).any(1)
        softarm=((post[:,arm]<e.limits[:,arm,0]-1e-4)|(post[:,arm]>e.limits[:,arm,1]+1e-4)|soft[:,arm]).any(1)
        drop=(postcore[:,2]<e.cfg['table_z']-.12)|((postcore[:,:2]-xyz[:2]).norm(dim=1)>.45)
        layer=((quat*postcore[:,None,3:7]).sum(-1).abs()<np.cos(np.deg2rad(15))).any(1)
        stable=((e.hand_impulse>success['contact_impulse_min']).all(1)&(postcore[:,2]-e.cfg['table_z']-.0285>=success['lift_m'])
                &(velocity[:,:3].norm(dim=1)<success['linear_speed_max'])&(velocity[:,3:].norm(dim=1)<success['angular_speed_max'])&(e.contacts[:,12]<1e-5))
        holds=torch.where(stable,holds+1,0)
        for gate,code in ((hard,-2),(softarm,-3),(drop,-4),(layer,-5),(e.contacts[:,10]>1e-6,-6),
                          (e.contacts[:,11]>1e-6,-7),(accepted,-8),(holds>=required,-9)):
            ends=torch.where(active&gate,code,ends);active &= ~gate
        if step+1==args.steps:ends=torch.where(active,-10,ends);active.zero_()
        y=label[:,ids];mask=~hit[:,ids]&~soft[:,ids]&(y>=policy.lower-1e-6)&(y<=policy.upper+1e-6)
        ii=lane[active&use&(step>0)&mask.any(1)]
        if ii.numel():
            batch=dict(x=x[ii],y=y[ii],mask=mask[ii],val=(((step-start[ii])//10)%5==4),lane=ii,
                       step=torch.full_like(ii,step),phase=ph[ii])
            bank.add({k:z.cpu().numpy() for k,z in batch.items()})
        steps_executed=step+1
        if steps_executed in (12,120,360,480):
            torch.cuda.synchronize(); snapshots[steps_executed]=time.monotonic()-t0
        if step%120==0:
            counts=torch.bincount(prev[active],minlength=len(PHASES)).tolist()
            print('VECTOR_STEP',step,'active',int(active.sum()),'active_phase',{k:v for k,v in zip(PHASES,counts) if v},flush=True)
        if not active.any():break
    torch.cuda.synchronize(); elapsed_rollout=time.monotonic()-t0
    arr=bank.finish();np.savez_compressed(out/'batch.npz',**arr)
    reasons={-1:'reset_planner_rejected',-2:'hard_joint_limit',-3:'soft_arm_limit',-4:'cube_drop',-5:'layer_deviation',
             -6:'robot_self_contact',-7:'robot_table_contact',-8:'controller_hold_reached_not_audit',-9:'environment_done',-10:'step_limit',**FAILURES}
    report=dict(schema='cuda_feedback_query_v3',role='collection',scope='reset_to_wrist_hold_or_termination',
                full_episode=False,eligible_success_demonstration=False,checkpoint=p['parent_sha'],seeds=seeds,
                num_envs=n,physics_device='cuda:0',action_dim=56,physics_hz=240,control_hz=30,steps_executed=steps_executed,requested_steps=args.steps,
                elapsed_rollout_s=elapsed_rollout,planner_init_s=plan_elapsed,prefix_rollout_s=snapshots,
                recipe='batched_ik_contact_teacher',eligible_per_stratum=bank.seen,solver=e.readback,cross_lane_contacts=e.cross,
                student_start_phase=handoff.tolist(),teacher_assisted_lanes=lane[handoff!=0].tolist(),reset_count_per_lane=1,
                phase_steps=[{k:v for k,v in zip(PHASES,row) if v} for row in phase_steps.tolist()],
                ends=[reasons[k] for k in ends.tolist()],source_scene_sha256=sha(source/'dynamic_training_scene.usda'),
                samples=len(arr['x']),data_sha256=sha(out/'batch.npz'),plan_sha256=sha(args.plan))
    write(out/'report.json',report);print('GPU_COLLECTION_COMPLETE',n,'rows',report['samples'],'planning_s',plan_elapsed,'rollout_s',elapsed_rollout,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=3500)
    args=p.parse_args()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    from isaac_compat import close_application
    try:
        import torch
        with torch.no_grad():sample(args)
    except BaseException:traceback.print_exc();raise
    finally:close_application(app)
