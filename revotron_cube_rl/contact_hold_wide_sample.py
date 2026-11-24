"""Parallel PhysX development collection from reset to failure or the wrist hold."""
import argparse
from pathlib import Path
import traceback


def sample(args):
    import time
    from types import SimpleNamespace as NS
    from functools import partial
    import numpy as np
    import torch
    from contact_hold_wide_scene import Scene,FINGERS
    from contact_hold_wide_replay import Bank
    from contact_hold_policy import restored
    from contact_hold_loop_data import checkpoint,read,write,ROOT
    from contact_hold_dataset import sha
    from phase_contact_policy import PHASES
    from joint_safety import project_target
    from contact_hold_student_eval import StudentEvaluation
    from rewards import pickup_reward
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
    if not np.all(e.contacts[:,12]>0):raise ValueError('No table impulse at reset in one or more lanes')
    active=np.ones(n,bool);reasons=['']*n;lastphase=['']*n;start=[0]*n
    phase_steps=[{} for _ in seeds]; teachers=[];envs=[];target=e.home.clone()
    cube=e.core().cpu(); holds=[0]*n
    scheduler=StudentEvaluation.teacher_class(NS(candidate=meta['sources'][0]['candidate']))
    def gains(lane,prop,stiffness=None,damping=None,joint_ids=()):
        e.lane_gain(lane,prop,joint_ids,stiffness if stiffness is not None else damping)
    plan_started=time.monotonic()
    from contact_hold_wide_planner import reset_paths,bind_path
    from opposition_teacher import OppositionPlanner
    from unittest.mock import patch
    for j,measured,yaw,path,diag,rejected in reset_paths(cube):
        if j%64==0:print('PLAN_LANES',j,'of',n,flush=True)
        lane_dir=out/f'lane_{j:03d}';lane_dir.mkdir()
        data=NS(root_state_w=cube[j:j+1],root_pos_w=cube[j:j+1,:3])
        robot=NS(data=NS(),write_joint_stiffness_to_sim_index=partial(gains,j,'stiffnesses'),
                 write_joint_damping_to_sim_index=partial(gains,j,'dampings'))
        env=NS(run_dir=lane_dir,cfg=e.cfg,ids=list(range(56)),cube=NS(data=data),robot=robot,metrics={},
               scales=e.scales.cpu(),targets=e.home[j:j+1].cpu(),contact_impulse=[0.,0.])
        if rejected:
            active[j]=False;reasons[j]='reset_planner_rejected: '+rejected;teacher=None
        else:
            # Only supply the per-lane initial IK solve; original action/feedback/gains are unmodified.
            with patch.object(OppositionPlanner,'plan',bind_path(measured,yaw,path,diag)):
                teacher=scheduler(env,ROOT/'opposition_grasp_v20.json')
        teachers.append(teacher);envs.append(env)
        marker=lane_dir/'contact_hold_candidate.json'
        if marker.exists():marker.rename(lane_dir/'development_controller.json')
    bank=Bank(8192,seeds[0]);plan_elapsed=time.monotonic()-plan_started;del cube
    xyz=torch.tensor(e.cfg['cube_xyz'],device='cuda'); contact=np.zeros((n,10))
    cap=torch.minimum(e.step,e.vel*(e.dt*e.dec))
    # Two student-from-reset lanes per group of four, one handoff at squeeze and one at lift.
    handoff=[(0,4,0,5)[j%4] for j in range(n)]
    t0=time.monotonic();velocity=e.views['Core'].get_velocities().cpu()
    for step in range(args.steps):
        q=e.robot.get_dof_positions()[:,e.ids]; v=e.robot.get_dof_velocities()[:,e.ids]; core=e.core()
        qs,vs,cs,ts=[t.cpu() for t in (q,v,core,target)]
        raw=ts.clone(); phase=[0]*n
        for j,(teacher,env) in enumerate(zip(teachers,envs)):
            if not active[j]:continue
            env.robot.data.joint_pos=qs[j:j+1];env.robot.data.joint_vel=vs[j:j+1]
            env.cube.data.root_pos_w=cs[j:j+1,:3];env.cube.data.root_state_w=cs[j:j+1]
            env.targets=ts[j:j+1]
            env.metrics=dict(height_m=float(cs[j,2])-e.cfg['table_z']-.0285,cube_table_contact=e.contacts[j,12]>=1e-5,
                             linear_speed=float(velocity[j,:3].norm()),angular_speed=float(velocity[j,3:].norm()))
            env.contact_impulse=e.hand_impulse[j].tolist()
            teacher.finger_contacts={s:dict(zip(FINGERS,contact[j,i*5:i*5+5])) for i,s in enumerate(('left','right'))}
            try:action=teacher.action()
            except ValueError as exc:
                # Bounded post-turn IK rejections are terminal in the original evaluator.
                active[j]=False;reasons[j]='planner_rejected: '+str(exc);continue
            name=teacher.name
            if teacher.failure_reason:
                active[j]=False; reasons[j]=teacher.failure_reason;continue
            if name not in PHASES[:10]:raise ValueError('Unknown pickup-to-wrist phase: '+name)
            if name!=lastphase[j]:lastphase[j]=name;start[j]=step
            phase_steps[j][name]=phase_steps[j].get(name,0)+1
            phase[j]=PHASES.index(name)
            raw[j]=ts[j]+env.scales[0]*action[0]
        if not active.any():break
        ph=torch.tensor(phase,device='cuda');elapsed=torch.tensor([min((step-s)/300.,1.) for s in start],device='cuda')[:,None]
        x=torch.cat([q,v,target,core[:,:3],core[:,[6,3,4,5]],
                     torch.log1p(torch.as_tensor(e.feature_contacts,device='cuda',dtype=q.dtype)/.001) if step else torch.zeros((n,10),device='cuda'),
                     torch.nn.functional.one_hot(ph,len(PHASES)).float(),elapsed],1)
        if x.shape!=(n,198) or not x.isfinite().all():raise ValueError('Bad parallel observation')
        teacher=raw.cuda();label,hit=project_target(target,teacher,e.limits,e.step,e.vel,e.dt*e.dec)
        with torch.inference_mode():student=policy(x)
        request=teacher.clone()
        use=np.array([bool(active[j] and phase[j]>=handoff[j]) for j in range(n)])
        chosen=torch.as_tensor(use,device='cuda')
        request[:,ids]=torch.where(chosen[:,None],target[:,ids]+(student-target[:,ids]).clamp(-cap[:,ids],cap[:,ids]),request[:,ids])
        cmd,soft=project_target(target,request,e.limits,e.step,e.vel,e.dt*e.dec)
        e.step_physics(cmd);target=cmd
        contact=e.contacts[:,:10].copy()
        post=e.robot.get_dof_positions()[:,e.ids];postcore=e.core()
        velocity=e.views['Core'].get_velocities().cpu()
        unsafe=((post<e.hard[...,0]-1e-4)|(post>e.hard[...,1]+1e-4)).any(1) | ((post[:,arm]<e.limits[:,arm,0]-1e-4)|(post[:,arm]>e.limits[:,arm,1]+1e-4)|soft[:,arm]).any(1)
        unsafe |= (postcore[:,2]<e.cfg['table_z']-.12)|((postcore[:,:2]-xyz[:2]).norm(dim=1)>.45)
        quat=torch.stack([obj.get_transforms()[:,3:7] for obj in e.views.values()],1)
        if not post.isfinite().all() or not quat.isfinite().all() or not postcore.isfinite().all():raise ValueError('Nonfinite GPU state')
        unsafe |= ((quat*postcore[:,None,3:7]).sum(-1).abs()<np.cos(np.deg2rad(15))).any(1)
        bad=unsafe.cpu().numpy()|(e.contacts[:,10]>1e-6)|(e.contacts[:,11]>1e-6)
        for j in np.flatnonzero(active&bad):active[j]=False;reasons[j]='safety_or_cube_shape_gate'
        heights=(postcore[:,2]-e.cfg['table_z']-.0285).cpu().numpy()
        for j in np.flatnonzero(active):
            # Distances/action cost affect the discarded reward only; the terminal hold uses measured quantities.
            _,_,holds[j],env_done,_=pickup_reward(e.cfg,(0.,0.),heights[j],e.hand_impulse[j],e.contacts[j,12],
                float(velocity[j,:3].norm()),float(velocity[j,3:].norm()),holds[j],0.,0.,False)
            if teachers[j].accepted or env_done or step+1==args.steps:
                active[j]=False
                reasons[j]='controller_hold_reached_not_audit' if teachers[j].accepted else ('environment_done' if env_done else 'step_limit')
        good=np.flatnonzero(active&use&(step>0))
        if len(good):
            ii=torch.as_tensor(good,device='cuda')
            y=label[:,ids];mask=~hit[:,ids]&~soft[:,ids]&(y>=policy.lower-1e-6)&(y<=policy.upper+1e-6)
            batch={k:z.cpu().numpy() for k,z in dict(x=x[ii],y=y[ii],mask=mask[ii],val=torch.tensor([((step-start[j])//10)%5==4 for j in good]),lane=torch.from_numpy(good),step=torch.full((len(good),),step),phase=ph[ii]).items()}
            ok=batch['mask'].any(1)
            bank.add({k:v[ok] for k,v in batch.items()})
        if step%120==0:
            from collections import Counter
            print('VECTOR_STEP',step,'active',int(active.sum()),'phase',dict(Counter(lastphase)),flush=True)
    arr=bank.finish();np.savez_compressed(out/'batch.npz',**arr)
    report=dict(schema='wide_cuda_wrist_query_v2',role='collection',scope='reset_to_wrist_hold_or_termination',
                full_episode=False,eligible_success_demonstration=False,checkpoint=p['parent_sha'],seeds=seeds,
                num_envs=n,physics_device='cuda:0',action_dim=56,physics_hz=240,control_hz=30,
                elapsed_rollout_s=time.monotonic()-t0,planner_init_s=plan_elapsed,recipe='native_env_ids_two_contact_views',eligible_per_stratum=bank.seen,solver=e.readback,cross_lane_contacts=e.cross,
                student_start_phase=handoff,teacher_assisted_lanes=[i for i in range(n) if handoff[i]],reset_count_per_lane=1,phase_steps=phase_steps,
                ends=[r or 'step_limit' for r in reasons],source_scene_sha256=sha(source/'dynamic_training_scene.usda'),
                samples=len(arr['x']),data_sha256=sha(out/'batch.npz'),plan_sha256=sha(args.plan))
    write(out/'report.json',report);print('GPU_COLLECTION_COMPLETE',n,'lanes',report['samples'],'rows','planning_s',plan_elapsed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=3500)
    args=p.parse_args()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    from isaac_compat import close_application
    try:sample(args)
    except BaseException:traceback.print_exc();raise
    finally:close_application(app)
