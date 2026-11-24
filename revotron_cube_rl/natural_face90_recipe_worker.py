"""Continuous R27 -> natural presentation -> measured-state PPO specialist.

Original V5 safety observer and R27 prefix are imported unchanged. Only the
post-presentation right-joint command source changes. No post-reset pose writes.
"""
from pathlib import Path
import argparse
import json
import math
import os
import time
from natural_face90_v5_worker import (
    ROOT, PARENT_HASH, PLAN_HASH, REPORT_HASH, inputs, measurements, require,
    require_locks, sha, read, save, PostPhase as ScriptedPostPhase,
)


class PostPhase(ScriptedPostPhase):
    def __init__(self, e, req, out):
        import torch
        from natural_face90_recipe_frames import contact_frame
        from natural_face90_v5_metrics import rotate
        from natural_face90_recipe_policy import PPO
        from natural_face90_recipe_guide import ArmExpert, IndexApproach, restore_expert
        super().__init__(e, req, out)
        from natural_face90_recipe_long import fingerprint, worker_software
        require(worker_software(fingerprint())==req['software'], 'Sources changed during simulator startup')
        require(all(req['software'].get(k)==v for k,v in self.software.items()),
                'Imported observer/controller source mismatch')
        self.software = req['software']
        self.phase_names += ('ppo_contact', 'ppo_turn', 'ppo_retreat')
        self.learning = torch.zeros(e.n, dtype=torch.bool, device=e.home.device)
        self.learn_steps = torch.zeros(e.n, dtype=torch.long, device=e.home.device)
        self.rids = [i for i,k in enumerate(e.names) if k.startswith('right_') or k.endswith('_R')]
        require(len(self.rids)==28, 'Wrong right-hand canonical action subset')
        torch.manual_seed(req['policy_seed'])
        self.guide = IndexApproach(self.controller)
        self.model = ArmExpert([self.rids.index(i) for i in self.guide.arm]).to(e.home.device)
        self.algorithm = PPO(self.model)
        self.training = req['learn']
        self.contract = dict(schema=req['recipe_schema'], parent_sha=PARENT_HASH, software=req['software'],
            joint_names=e.names, right_indices=self.rids, target_deg=15, action_dim=56,
            learned_action_dim=7, observation_kind='privileged_simulation_state',
            left_support='frozen measured natural handoff target',
            no_post_reset_pose_writes=True, arm_indices=self.guide.arm,
            action_mode='bounded_residual_on_index_pad_ik')
        if req['checkpoint'] is not None:
            path = Path(req['checkpoint'])
            require(sha(path)==req['checkpoint_sha'], 'Changed PPO checkpoint')
            ck = torch.load(path, map_location=e.home.device, weights_only=True)
            restore_expert(self.model, self.algorithm.optimizer, ck, self.contract, req)
        # The reset/physics random seed is independent of the policy sampling RNG.
        torch.manual_seed(req['policy_seed'] + req['iteration']*104729 + (0 if self.training else 1))
        self.model.eval()
        self.data = []; self.pending = None
        self.reward_log = (out/'policy_transitions.jsonl').open('x', buffering=1)
        self.max_steps = 2200
        self.handoff_target = e.home.clone()
        self.cumulative_reward = torch.zeros(e.n, device=e.home.device)
        self.best_distance = torch.full((e.n,), float('inf'), device=e.home.device)
        calibration = ROOT/'face_contact_calibration_v1.json'
        require(sha(calibration)=='b9ab73c9e7e2c922ffaf1270196fce8c2a4acd635dfe8842def32d98bcbe7b44',
                'Changed simulated fingertip contact anchor')
        cal = read(calibration)
        body, off, quat = contact_frame(e.stage, '/World/envs/env_0000/Robot', cal['source_link'])
        names = list(e.robot.shared_metatype.link_names)
        self.pad_body = names.index(body)
        qt = torch.tensor([*quat[1:],quat[0]], dtype=torch.float64, device=e.home.device)
        self.pad_offset = torch.tensor(off, dtype=torch.float64, device=e.home.device) + rotate(
            qt, torch.tensor(cal['point_local'], dtype=torch.float64, device=e.home.device))

    def state(self, target, remaining):
        import torch
        from natural_face90_v5_metrics import rotate, conj
        from natural_face90_recipe_policy import pack_observation, potential
        e=self.e; m=self.obs
        core, pieces = self.monitor.cube()
        bodies = e.robot.get_link_transforms().double()
        tip = bodies[:,self.pad_body,:3] + rotate(bodies[:,self.pad_body,3:], self.pad_offset)
        local_tip = rotate(conj(core[:,3:]), tip-core[:,:3])
        self.pad_geometry = core, tip, local_tip
        dip = bodies[:,self.monitor.fingers,:3]
        tip_local = rotate(conj(core[:,None,3:]), dip-core[:,None,:3])
        # A surface sticker attached to the selected face. Its position follows
        # the measured layer angle; it is never a prescribed cube pose.
        radial = self.controller.radial
        theta = -m['face_deg'].double()*math.pi/180
        x,y = radial.unbind(-1)
        xy = torch.stack((theta.cos()*x-theta.sin()*y,
                          theta.sin()*x+theta.cos()*y), -1)*.030
        reached = m['contact_goal_reached'].bool()
        z = torch.where(reached, -.100, -.045).double()
        marker = torch.cat((xy, z[:,None]), -1)
        delta = marker-local_tip
        distance = delta.norm(dim=-1)
        p = torch.cat((self.monitor.layer.measured_pos, self.monitor.layer.measured_q), -1)
        c = core.clone(); c[:,:3] -= e.offset
        obs,critic = pack_observation(e.robot.get_dof_positions()[:,e.ids],
            e.robot.get_dof_velocities()[:,e.ids], target, c, tip_local, delta,
            m['face_deg'], m['progress'], reached, remaining, p, self.monitor.r.acc,
            e.views['Core'].get_velocities())
        phi = potential(distance.float(), m['progress'].float(), m['hold_fraction'].float(),
                        m['actual_retraction_m'].float(), reached)
        return obs,critic,phi,distance

    def propose(self, target, active, steps):
        import torch
        # Handoff before the scripted hover/contact, after measured natural
        # readiness and the collision-avoiding forward detour/outside approach.
        eligible = active & self.obs['natural_ready'].bool() & self.controller.detour_done & (self.controller.phase>=2)
        fresh = eligible & ~self.learning
        self.handoff_target[fresh] = target[fresh]
        self.learning |= eligible
        learned = active & self.learning
        command = super().propose(target, active & ~self.learning, steps)
        self.pending = None
        if not learned.any(): return command
        remaining = ((self.max_steps-steps).float()/self.max_steps).clamp(0,1)
        obs, critic, phi, distance = self.state(target, remaining)
        latent, logp, value = self.model.sample(obs, critic, deterministic=not self.training)
        q = self.e.robot.get_dof_positions()[:,self.e.ids]
        proposed = self.guide.command(target, q, self.handoff_target, fresh, learned, latent,
                                      *self.pad_geometry, self.obs)
        command = torch.where(learned[:,None], proposed, command)
        retreat = self.obs['contact_goal_reached'].bool()
        self.retract = torch.where(learned, retreat, self.retract)
        self.action_phase = torch.where(learned, torch.where(retreat,8,torch.where(self.obs['turn_started'],7,6)), self.action_phase)
        self.learn_steps += learned.long()
        self.best_distance = torch.where(learned,torch.minimum(self.best_distance,distance.float()),self.best_distance)
        self.pending = dict(obs=obs,critic=critic,latent=latent,logp=logp,value=value,
                            valid=learned.clone(),potential=phi,command=command.clone())
        return command

    def after_step(self, before_post, alive, unsafe_seen, reasons, steps):
        import torch
        from natural_face90_recipe_policy import step_reward
        if self.pending is None: return
        p=self.pending; valid=p['valid']; done=valid & ~alive
        _,_,phi,distance = self.state(p['command'], ((self.max_steps-steps).float()/self.max_steps).clamp(0,1))
        success = torch.tensor([r=='post_turn_verified' for r in reasons],device=self.device)
        reward = step_reward(p.pop('potential'),phi,done,unsafe_seen,success,p['latent'])
        reward = torch.where(valid,reward,0.)
        p.pop('command'); p.update(reward=reward,done=done)
        if self.training: self.data.append({k:v.detach().clone() for k,v in p.items()})
        self.cumulative_reward += reward
        for i in (done | (valid & (steps%60==0))).nonzero(as_tuple=True)[0].tolist():
            self.reward_log.write(json.dumps(dict(lane=i,seed=self.req['seeds'][i],
                post_step=int(steps[i]),learned_steps=int(self.learn_steps[i]),
                reward=float(reward[i]),return_sum=float(self.cumulative_reward[i]),
                contact_marker_distance_m=float(distance[i]),unsafe=bool(unsafe_seen[i]),
                terminal=bool(done[i]),reason=reasons[i]),allow_nan=False)+'\n')
        self.pending=None

    def record_state(self, mask, step, target, event, reason=None, prefix=None):
        import torch
        # Script IK goal/error fields are stale after handoff. Store the learned
        # policy's measured contact-marker error instead of mislabelling them.
        super().record_state(mask & ~self.learning,step,target,event,reason,prefix)
        learned=mask & self.learning
        if not learned.any():return
        _,_,_,distance=self.state(target,torch.zeros(self.n,device=self.device))
        q=self.e.robot.get_dof_positions()[:,self.e.ids]
        dq=self.e.robot.get_dof_velocities()[:,self.e.ids]
        core=self.e.core();bodies=self.body_poses()
        for i in learned.nonzero(as_tuple=True)[0].tolist():
            row=dict(event=event,step=step,lane=i,seed=self.req['seeds'][i],phase=self.stage_name(i),
                reason=reason if isinstance(reason,str) else reason[i] if reason is not None else None,
                q=q[i].tolist(),dq=dq[i].tolist(),command=target[i].tolist(),core=core[i].tolist(),
                joint_names=self.e.names,body_names=self.body_names,body_poses=bodies[i].tolist(),
                post_controller_started=True,learned_policy_active=True,
                learned_contact_marker_error_m=float(distance[i]),learned_steps=int(self.learn_steps[i]))
            self.traces.write(json.dumps(row,allow_nan=False)+'\n')

    def finish_learning(self):
        import torch
        from natural_face90_recipe_long import fingerprint, worker_software
        require(worker_software(fingerprint())==self.req['software'], 'Sources changed during rollout')
        self.reward_log.close()
        stats=dict(kind='state_PPO_specialist', learned_lanes=int(self.learning.sum()),
                   learned_transitions=int(self.learn_steps.sum()), deterministic=not self.training,
                   input_checkpoint=self.req['checkpoint'], input_checkpoint_sha=self.req['checkpoint_sha'],
                   actor_updated=False, task_success_claim=False,
                   transition_evidence_sha=sha(self.out/'policy_transitions.jsonl'))
        if self.training:
            require(bool(self.data),'No measured policy handoff; inspect prefix/natural/outside phases')
            data={k:torch.stack([r[k] for r in self.data]) for k in self.data[0]}
            stats.update(self.algorithm.update(data),actor_updated=True)
            checkpoint=self.out/'checkpoint.pt'
            with checkpoint.open('xb') as f:
                torch.save(dict(contract=self.contract,policy=self.model.state_dict(),
                    optimizer=self.algorithm.optimizer.state_dict(),iteration=self.req['iteration'],
                    trained_seeds=self.req['seeds'],prior_checkpoint_sha=self.req['checkpoint_sha'],
                    status='unvalidated_state_specialist'),f)
            stats.update(output_checkpoint=str(checkpoint),output_checkpoint_sha=sha(checkpoint))
        return stats

def run(req,out):
    import torch
    from contact_hold90_cuda_domain import make_plan,make_teacher,FAILURES90
    from natural_face90_recipe_contacts import open_scene,step_physics
    from contact_hold90_cuda_sample import local_palm_reader,HoldMeter
    from phase_contact_policy import PHASES
    from joint_safety import project_target

    source,grasp,meta,actor=inputs(req);seeds=req['seeds'];n=len(seeds)
    actor=actor.cuda().eval();ids=meta['output_indices']
    e=open_scene(source,n,grasp)
    save(out/'contact_capacity.json',e.domain_receipt['raw_contact_capacity'])
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
            step_physics(e,cmd,out);target=cmd;feedback=e.contacts
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
            new.after_step(before_post,post,unsafe_seen,reason,step_q)
            if step%120==0 or transfer.any() or time.monotonic()-last_save>90:
                event=dict(step=step,prefix=int(prefix.sum()),post=int(post.sum()),entered_post=int(admitted.sum()),closed=sum(bool(x) for x in reason),elapsed_s=round(time.monotonic()-started,2))
                fp.write(json.dumps(event)+'\n');fp.flush();save(out/'progress.json',event);print('NATURAL90_STEP',event,flush=True);last_save=time.monotonic()
    require(not (prefix|post).any() and all(reason),'Unresolved episode termination')
    learning=new.finish_learning()
    new.evidence.close();new.incidents.close();new.traces.close()
    rows=[]
    for i,s in enumerate(seeds):
        row=new.row(i);row.update(seed=s,lane=i,parameters=req['parameters'][i],prefix_steps=int(step_p[i]),post_steps=int(step_q[i]),steps=int(step_p[i]+step_q[i]),prefix_endpoint_held=bool(admitted[i]),target_deg=req['target_deg'],success=(reason[i]=='post_turn_verified'),unsafe=bool(unsafe_seen[i]),end_reason=reason[i])
        if row['success']:require(row['natural_ready'] and row['turn_started'] and not row['unsafe'],'Invalid physical success evidence')
        rows.append(row)
    report=dict(schema='natural_face90_rollout_v1',role=req['role'],seeds=seeds,rows=rows,parent_sha=PARENT_HASH,source_plan_sha=PLAN_HASH,
        order='R27 student pickup from reset, measured endpoint/steady-hold gate, post policy in same continuously stepped scene',
        action_dim=56,control_hz=30,physics_hz=240,reset_count=1,simulation_device='cuda:0',physics_stepped=True,face='nZ',domain_receipt=e.domain_receipt,
        controller_evidence_sha256=sha(out/'controller_states.jsonl'),collision_evidence_sha256=sha(out/'collision_events.jsonl'),request_sha256=sha(out/'request.json'),post_evidence_sha256=sha(out/'post_measurements.jsonl'),reference_report_sha256=REPORT_HASH,expert_learning=learning,accepted_model=False,simulation_only=True,cross_lane_contacts=e.cross,software=new.software,elapsed_s=time.monotonic()-started)
    require(not e.cross,'Cross lane contacts');save(out/'report.json',report);print('NATURAL90_COMPLETE',n,sum(r['success'] for r in rows),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); req=read(args.request)
    from natural_face90_recipe_long import fingerprint, worker_software, validate_request
    validate_request(req)
    require(req['software']==worker_software(fingerprint()), 'Worker source pins changed')
    require_locks()
    out=args.out; out.mkdir(parents=True,exist_ok=False);save(out/'request.json',req)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    try:
        import torch
        with torch.no_grad(): run(req,out)
    except BaseException as error:
        save(out/'failed.json',dict(error=repr(error)));raise
    finally:
        from isaac_compat import close_application
        close_application(app)


if __name__=='__main__': main()

