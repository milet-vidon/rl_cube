"""Bounded state-only training, separate development/test seeds, audited demonstrations."""
from pathlib import Path
import argparse,json,time,sys,os,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('mode',choices=['collect','bc','evaluate','ppo'])
p.add_argument('--name',required=True);p.add_argument('--seeds',default='4001,4002,4003,4004,4005,4006,4007,4008')
p.add_argument('--data');p.add_argument('--checkpoint');p.add_argument('--epochs',type=int,default=600)
p.add_argument('--iterations',type=int,default=24);p.add_argument('--steps',type=int,default=600)
p.add_argument('--record',action='store_true');p.add_argument('--split',choices=['training','development','test'],default='training')
p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--bc-noise',type=float,default=0.)
p.add_argument('--bc-lr',type=float,default=5e-4)
p.add_argument('--bc-final-lr',type=float,default=5e-4)
p.add_argument('--drop-command-history',action='store_true')
p.add_argument('--freeze-constant-targets',action='store_true')
p.add_argument('--bounded-targets',action='store_true')
p.add_argument('--teacher-tail-steps',type=int,default=0)
a=p.parse_args();seeds=list(map(int,a.seeds.split(',')))
from stage_gate import require_staged_gate,sha256
teacher_gate=require_staged_gate()
run=ROOT/'runs'/a.name;run.mkdir(parents=True,exist_ok=False)
shutil.copy2(ROOT/'config.json',run/'config.json')
config_hash=sha256(ROOT/'config.json');training_config=vars(a)|{'observation_schema':'state261_absolute56_v1','actor_ground_truth':True,'teacher_actions_during_policy_evaluation':False,'config_sha256':config_hash,'num_envs':1,'gpu_count':1,'critic_warmup_iterations':4,'ppo_clip':.05,'ppo_entropy':0.,'ppo_noise_std':.01,'unsafe_terminal_penalty':-2.}
(run/'training_config.json').write_text(json.dumps(training_config,indent=2))
(run/'provenance.json').write_text(json.dumps({'teacher_gate_sha256':sha256(ROOT/'staged_teacher_gate.json'),
    'validated_code_sha256':teacher_gate['identity'],'assets_sha256':teacher_gate['assets'],
    'soft_limit_config_sha256':config_hash,'rsl_compat_sha256':sha256(ROOT/'rsl_compat.py')},indent=2))
for f in ['state_learning.py','state_policy.py']:shutil.copy2(ROOT/f,run/f)
import torch
from tensordict import TensorDict
from state_policy import StatePolicy,observations,SCHEMA
from rsl_compat import PPO
from acceptance import assess_pickup
from isaac_compat import tensor,runtime_versions
started=time.monotonic();torch.manual_seed(73);np.random.seed(73)

def write(name,data):
    (run/name).write_text(json.dumps(data,indent=2,allow_nan=False))

def save(policy,iteration,extra=None):
    torch.save({'policy':policy.state_dict(),'schema':SCHEMA,'config_sha256':config_hash,'runtime_versions':runtime_versions(),
        'training_config_sha256':sha256(run/'training_config.json'),'code_sha256':{f:sha256(run/f) for f in ['state_learning.py','state_policy.py']},'mean':policy.actor.mean,'std':policy.actor.std,'center':policy.actor.center,'scale':policy.actor.scale,
        'algorithm':a.mode,'bounded_targets':policy.actor.bounded,'iteration':iteration,'training_seeds':seeds,'status':'state_policy_not_yet_independently_validated',**(extra or {})},run/'checkpoint.pt')

def load_policy(obs):
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved['schema']!=SCHEMA or saved['config_sha256']!=config_hash:raise ValueError('Checkpoint contract mismatch')
    if a.split in ('development','test') and set(seeds)&set(saved['training_seeds']):raise ValueError('Evaluation seed leakage')
    if a.mode=='evaluate':shutil.copy2(a.checkpoint,run/'checkpoint.pt')
    policy=StatePolicy(obs,*[saved[x] for x in ['mean','std','center','scale']],bounded=saved.get('bounded_targets',False));policy.load_state_dict(saved['policy']);return policy,saved

if a.mode=='bc':
    d=np.load(a.data,allow_pickle=False);x=torch.from_numpy(d['obs']);targets=torch.from_numpy(d['targets'])
    seeds=sorted(set(map(int,d['seed'])))
    if str(d['config_sha256'])!=config_hash:raise ValueError('Data/config mismatch')
    center=targets.mean(0);scale=targets.std(0).clamp_min(.1)
    if a.bounded_targets:
        low,high=targets.min(0).values,targets.max(0).values
        center=(low+high)/2;scale=(high-low)/2
    if a.freeze_constant_targets:
        scale[(targets.max(0).values-targets.min(0).values)<1e-6]=0.
    obs=TensorDict({'policy':x[:1],'critic':x[:1].clone()},batch_size=[1])
    obs_std=x.std(0).clamp_min(.05)
    # Prevent the actor from merely copying its own previous command. Physical
    # joint/cube state remains available; privileged state is explicitly declared.
    if a.drop_command_history:obs_std[112:224]=1e6
    policy=StatePolicy(obs,x.mean(0),obs_std,center,scale,a.bounded_targets)
    noise_std=x.std(0).clamp_min(.05)*a.bc_noise
    noise_std[112:224]=0.;noise_std[247:]=0.
    opt=torch.optim.Adam(policy.actor.parameters(),lr=a.bc_lr);history=[]
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.bc_final_lr)
    # Only training rows enter optimization. Closed-loop development is the selection criterion.
    for epoch in range(a.epochs):
        perm=torch.randperm(len(x));loss_sum=0.
        for ids in perm.split(512):
            inputs=x[ids]+torch.randn_like(x[ids])*noise_std if a.bc_noise else x[ids]
            pred=policy.actor.targets(policy.actor(inputs));loss=((pred-targets[ids])/.025).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);opt.step();loss_sum+=float(loss.detach())*len(ids)
        scheduler.step()
        if (epoch+1)%100==0 or epoch==0:
            history.append({'epoch':epoch+1,'training_loss':loss_sum/len(x),'elapsed_s':time.monotonic()-started});print('BC',history[-1],flush=True)
            save(policy,epoch+1,{'source_data_sha256':sha256(a.data)})
            write('result.json',{'kind':'BC_training_only','history':history,'task_success_claim':False,'training_seeds':seeds})
    save(policy,a.epochs,{'source_data_sha256':sha256(a.data)})
    write('result.json',{'kind':'BC_training_only','history':history,'task_success_claim':False,'training_seeds':seeds,'checkpoint_sha256':sha256(run/'checkpoint.pt'),'elapsed_s':time.monotonic()-started})
    raise SystemExit(0)

from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import close_application
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None
try:
    from environment import CubePickupEnv
    from traditional_controller import StagedDownwardController
    from PIL import Image,ImageDraw
    env=CubePickupEnv(vision=a.record,preview=a.record,auto_reset=False);env.max_episode_length=900
    write('asset_contract.json',env.contract)
    all_obs=[];all_targets=[];all_seeds=[];results=[];policy=None
    def reset(seed):
        env.rng=np.random.default_rng(seed);env.reset();env.metrics={};env.pair_impulses={}
        initial=tensor(env.cube.data.root_pos_w).clone()
        return initial,observations(env,initial)
    if a.mode=='ppo':
        initial,obs=reset(seeds[0]);policy,saved=load_policy(obs)
        with torch.no_grad():policy.log_std.fill_(np.log(.01))
        alg=PPO(policy,num_learning_epochs=2,num_mini_batches=4,learning_rate=a.lr,entropy_coef=0.,gamma=env.cfg.get('discount',.998),schedule='fixed',clip_param=.05,device='cpu')
        alg.init_storage('rl',1,a.steps,obs,[56]);episode=0;rows=[];completed=[]
        for iteration in range(a.iterations):
            # Warm up only the critic on the cloned actor's rollouts first.
            warmup=iteration<4
            for parameter in policy.actor.parameters():parameter.requires_grad_(not warmup)
            policy.log_std.requires_grad_(not warmup)
            before=next(policy.actor.net.parameters()).detach().clone()
            for step in range(a.steps):
                with torch.no_grad():
                    action=alg.act(obs);target=policy.actor.targets(action)
                    _,reward,done,extra=env.step((target-env.targets)/env.scales)
                    rows.append(dict(step=len(rows),**env.metrics))
                    unsafe=env.metrics['robot_table_impulse']>1e-6 or env.metrics['arm_soft_limit_hit'] or env.metrics['actual_arm_soft_limit_violation'] or any(env.metrics['hard_limit_hit'])
                    if unsafe:
                        done=torch.ones(1,dtype=torch.bool)
                        reward=reward-2.-env.cfg.get('discount',.998)*env.metrics['terms']['potential']
                    new=observations(env,initial);alg.process_env_step(new,reward,done,extra)
                    if done[0]:
                        outcome=assess_pickup(rows,env.cfg);completed.append({'episode':episode,'seed':seeds[episode%len(seeds)],'acceptance':outcome})
                        write(f'trajectory_{episode}.json',rows);rows=[];episode+=1;initial,new=reset(seeds[episode%len(seeds)])
                    obs=new
            with torch.no_grad():alg.compute_returns(obs)
            losses=alg.update()
            change=float((next(policy.actor.net.parameters()).detach()-before).abs().max())
            entry={'iteration':iteration,'critic_warmup':warmup,'loss':losses,'actor_delta':change,'completed_episodes':len(completed),'accepted_episodes':sum(r['acceptance']['accepted'] for r in completed),'elapsed_s':time.monotonic()-started}
            results.append(entry);save(policy,iteration,{'optimizer':alg.optimizer.state_dict(),'parent_checkpoint_sha256':sha256(a.checkpoint),'training_seeds':sorted(set(seeds)|set(saved['training_seeds']))})
            write('result.json',{'kind':'state_PPO','iterations':results,'episodes':completed,'task_success_claim':False});print('PPO',entry,flush=True)
        if rows:write(f'trajectory_{episode}_partial.json',rows)
        write('result.json',{'kind':'state_PPO','iterations':results,'episodes':completed,'task_success_claim':False,'checkpoint_sha256':sha256(run/'checkpoint.pt'),'elapsed_s':time.monotonic()-started})
    else:
        for seed in seeds:
            initial,obs=reset(seed);rows=[];frames=[];episode_obs=[];episode_targets=[];first_success=None
            teacher=StagedDownwardController(env,width=.096) if a.mode=='collect' else None
            if a.mode=='evaluate' and policy is None:policy,saved=load_policy(obs);policy.eval()
            for step in range(900):
                with torch.no_grad():
                    if teacher:action=teacher.action()
                    else:action=(policy.actor.targets(policy.act_inference(obs))-env.targets)/env.scales
                    episode_obs.append(obs['policy'][0].numpy().copy())
                    _,_,done,_=env.step(action);episode_targets.append(env.targets[0].numpy().copy())
                    rows.append(dict(step=step,phase=teacher.name if teacher else 'learned_state_policy',**env.metrics))
                    obs=observations(env,initial)
                if a.record and step%3==0:
                    frame=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].cpu().numpy())
                    method='TEACHER' if teacher else ('PPO STATE POLICY' if saved.get('parent_checkpoint_sha256') else 'BC STATE POLICY')
                    draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,30),fill='black');draw.text((8,8),f"{method} | seed {seed} | step {step}",fill='white')
                    draw.rectangle((0,480,640,512),fill='black')
                    draw.text((8,489),f"lift {env.metrics['height_m']*100:.1f} cm | hold {env.metrics['hold_steps']/env.cfg['control_hz']:.2f} s | robot-table {env.metrics['robot_table_impulse']:.5f} Ns",fill='white');frames.append(frame)
                unsafe=env.metrics['robot_table_impulse']>1e-6 or env.metrics['arm_soft_limit_hit'] or env.metrics['actual_arm_soft_limit_violation'] or any(env.metrics['hard_limit_hit'])
                finished=bool(done[0])
                if teacher and a.teacher_tail_steps and env.metrics['success']:
                    if first_success is None:first_success=step
                    finished=step>=first_success+a.teacher_tail_steps
                if finished or unsafe or (teacher and teacher.failure_reason):break
            outcome=assess_pickup(rows,env.cfg)
            result={'seed':seed,'split':a.split,'kind':'teacher' if teacher else 'pure_state_policy','acceptance':outcome,'steps':len(rows),'max_height_m':max(r['height_m'] for r in rows),'robot_table_impulse_sum':sum(r['robot_table_impulse'] for r in rows)}
            lifted_at=next((i for i,r in enumerate(rows) if r['height_m']>=env.cfg['success']['lift_m']),len(rows))
            result['returned_to_table_after_lift']=any(r['cube_table_contact'] for r in rows[lifted_at:])
            result['drop_flag_definition']='Environment drop flag means below table minus 12 cm or excessive lateral escape; return to tabletop is reported separately.'
            results.append(result);write(f'trajectory_{seed}.json',rows);print('EPISODE',json.dumps(result),flush=True)
            if teacher and outcome['accepted']:
                all_obs.extend(episode_obs);all_targets.extend(episode_targets);all_seeds.extend([seed]*len(rows))
            if frames:
                label='SUCCESS' if outcome['accepted'] else 'FAILED: '+','.join(outcome['failure_reasons'])
                draw=ImageDraw.Draw(frames[-1]);draw.rectangle((0,480,640,512),fill='black');draw.text((8,490),label,fill='white')
                frames[0].save(run/f'rollout_{seed}.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
                env.save_rgb(run/f'chest_{seed}.png')
            write('result.json',{'kind':'teacher_dataset' if teacher else 'pure_state_policy_evaluation','split':a.split,'trials':results,'accepted':sum(r['acceptance']['accepted'] for r in results),'total':len(results),'elapsed_s':time.monotonic()-started,'checkpoint_sha256':sha256(a.checkpoint) if a.checkpoint else None,'actor_observation':'privileged simulator state, own control history, initial cube position and clock; no teacher actions'})
        if all_obs:
            np.savez_compressed(run/'state_demos.npz',obs=np.asarray(all_obs,dtype=np.float32),targets=np.asarray(all_targets,dtype=np.float32),seed=np.asarray(all_seeds),config_sha256=config_hash)
finally:
    import threading
    code=1 if sys.exc_info()[0] else 0
    threading.Timer(15,lambda:os._exit(code)).start()
    if env:env.close()
    close_application(app)
