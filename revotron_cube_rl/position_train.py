from isaac_compat import close_application
"""Isaac rollout for the explicit cube-position actor."""
from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
from pathlib import Path
import argparse, json, time, hashlib, faulthandler
import numpy as np
faulthandler.dump_traceback_later(90, repeat=True)
parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['train','evaluate'],default='evaluate'); parser.add_argument('--iterations',type=int,default=1); parser.add_argument('--steps',type=int,default=600); parser.add_argument('--seed',type=int,default=17); parser.add_argument('--checkpoint',required=True); parser.add_argument('--learning-rate',type=float,default=5e-5); parser.add_argument('--noise-std',type=float); parser.add_argument('--name',required=True); parser.add_argument('--record',action='store_true'); parser.add_argument('--action-gain',type=float,default=1.0); parser.add_argument('--initial-action-clip',type=float,default=1.0); parser.add_argument('--initial-action-steps',type=int,default=0); args=parser.parse_args()
if args.steps<1 or args.iterations<1: raise ValueError('steps and iterations must be positive')
ROOT=Path(__file__).resolve().parent; run=ROOT/'runs'/args.name; run.mkdir(parents=True,exist_ok=True)
from stage_gate import require_visual_training_gate
if args.mode=='train': require_visual_training_gate()
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app=AppLauncher(headless=True,enable_cameras=True,kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
print('APP_READY',flush=True)
import torch
from environment import CubePickupEnv
from position_policy import PositionActorCritic, rgb_to_position_features
from absolute_policy import AbsolutePositionActorCritic
from rsl_compat import PPO

def position_obs(obs, resolution, feature_dim=6, progress=None, latched_features=None):
    out=obs.clone(); raw=obs['policy']; feat=latched_features if latched_features is not None else rgb_to_position_features(raw[:,224:],resolution)
    if feature_dim == 7:
        if progress is None: raise ValueError('Phase-conditioned checkpoint requires episode progress')
        feat=torch.cat((feat, progress.reshape(-1,1).to(feat.device,dtype=feat.dtype)),dim=-1)
    if feat.shape[-1] != feature_dim: raise ValueError(f'Expected {feature_dim} position features, got {feat.shape[-1]}')
    out['policy']=torch.cat((raw[:,:224],feat),-1); return out

env=None
try:
    env=CubePickupEnv(args.seed,vision=True,preview=args.record)
    config_sha=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(); raw=env.get_observations()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('config_sha256')!=config_sha: raise ValueError('Checkpoint/config contract mismatch')
    absolute_mode = saved.get('output_mode') == 'absolute_joint_target'
    open_loop_phase = bool(saved.get('open_loop_phase',False)); latch_position = bool(saved.get('latch_position',False))
    feature_dim=int(saved.get('position_feature_dim',6)); phase_horizon=int(saved.get('phase_horizon_steps',env.max_episode_length)); progress=env.episode_length_buf.float()/max(1,phase_horizon-1)
    def prepare_raw(value):
        if not open_loop_phase: return value
        value=value.clone(); value['policy'][:,:224]=0.; return value
    latched_features = rgb_to_position_features(raw['policy'][:,224:],env.cfg['camera_resolution']) if latch_position else None
    obs=position_obs(prepare_raw(raw),env.cfg['camera_resolution'],feature_dim,progress,latched_features)
    if absolute_mode:
        policy=AbsolutePositionActorCritic(obs, saved['joint_center'], saved['joint_half_range'],position_dim=feature_dim)
    else:
        policy=PositionActorCritic(obs)
    policy.load_state_dict(saved['policy']); policy.train(args.mode == 'train')
    alg=PPO(policy,num_learning_epochs=3,num_mini_batches=4,learning_rate=args.learning_rate,entropy_coef=.0005,gamma=env.cfg.get('discount',.998),schedule='fixed',device='cpu')
    optimizer_resume='fresh_actor_critic_optimizer'
    if args.mode=='train' and 'optimizer' in saved:
        saved_runtime=saved.get('asset_contract',{}).get('runtime_versions',{})
        if saved_runtime.get('rsl-rl-lib') != runtime_versions()['rsl-rl-lib']:
            raise ValueError('PPO optimizer runtime mismatch; re-warmstart in the current RSL-RL environment')
        try:
            alg.optimizer.load_state_dict(saved['optimizer']); optimizer_resume='resumed'
        except (ValueError, KeyError, RuntimeError):
            optimizer_resume='fresh_after_supervised_checkpoint'
    for g in alg.optimizer.param_groups: g['lr']=args.learning_rate
    if args.noise_std is not None:
        if not np.isfinite(args.noise_std) or args.noise_std <= 0: raise ValueError('noise-std must be positive')
        with torch.no_grad(): policy.log_std.fill_(np.log(args.noise_std))
    alg.init_storage('rl',1,args.steps,obs,[56])
    evidence={'runtime_versions':runtime_versions(),'mode':args.mode,'seed':args.seed,'real_rgb':True,'action_dim':56,'position_feature_dim':feature_dim,'phase_horizon_steps':phase_horizon,'open_loop_phase':open_loop_phase,'latch_position':latch_position,'output_mode':'absolute_joint_target' if absolute_mode else 'incremental_joint_target','config_sha256':config_sha,'iterations':[],'task_success_claim':False,'noise_std':args.noise_std,'optimizer_resume':optimizer_resume}
    with torch.no_grad():
        act=policy.act_inference(obs); hidden=obs.clone(); hidden['critic']=hidden['critic']+1.; evidence['privileged_actor_leak_max']=float((act-policy.act_inference(hidden)).abs().max()); blank=obs.clone(); blank['policy'][:,224:]=0.; evidence['blank_position_action_delta']=float((act-policy.act_inference(blank)).abs().max())
    start=time.time(); trajectory=[]; frames=[]; iterations=args.iterations if args.mode=='train' else 1
    for it in range(iterations):
        reward_sum=0.; before=torch.cat([p.detach().flatten().clone() for p in policy.actor.parameters()])
        for step in range(args.steps):
            with torch.no_grad():
                action=alg.act(obs) if args.mode=='train' else policy.act_inference(obs)
                if absolute_mode:
                    target=policy.actor.decode_targets(action)
                    if args.action_gain != 1.0:
                        current=env.targets.to(target.device)
                        target=current + args.action_gain*(target-current)
                    action=(target-env.targets.to(target.device))/env.scales.to(target.device)
                else:
                    action=action * args.action_gain
                if step < args.initial_action_steps:
                    action=action.clamp(-args.initial_action_clip,args.initial_action_clip)
                obs_raw,reward,done,extra=env.step(action); progress=env.episode_length_buf.float()/max(1,phase_horizon-1)
                if latch_position and bool(done[0]): latched_features=rgb_to_position_features(obs_raw['policy'][:,224:],env.cfg['camera_resolution'])
                obs=position_obs(prepare_raw(obs_raw),env.cfg['camera_resolution'],feature_dim,progress,latched_features);
            if args.mode=='train': alg.process_env_step(obs,reward,done,extra)
            reward_sum+=float(reward[0]); trajectory.append(dict(iteration=it,step=step,**env.metrics))
            if args.record and len(trajectory)<=env.max_episode_length and step%3==0: frames.append(__import__('PIL').Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].cpu().numpy()))
        loss={}
        if args.mode=='train':
            with torch.no_grad(): alg.compute_returns(obs)
            loss=alg.update()
        delta=float((torch.cat([p.detach().flatten() for p in policy.actor.parameters()])-before).abs().max()); rec={'iteration':it,'mean_reward':reward_sum/args.steps,'loss':loss,'actor_parameter_max_delta':delta,'successes':env.successes,'episodes':env.episodes,'elapsed_s':time.time()-start}; evidence['iterations'].append(rec); print('ITERATION',json.dumps(rec),flush=True)
        if args.mode=='train':
            actor=policy.actor
            checkpoint={'policy':policy.state_dict(),'optimizer':alg.optimizer.state_dict(),'iteration':it,'config_sha256':config_sha,'asset_contract':env.contract,'seed':args.seed,'status':'absolute_target_ppo_not_validated_skill','position_features':True,'position_feature_dim':feature_dim,'phase_horizon_steps':phase_horizon,'open_loop_phase':open_loop_phase,'latch_position':latch_position,'output_mode':'absolute_joint_target' if absolute_mode else 'incremental_joint_target'}
            if absolute_mode:
                checkpoint.update({'joint_center':actor.joint_center.detach().cpu().tolist(),'joint_half_range':actor.joint_half_range.detach().cpu().tolist()})
            torch.save(checkpoint,run/'checkpoint.pt')
    evidence.update({'steps_total':len(trajectory),'successes':env.successes,'episodes':env.episodes,'phase_horizon_steps':phase_horizon,'elapsed_s':time.time()-start,'contact_events':env.contact_events,'last_metrics':env.metrics,'status':'completed'}); (run/'trajectory.json').write_text(json.dumps(trajectory,indent=2)); (run/'result.json').write_text(json.dumps(evidence,indent=2)); env.save_rgb(run/'chest_final.png'); print('RESULT',json.dumps(evidence),flush=True)
finally:
    import os, threading
    code=1 if __import__('sys').exc_info()[0] else 0; faulthandler.cancel_dump_traceback_later()
    def deadline(): time.sleep(15); os._exit(code)
    threading.Thread(target=deadline,daemon=True).start()
    if env is not None: env.close()
    close_application(app)
