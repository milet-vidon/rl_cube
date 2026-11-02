from isaac_compat import close_application
"""Real Isaac RGB -> 56 actions -> PhysX -> RSL-RL PPO, bounded by --iterations."""
from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
from pathlib import Path
import argparse, json, time, hashlib, faulthandler
import numpy as np
faulthandler.dump_traceback_later(90,repeat=True)
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['smoke','train','evaluate'],default='smoke')
parser.add_argument('--iterations',type=int,default=10)
parser.add_argument('--steps',type=int,default=32)
parser.add_argument('--seed',type=int,default=7)
parser.add_argument('--checkpoint')
parser.add_argument('--learning-rate',type=float,default=5e-5)
parser.add_argument('--noise-std',type=float,help='Override exploration std only when explicitly requested')
parser.add_argument('--name',default='pilot')
parser.add_argument('--record',action='store_true',help='Record first full episode horizon, including overview')
parser.add_argument('--no-vision',action='store_true',help='Physics diagnostics only; training refuses this option.')
parser.add_argument('--action-gain',type=float,default=1.0,
                    help='Multiply policy actions before the incremental target update (evaluation diagnostic)')
parser.add_argument('--initial-action-clip',type=float,default=1.0,
                    help='Clip normalized actions during the initial warmup steps (evaluation diagnostic)')
parser.add_argument('--initial-action-steps',type=int,default=0,
                    help='Number of initial control steps for --initial-action-clip')
args=parser.parse_args()
if args.mode!='smoke' and args.no_vision:raise ValueError('Visual training/evaluation requires the real chest camera')
if args.record and args.no_vision:raise ValueError('Recording requires cameras')
if args.steps<1 or args.iterations<1:raise ValueError('Steps and iterations must be positive')
if not np.isfinite(args.action_gain) or args.action_gain<=0 or args.action_gain>1:
    raise ValueError('action-gain must be finite and within (0,1]')
if not np.isfinite(args.initial_action_clip) or args.initial_action_clip<=0 or args.initial_action_clip>1 or args.initial_action_steps<0:
    raise ValueError('initial-action-clip must be in (0,1] and initial-action-steps nonnegative')
if args.mode=='train' and (args.steps<4 or args.steps%4):raise ValueError('PPO rollout steps must be a positive multiple of 4')
if args.mode=='evaluate' and not args.checkpoint:raise ValueError('Evaluation requires --checkpoint')
ROOT=Path(__file__).resolve().parent;run=ROOT/'runs'/args.name;run.mkdir(parents=True,exist_ok=True)
from stage_gate import require_visual_training_gate
if args.mode=='train': require_visual_training_gate()
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app=AppLauncher(headless=True,enable_cameras=not args.no_vision,kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
print('APP_READY',flush=True)
import torch
from environment import CubePickupEnv
from policy import VisionActorCritic
from rsl_compat import PPO
env=None
try:
    env=CubePickupEnv(args.seed,vision=not args.no_vision,preview=args.mode=='smoke' or args.record)
    faulthandler.cancel_dump_traceback_later()
    import shutil
    shutil.copy2(ROOT/'config.json',run/'config.json')
    (run/'asset_contract.json').write_text(json.dumps(env.contract,indent=2))
    obs=env.get_observations();initial=obs.clone()
    if not args.no_vision:
        env.save_rgb(run/'chest_initial.png')
        assert float(obs['policy'][:,env.proprio_dim:].std())>.015,'Black/constant camera image'
    config_sha=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()
    policy=VisionActorCritic(obs,resolution=env.cfg['camera_resolution'])
    alg=PPO(policy,num_learning_epochs=3,num_mini_batches=4,learning_rate=args.learning_rate,entropy_coef=.0005,gamma=env.cfg.get('discount',.998),schedule='fixed',device='cpu')
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert saved['config_sha256']==config_sha,'Checkpoint/config contract mismatch'
        policy.load_state_dict(saved['policy'])
        if args.mode=='train':
            saved_runtime=saved.get('asset_contract',{}).get('runtime_versions',{})
            if saved_runtime.get('rsl-rl-lib') != runtime_versions()['rsl-rl-lib']:
                raise ValueError('PPO optimizer runtime mismatch; re-warmstart in the current RSL-RL environment')
            alg.optimizer.load_state_dict(saved['optimizer'])
    for group in alg.optimizer.param_groups:group['lr']=args.learning_rate
    if args.noise_std is not None:
        assert args.noise_std>0
        with torch.no_grad():policy.log_std.fill_(np.log(args.noise_std))
    alg.init_storage('rl',1,args.steps,obs,[56])
    evidence={'runtime_versions':runtime_versions(),'mode':args.mode,'seed':args.seed,'real_rgb':not args.no_vision,'action_dim':56,'action_gain':args.action_gain,
              'initial_action_clip':args.initial_action_clip,'initial_action_steps':args.initial_action_steps,
              'config_sha256':config_sha,'iterations':[],'task_success_claim':False}
    # Verify that only privileged critic reacts to changing hidden object state.
    with torch.no_grad():
        a=policy.act_inference(obs);changed=obs.clone();changed['critic']+=1.
        evidence['privileged_actor_leak_max']=float((a-policy.act_inference(changed)).abs().max())
        changed=obs.clone();changed['policy'][:,env.proprio_dim:]=0.
        evidence['black_image_action_delta']=float((a-policy.act_inference(changed)).abs().max())
    start=time.time();trajectory=[];frames=[]
    iterations=args.iterations if args.mode=='train' else 1
    def progress(status):
        entry={'status':status,'mode':args.mode,'steps_completed':len(trajectory),'steps_planned':iterations*args.steps,
               'iterations_completed':len(evidence['iterations']),'successes':env.successes,'episodes':env.episodes,
               'elapsed_s':time.time()-start,'last_metrics':getattr(env,'metrics',{})}
        temporary=run/'progress.json.tmp';temporary.write_text(json.dumps(entry));temporary.replace(run/'progress.json')
    progress('running')
    for it in range(iterations):
        reward_sum=0.;params_before=torch.cat([p.detach().flatten().clone() for p in policy.actor.parameters()])
        for step in range(args.steps):
            with torch.no_grad():
                if args.mode=='train':action=alg.act(obs)
                elif args.mode=='evaluate':action=policy.act_inference(obs)
                else:action=torch.zeros(1,56)
                action = action * args.action_gain
                if step < args.initial_action_steps:
                    action = action.clamp(-args.initial_action_clip,args.initial_action_clip)
                obs,reward,done,extra=env.step(action)
                if args.mode=='train':alg.process_env_step(obs,reward,done,extra)
            reward_sum+=float(reward[0]);trajectory.append(dict(iteration=it,step=step,**env.metrics))
            if len(trajectory)%100==0:progress('running')
            if args.record and len(trajectory)<=env.max_episode_length and step%3==0:
                from PIL import Image
                frames.append(Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].cpu().numpy()))
        loss={}
        if args.mode=='train':
            with torch.no_grad():alg.compute_returns(obs)
            loss=alg.update()
        delta=float((torch.cat([p.detach().flatten() for p in policy.actor.parameters()])-params_before).abs().max())
        record={'iteration':it,'mean_reward':reward_sum/args.steps,'loss':loss,'actor_parameter_max_delta':delta,'successes':env.successes,'episodes':env.episodes,'elapsed_s':time.time()-start}
        assert all(np.isfinite(v) for v in loss.values()),loss
        evidence['iterations'].append(record);print('ITERATION',json.dumps(record),flush=True)
        if args.mode=='train':
            torch.save({'policy':policy.state_dict(),'optimizer':alg.optimizer.state_dict(),'iteration':it,'config_sha256':config_sha,'asset_contract':env.contract,'seed':args.seed,'status':'pilot_not_validated_skill'},run/'checkpoint.pt')
        (run/'result.json').write_text(json.dumps(evidence,indent=2))
        progress('running')
    if not args.no_vision:env.save_rgb(run/'chest_final.png')
    if frames:frames[0].save(run/'policy_rollout.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    evidence.update({'steps_total':len(trajectory),'successes':env.successes,'episodes':env.episodes,'elapsed_s':time.time()-start,'contact_events':env.contact_events,'last_metrics':env.metrics,'status':'completed'})
    (run/'trajectory.json').write_text(json.dumps(trajectory,indent=2));(run/'result.json').write_text(json.dumps(evidence,indent=2))
    progress('completed')
    print('RESULT',json.dumps(evidence),flush=True)
finally:
    import sys, os, threading, traceback
    exit_code=1 if sys.exc_info()[0] is not None else 0
    if exit_code:traceback.print_exc()
    faulthandler.cancel_dump_traceback_later()
    # This driver can hang in Kit shutdown after all work is saved. Bound cleanup
    # in this standalone child process; preserve failure exit codes.
    def cleanup_deadline():
        time.sleep(15)
        print('Kit cleanup timeout; terminating this completed child process',flush=True)
        os._exit(exit_code)
    threading.Thread(target=cleanup_deadline,daemon=True).start()
    if env is not None:env.close()
    close_application(app)
