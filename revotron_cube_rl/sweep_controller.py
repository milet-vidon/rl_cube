from acceptance import assess_pickup
from isaac_compat import close_application
from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
from pathlib import Path
import argparse,json,os,sys,time,threading,itertools,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--visual',action='store_true');p.add_argument('--name',default='controller_sweep');p.add_argument('--candidate',type=int);p.add_argument('--seeds',default='17');p.add_argument('--approach-height',type=float,default=.068);p.add_argument('--pregrasp-scale',type=float,default=.35);p.add_argument('--tilt-deg',type=float,default=12.);p.add_argument('--grasp',choices=['horizontal','downward','staged_parallel','staged_oblique'],default='horizontal');p.add_argument('--widths');p.add_argument('--hold-seconds',type=float);p.add_argument('--linear-speed-max',type=float);p.add_argument('--angular-speed-max',type=float);args=p.parse_args()
if args.visual and any(v is not None for v in (args.hold_seconds,args.linear_speed_max,args.angular_speed_max)):p.error('Use unmodified config for visual training demonstrations; stronger gates are diagnostic only.')
ROOT=Path(__file__).resolve().parent;out=ROOT/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
for source in ('config.json','traditional_controller.py'):shutil.copy2(ROOT/source,out/source)
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app=AppLauncher(headless=True,enable_cameras=args.visual,kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
from environment import CubePickupEnv
from traditional_controller import GraspController,DownwardController,StagedDownwardController
import numpy as np,torch
from PIL import Image
env=None
try:
    env=CubePickupEnv(vision=args.visual,preview=args.visual,auto_reset=False);env.max_episode_length=900
    for key,value in [('hold_seconds',args.hold_seconds),('linear_speed_max',args.linear_speed_max),('angular_speed_max',args.angular_speed_max)]:
        if value is not None:env.cfg['success'][key]=value
    (out/'asset_contract.json').write_text(json.dumps(env.contract,indent=2))
    (out/'runtime_config.json').write_text(json.dumps(env.cfg,indent=2))
    candidates=[dict(width=w,backoff=b,curl=c,thumb=t,relax=r) for w,b,c,t,r in [
        (.04,.065,.15,1.7,.002),(.04,.065,.25,1.7,.003),(.04,.065,0.,1.7,.003),(.04,.065,0.,1.7,.006),
        (.038,.08,.25,1.4,.003),(.045,.10,.35,1.2,.003),(.055,.12,.4,1.2,.002),(.04,.065,.15,1.7,0.)]]
    if args.grasp in ('downward','staged_parallel','staged_oblique'):
        candidates=[dict(width=w,relax=r) for w,r in [(.102,0.),(.100,0.),(.098,0.),(.100,.002)]]
    if args.widths:candidates=[dict(width=float(w),relax=0.) for w in args.widths.split(',')]
    results=[];observations=[];critics=[];actions=[];episode_ids=[]
    for seed in map(int,args.seeds.split(',')):
      for idx,params in enumerate(candidates):
        if args.candidate is not None and idx!=args.candidate:continue
        params=dict(params,approach_height=args.approach_height)
        started=time.monotonic();env.rng=np.random.default_rng(seed);env.reset();env.pair_impulses={}
        if args.grasp=='downward': controller=DownwardController(env,**params)
        elif args.grasp=='staged_parallel': controller=StagedDownwardController(env,grasp_mode='parallel',pregrasp_scale=args.pregrasp_scale,**params)
        elif args.grasp=='staged_oblique': controller=StagedDownwardController(env,grasp_mode='oblique',tilt_deg=args.tilt_deg,pregrasp_scale=args.pregrasp_scale,**params)
        else: controller=GraspController(env,**params)
        rows=[];frames=[]
        for step in range(920 if args.grasp in ('downward','staged_parallel','staged_oblique') else 390):
            obs=env.get_observations();a=controller.action()
            if args.visual:
                observations.append(obs['policy'][0].numpy().copy());critics.append(obs['critic'][0].numpy().copy());actions.append(a[0].numpy().copy());episode_ids.append(len(results))
            _,_,done,_=env.step(a);rows.append(dict(phase=controller.name,**env.metrics,
                q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),targets=env.targets[0].tolist(),
                palm_centers={s:env.palm_pos(s)[0].tolist() for s in ('left','right')}))
            if step%150==0:print('PROGRESS',seed,idx,step,controller.name,round(env.metrics['height_m'],4),flush=True)
            if args.visual and step%3==0:frames.append(Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].cpu().numpy()))
            if done[0] or getattr(controller,'failure_reason',None):break
        result={'runtime_versions':runtime_versions(),'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'candidate':idx,'seed':seed,'grasp':args.grasp,'parameters':params,'success_criteria':env.cfg['success'],'success':bool(env.metrics['success']),'max_height_m':max(r['height_m'] for r in rows),'max_hold_steps':max(r['hold_steps'] for r in rows),'robot_table_impulse_sum':sum(r['robot_table_impulse'] for r in rows),'contact_pairs':env.pair_impulses,'steps':len(rows),'elapsed_s':time.monotonic()-started}
        result['acceptance']=assess_pickup(rows,env.cfg)
        result['controller_kind']='traditional_teacher'
        result['controller_failure']=getattr(controller,'failure_reason',None)
        result['controller_sha256']=hashlib.sha256((ROOT/'traditional_controller.py').read_bytes()).hexdigest()
        results.append(result);print('TRIAL',json.dumps(result),flush=True)
        (out/f'trajectory_{idx}_{seed}.json').write_text(json.dumps(rows))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        (out/'result.json').write_text(json.dumps({'trials':results,'accepted':all(r['acceptance']['accepted'] for r in results),'checkpoint':None,'checkpoint_reason':'traditional controller; no learned model','policy_success_rate':None},indent=2))
        if args.visual:
            env.save_rgb(out/f'chest_{idx}_{seed}.png')
            frames[0].save(out/f'controller_{idx}_{seed}.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    if args.visual:
        good=[i for i,r in enumerate(results) if r['acceptance']['accepted']]
        mask=np.isin(episode_ids,good)
        if np.any(mask):np.savez_compressed(out/'demonstrations.npz',policy_obs=np.asarray(observations)[mask],critic_obs=np.asarray(critics)[mask],actions=np.asarray(actions)[mask],episode_id=np.asarray(episode_ids)[mask],config_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest())
finally:
    code=1 if sys.exc_info()[0] else 0
    def end():time.sleep(15);os._exit(code)
    threading.Thread(target=end,daemon=True).start()
    if env:env.close()
    close_application(app)
