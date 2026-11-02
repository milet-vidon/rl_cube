"""Evaluate BC waypoint network plus contact supervision, without runtime IK/teacher."""
from pathlib import Path
import argparse,json,time,sys,os,threading,shutil
import numpy as np
import torch
from side_gate import require_side_gate,sha
from side_waypoints import WaypointNet
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--seeds',required=True);p.add_argument('--split',choices=['development','test','replay'],required=True);p.add_argument('--record',action='store_true');a=p.parse_args()
gate=require_side_gate();ROOT=Path(__file__).resolve().parent;run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if saved['side_teacher_gate_sha256']!=sha(ROOT/'side_teacher_gate.json'):raise ValueError('Model safety contract mismatch')
for f,h in saved['inference_code_sha256'].items():
    if sha(ROOT/f)!=h:raise ValueError('Inference code mismatch: '+f)
net=WaypointNet(saved['mean'],saved['scale']);net.load_state_dict(saved['model']);net.eval()
def write(n,x):(run/n).write_text(json.dumps(x,indent=2,allow_nan=False))
write('evaluation_config.json',vars(a)|{'kind':'BC_waypoint_network_with_contact_supervisor','runtime_teacher_or_IK':False,'pure_end_to_end_PPO':False,'checkpoint_sha256':sha(a.checkpoint)})
shutil.copy2(a.checkpoint,run/'checkpoint.pt')
for f in ['side_policy_evaluate.py','side_waypoints.py','side_control.py','self_contact_audit.py','side_video.py','config.json']:shutil.copy2(ROOT/f,run/f)
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import tensor,close_application
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None;started=time.monotonic()
try:
 from environment import CubePickupEnv
 from side_control import SegmentController
 from acceptance import assess_pickup
 from self_contact_audit import SelfContactAudit
 from PIL import Image,ImageDraw
 env=CubePickupEnv(vision=a.record,preview=a.record,auto_reset=False);audit=SelfContactAudit();env.max_episode_length=1200;env.cfg['success']['hold_seconds']=3.
 write('runtime_config.json',env.cfg);write('asset_contract.json',env.contract);results=[]
 for seed in map(int,a.seeds.split(',')):
    env.rng=np.random.default_rng(seed);env.reset();env.metrics={};env.pair_impulses={}
    xyz=tensor(env.cube.data.root_pos_w)[0].numpy().copy();x=torch.tensor([[(xyz[0]-.48)/.017,xyz[1]/.017]],dtype=torch.float32)
    if float(x.abs().max())>1.:raise ValueError('Outside calibrated training workspace')
    with torch.no_grad():goals=net(x)[0].numpy()
    meta=saved['metadata'];begin=np.array(meta['home']);segments=[]
    for phase,goal,steps in zip(meta['names'],goals,meta['steps']):segments.append((phase,begin,goal,steps));begin=goal
    controller=SegmentController(env,segments);rows=[];frames=[]
    for step in range(1200):
        audit.reset_step();_,_,done,_=env.step(controller.action());row=dict(step=step,phase=controller.name,**env.metrics);row.update(audit.metrics());row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();rows.append(row)
        if step%150==0:print('PROGRESS',seed,step,controller.name,row['height_m'],flush=True)
        if a.record and step%3==0:
            im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,28),fill='black');d.text((8,8),f'BC WAYPOINTS + CONTACT GATES | seed {seed} | {controller.name}',fill='white')
            d.rectangle((0,480,640,512),fill='black');d.text((8,490),f'LIFT {row["height_m"]*100:.1f} cm | HOLD {row["hold_steps"]/30:.2f} s | TABLE {row["robot_table_impulse"]:.6f} Ns',fill='white');frames.append(im)
        unsafe=row['self_contact_impulse']>1e-6 or row['arm_body_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
        if done[0] or unsafe or controller.failure_reason:break
    accepted=assess_pickup(rows,env.cfg)
    if sum(r['arm_body_impulse'] for r in rows)>1e-6:accepted['accepted']=False;accepted['failure_reasons'].append('arm_body_contact')
    if sum(r['self_contact_impulse'] for r in rows)>1e-6:accepted['accepted']=False;accepted['failure_reasons'].append('robot_self_contact')
    lifted=next((i for i,r in enumerate(rows) if r['height_m']>.005 and not r['cube_table_contact']),len(rows))
    result={'seed':seed,'kind':'BC_waypoint_network_with_contact_supervisor','split':a.split,'acceptance':accepted,'controller_failure':controller.failure_reason,'max_height_m':max(r['height_m'] for r in rows),'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse'] for r in rows),'arm_body_impulse_sum_Ns':sum(r['arm_body_impulse'] for r in rows),'self_contact_impulse_sum_Ns':sum(r['self_contact_impulse'] for r in rows),'returned_after_any_lift':any(r['cube_table_contact'] for r in rows[lifted:]),'steps':len(rows),'initial_xyz':xyz.tolist()}
    results.append(result);write(f'trajectory_{seed}.json',rows);write('result.json',{'trials':results,'accepted':sum(r['acceptance']['accepted'] for r in results),'total':len(results),'checkpoint_sha256':sha(a.checkpoint),'elapsed_s':time.monotonic()-started,'split':a.split,'runtime_teacher_or_IK':False,'pure_end_to_end_PPO':False});print('EPISODE',json.dumps(result),flush=True)
    if frames:
        im=frames[-1];d=ImageDraw.Draw(im);d.text((450,490),'PASS' if accepted['accepted'] else 'FAIL',fill='white')
        frames[0].save(run/f'rollout_{seed}.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
        from side_video import encode_mp4
        encode_mp4(run/f'rollout_{seed}.mp4',frames+[frames[-1]]*15)
        env.save_rgb(run/f'chest_{seed}.png')
finally:
 code=1 if sys.exc_info()[0] else 0;threading.Timer(15,lambda:os._exit(code)).start()
 if env:env.close()
 close_application(app)
