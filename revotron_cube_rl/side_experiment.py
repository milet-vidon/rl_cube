"""Audited new-pose teacher development. No original teacher certificate is reused."""
from pathlib import Path
import argparse,json,time,sys,os,threading,shutil,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seeds',default='4201')
p.add_argument('--widths',default='.096');p.add_argument('--offsets',default='0.');p.add_argument('--record',action='store_true')
p.add_argument('--wrist',type=float,default=35.);p.add_argument('--tilt',type=float,default=15.)
p.add_argument('--finger-mode',choices=['curled','flat'],default='curled')
p.add_argument('--hold',type=float,default=3.);a=p.parse_args()
ROOT=Path(__file__).resolve().parent;run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False)
def write(n,x):(run/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
write('experiment.json',vars(a));shutil.copy2(ROOT/'config.json',run/'config.json')
for f in ['side_grasp.py','side_control.py','self_contact_audit.py','side_experiment.py']:shutil.copy2(ROOT/f,run/f)
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import tensor,close_application
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None;started=time.monotonic()
try:
 from environment import CubePickupEnv
 from side_grasp import SideTeacher
 from acceptance import assess_pickup
 from self_contact_audit import SelfContactAudit
 from PIL import Image,ImageDraw
 env=CubePickupEnv(vision=a.record,preview=a.record,auto_reset=False);audit=SelfContactAudit();env.max_episode_length=1200;env.cfg['success']['hold_seconds']=a.hold
 write('runtime_config.json',env.cfg);write('asset_contract.json',env.contract)
 results=[]
 for seed in map(int,a.seeds.split(',')):
  for width in map(float,a.widths.split(',')):
   for offset in map(float,a.offsets.split(',')):
    idx=len(results);env.rng=np.random.default_rng(seed);env.reset();env.metrics={};env.pair_impulses={}
    teacher=SideTeacher(env,width,offset,a.wrist,a.tilt,a.finger_mode);rows=[];frames=[]
    for step in range(1200):
     audit.reset_step();action=teacher.action();_,_,done,_=env.step(action)
     row=dict(step=step,phase=teacher.name,segment=teacher.index,tick=teacher.tick,**env.metrics)
     row.update(audit.metrics());row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();rows.append(row)
     unsafe=row['self_contact_impulse']>1e-6 or row['arm_body_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
     if step%150==0:print('PROGRESS',seed,idx,step,teacher.name,round(row['height_m'],4),row['contact_impulse'],flush=True)
     if a.record and step%3==0:
      im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,26),fill='black');d.text((8,7),f'SIDE TEACHER | seed {seed} | {teacher.name} | {row["height_m"]*100:.1f} cm',fill='white');frames.append(im)
     if done[0] or unsafe or teacher.failure_reason:break
    outcome=assess_pickup(rows,env.cfg)
    if sum(r['arm_body_impulse'] for r in rows)>1e-6:outcome['accepted']=False;outcome['failure_reasons'].append('arm_body_contact')
    if sum(r['self_contact_impulse'] for r in rows)>1e-6:outcome['accepted']=False;outcome['failure_reasons'].append('robot_self_contact')
    result={'seed':seed,'candidate':idx,'width':width,'offset':offset,'steps':len(rows),'acceptance':outcome,'controller_failure':teacher.failure_reason,'max_height_m':max(r['height_m'] for r in rows),'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse'] for r in rows),'arm_body_impulse_sum_Ns':sum(r['arm_body_impulse'] for r in rows),'self_contact_impulse_sum_Ns':sum(r['self_contact_impulse'] for r in rows),'contact_pairs':env.pair_impulses,'kind':'traditional_side_teacher','elapsed_s':time.monotonic()-started}
    results.append(result);write(f'trajectory_{idx}_{seed}.json',rows);write('result.json',{'trials':results,'accepted':sum(r['acceptance']['accepted'] for r in results),'total':len(results),'controller_sha256':sha(ROOT/'side_grasp.py'),'config_sha256':sha(run/'runtime_config.json'),'elapsed_s':time.monotonic()-started});print('EPISODE',json.dumps(result),flush=True)
    if frames:
     label='SUCCESS' if outcome['accepted'] else 'FAILED: '+','.join(outcome['failure_reasons'])
     d=ImageDraw.Draw(frames[-1]);d.rectangle((0,480,640,512),fill='black');d.text((8,490),label,fill='white')
     frames[0].save(run/f'rollout_{idx}_{seed}.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
     env.save_rgb(run/f'chest_{idx}_{seed}.png')
finally:
 code=1 if sys.exc_info()[0] else 0;threading.Timer(15,lambda:os._exit(code)).start()
 if env:env.close()
 close_application(app)
