"""Transport first, then oppose the right thumb and four fingers around the target layer."""
from pathlib import Path
import json,os,subprocess,time
import numpy as np,torch
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw
from isaac_compat import tensor,quaternion_from_native
from layer_turn_metrics import measure_layers
from wrist_audit import wrist_geometry
from elbow_audit import measured_elbow
from palm_handover_audit import measured_hand_frames
from tilt_audit import contact_topology
ROOT=Path(__file__).resolve().parent

def host_plan(script,request,run,label,timeout=240):
 (run/(label+'_request.json')).write_text(json.dumps(request,indent=2));host=os.environ.copy()
 for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD'):host.pop(key,None)
 host['PYTHONPATH']='/opt/ros/jazzy/lib/python3.12/site-packages';host['LD_LIBRARY_PATH']='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu'
 with (run/(label+'_preflight.log')).open('w')as log:
  try:p=subprocess.run(['/usr/bin/python3',str(ROOT/script),str(run/(label+'_request.json')),str(run/(label+'_plan.json'))],env=host,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  except subprocess.TimeoutExpired:return {'passed':False,'error':'preflight_timeout'}
 dest=run/(label+'_plan.json')
 if not dest.exists():return {'passed':False,'error':'preflight_process_error','exit_code':p.returncode}
 plan=json.loads(dest.read_text())
 if p.returncode:plan['passed']=False
 return plan

def append_row(env,teacher,fingers,audit,rows,run,phase,target):
 row=dict(step=len(rows),phase=phase,**env.metrics);row.update(fingers.metrics());row.update(audit.metrics())
 row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();row['cube_quaternion_wxyz']=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).tolist()
 row.update(tilt_degrees=-90,right_contact_axis=1,right_thumb_sign=-1,measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env),tilt_reference_wxyz=teacher.reference.tolist())
 row['measured_wrist']=wrist_geometry(row,env.cfg['joint_names']);row['nominal_target']=target.tolist();row['cubelets']=env.cube.states();row['layer_measurement']=measure_layers(row['cubelets']);row['pose_reset_count']=env.cube.pose_reset_count;rows.append(row)
 with (run/'trajectory.jsonl').open('a')as journal:journal.write(json.dumps(row,allow_nan=False)+'\n')
 return row

def safety_failure(row):
 if row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit']or row['actual_arm_soft_limit_violation']or any(row['hard_limit_hit']):return 'robot_safety_gate'
 if row['drop']or row['cube_table_contact']:return 'cube_drop_or_table_contact'
 if row['layer_measurement']['maximum_joint_origin_separation_m']>.002:return 'cube_joint_separation'
 return None

def frame(env,frames,args,row,label):
 if not args.record or row['step']%3:return
 im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,28),fill='black');d.text((6,8),f'FIVE-FINGER OPPOSITION | {row["phase"]} | seed {args.seed}',fill='white');d.rectangle((0,480,640,512),fill='black');d.text((6,490),label,fill='white');frames.append(im)

def execute(env,teacher,fingers,audit,rows,frames,run,args,stopped=lambda:False):
 first=len(rows);initial=rows[-1];names=env.cfg['joint_names'];request={'initial':dict(zip(names,initial['q'])),'joint_names':names,'cube_position':initial['cube_position'],'cube_quaternion':initial['cube_quaternion_wxyz'],'center_root':[.40,0.,.15]}
 print('CENTER_PREFLIGHT_BEGIN',flush=True);plan=host_plan('centered_transport_plan.py',request,run,'center')
 if not plan['passed']:return {'executed':False,'qualified':False,'failure':'center_preflight_rejected','detail':plan}
 print('CENTER_PREFLIGHT_PASSED',plan['checks'],flush=True);fixed=env.targets[0].numpy().copy();indices=[i for i,n in enumerate(names)if n.endswith('_L')];failure=None;lost=0;hold=0;longest=0;reset=env.cube.pose_reset_count;goal=np.array(plan['center_target_world_m']);rotation=R.from_quat(np.array(plan['cube_target_quaternion_wxyz'])[[1,2,3,0]])
 for phase,path in plan['phases']:
  for tick,point in enumerate(path):
   fingers.reset_step();audit.reset_step();target=fixed.copy();target[indices]=np.array(point)[indices];env.step((torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales);row=append_row(env,teacher,fingers,audit,rows,run,phase,target)
   failure=safety_failure(row)
   if stopped():failure='execution_interrupted'
   primary=contact_topology(row,'left');retained=primary['thumb']and sum(primary[f]for f in ('index','middle','ring','little'))>=3;lost=0 if retained else lost+1
   if lost>=10:failure='left_grasp_lost_during_center_transport'
   if row['layer_measurement']['maximum_relative_rotation_deg']>15:failure='cube_layers_deformed_during_center_transport'
   wrist=row['measured_wrist'];elbow=row['measured_elbow'];palm=np.array(row['measured_hand_frames']['left_palm_link']);pose_error=np.degrees((R.from_quat(np.array(row['cube_quaternion_wxyz'])[[1,2,3,0]])*rotation.inv()).magnitude())
   correct=all(primary.values())and row['height_m']>=.08 and row['linear_speed']<.03 and row['angular_speed']<.20 and row['layer_measurement']['assembled_for_grasp']and abs(row['cube_position'][1])<.03 and np.linalg.norm(np.array(row['cube_position'])-goal)<.015 and pose_error<5 and wrist['joint6_deg']>=5 and wrist['side_bend_deg']<=-5 and wrist['wrist_inside_elbow_m']>=.05 and (elbow['elbow_root_m'][1]-elbow['shoulder_root_m'][1])>=.16 and palm[1,0]<=-.6
   hold=hold+1 if phase=='center_hold'and correct and not failure else 0;longest=max(longest,hold)
   if env.cube.pose_reset_count!=reset:raise RuntimeError('Unexpected cube pose write during center transport')
   if tick%90==0 or failure:print('CENTER',row['step'],phase,'y',round(row['cube_position'][1],3),'hold',hold,'failure',failure,flush=True)
   frame(env,frames,args,row,f'CENTER Y {row["cube_position"][1]:+.3f}m | HOLD {hold/30:.1f}s | LAYER DRIFT {row["layer_measurement"]["maximum_relative_rotation_deg"]:.1f}')
   if failure:break
  if failure:break
 center={'passed':not failure and longest>=90,'longest_qualified_hold_s':longest/30,'initial_position_m':initial['cube_position'],'final_position_m':rows[-1]['cube_position'],'target_position_m':goal.tolist(),'plan_checks':plan['checks'],'failure':failure or ('center_hold_gate_not_met'if longest<90 else None)}
 (run/'center_result.json').write_text(json.dumps(center,indent=2))
 if not center['passed']or args.center_only:return {'executed':True,'qualified':False,'center':center,'failure':center['failure'],'scope':'center_transport_only','opposition_grasp_passed':False}
 from opposed_grasp_runtime import execute as execute_right
 return execute_right(env,teacher,fingers,audit,rows,frames,run,args,center,stopped)
