"""Development transfer to a freely articulated cube before robot face pushing."""
from pathlib import Path
import argparse,json,hashlib,shutil,os,time,sys,threading,traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=9501);p.add_argument('--record',action='store_true');p.add_argument('--max-steps',type=int,default=3500);p.add_argument('--friction',default='.9,.8');a=p.parse_args()
from evidence_paths import local_path,raw_sha
cert=json.loads((ROOT/'wrist_direction_gate_certificate.json').read_text())
if not cert['passed'] or cert['independent_accepted']<5:raise RuntimeError('Missing accepted wrist baseline')
for name,expected in cert['source_sha256'].items():
    if raw_sha(ROOT/name)!=expected:raise RuntimeError('Wrist baseline source changed: '+name)
if raw_sha(ROOT/'wrist_direction_protocol_v1.json')!=cert['protocol_sha256']:raise RuntimeError('Wrist protocol changed')
for name,expected in cert['assets_sha256'].items():
    if raw_sha(local_path(name,ROOT))!=expected:raise RuntimeError('Wrist baseline asset changed: '+name)
for trial in cert['trials']:
    d=local_path(trial['run'],ROOT)
    for name,key in [('config.json','config_sha256'),('runtime_config.json','runtime_config_sha256'),('asset_contract.json','asset_contract_sha256'),(f"trajectory_{trial['seed']}.json",'trajectory_sha256')]:
        if raw_sha(d/name)!=trial[key]:raise RuntimeError('Wrist evidence changed: '+str(d/name))
run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False)
def write(n,d):(run/n).write_text(json.dumps(d,indent=2,allow_nan=False))
sources=list(cert['source_sha256'])+['articulated_cube_env.py','layer_turn_metrics.py','single_face_experiment.py','dynamic_cube.py','wrist_gate.py']
for n in sources:shutil.copy2(ROOT/n,run/n)
write('experiment.json',vars(a)|{'split':'development','precondition_certificate_sha256':raw_sha(ROOT/'wrist_direction_gate_certificate.json'),'source_sha256':{n:raw_sha(ROOT/n)for n in sources},'direct_cube_control':False})
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import close_application,tensor,quaternion_from_native
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None;started=time.monotonic();rows=[];frames=[]
try:
    import torch
    from PIL import Image,ImageDraw
    from articulated_cube_env import DynamicPickupEnv
    from wrist_direction import WristDirectionTeacher
    from wrist_audit import assess_wrist,wrist_geometry
    from elbow_audit import measured_elbow
    from palm_handover_audit import measured_hand_frames,assess_handover,tilt_error
    from finger_contact_audit import FingerContactAudit
    from palm_contact_audit import DetailedSelfContactAudit
    from layer_turn_metrics import measure_layers
    env=DynamicPickupEnv(run,friction=tuple(map(float,a.friction.split(','))),seed=a.seed,vision=a.record,preview=a.record,auto_reset=False,arm_damping=8.)
    env.max_episode_length=a.max_steps;env.cfg['success']['hold_seconds']=3.
    write('config.json',env.cfg);write('asset_contract.json',env.contract)
    write('cube_asset.json',{'source_sha256':raw_sha(env.dynamic_source),'model':env.cube.model,'soft_limit_version':'revo3_user_soft_limits_v2'})
    env.rng=np.random.default_rng(a.seed);env.reset();env.metrics={};env.pair_impulses={}
    teacher=WristDirectionTeacher(env,ROOT/'opposition_grasp_v20.json');fingers=FingerContactAudit();audit=DetailedSelfContactAudit();failure=None
    initial_reset_count=env.cube.pose_reset_count
    if a.record:
        env.overview.set_world_poses_from_view(torch.tensor([[1.65,0.,1.45]]),torch.tensor([[.40,.06,1.16]]));env._render()
        render=env._render;counter=[0]
        def sampled_render():
            if counter[0]%3==0:render()
            counter[0]+=1
        env._render=sampled_render
    for step in range(a.max_steps):
        fingers.reset_step();audit.reset_step()
        try:action=teacher.action()
        except ValueError as e:failure=str(e);break
        env.step(action)
        row=dict(step=step,phase=teacher.name,**env.metrics);row.update(fingers.metrics());row.update(audit.metrics())
        row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();row['cube_quaternion_wxyz']=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).tolist()
        row.update(tilt_degrees=-90,right_contact_axis=0,right_thumb_sign=-1,measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env),tilt_reference_wxyz=teacher.reference.tolist()if teacher.reference is not None else None)
        row['measured_wrist']=wrist_geometry(row,env.cfg['joint_names']);row['nominal_target']=teacher.nominal_target[0].tolist()
        row['cubelets']=env.cube.states();row['layer_measurement']=measure_layers(row['cubelets']);row['pose_reset_count']=env.cube.pose_reset_count;rows.append(row)
        teacher.finger_contacts=row['finger_contact_impulses'];teacher.contact_points=row['finger_contact_points']
        unsafe=row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
        if unsafe:failure='robot_safety_gate'
        if row['drop']:failure='cube_drop'
        if teacher.failure_reason:failure=teacher.failure_reason
        if row['layer_measurement']['maximum_relative_rotation_deg']>30:failure='cube_layers_deformed_before_turn'
        if env.cube.pose_reset_count!=initial_reset_count:raise RuntimeError('Unexpected cube pose write during control')
        stopping=bool(failure or teacher.accepted)
        if step%120==0 or stopping:print('STEP',a.seed,step,row['phase'],'height',round(row['height_m'],3),'layer_deviation',round(row['layer_measurement']['maximum_relative_rotation_deg'],2),'failure',failure,flush=True)
        if a.record and (step%3==0 or stopping):
            im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,28),fill='black');d.text((6,8),f'DYNAMIC CUBE | {teacher.name} | seed {a.seed}',fill='white');d.rectangle((0,480,640,512),fill='black');d.text((6,490),f'LAYER DRIFT {row["layer_measurement"]["maximum_relative_rotation_deg"]:.1f} deg | HOLD {teacher.hold/30:.2f}s',fill='white');frames.append(im)
        if stopping:break
    if not teacher.accepted and failure is None:failure='episode_step_budget'
    measured=assess_handover(rows,env.cfg['joint_names'],regrasp=False,full_episode=True)if rows else {'accepted':False}
    wrist=assess_wrist([r if r['layer_measurement']['assembled_for_grasp']else r|{'phase':'cube_not_assembled'}for r in rows],env.cfg['joint_names'])if rows else {'accepted':False}
    clean_hold=0;longest=0
    for row in rows:
        clean_hold=clean_hold+1 if row['phase']=='tilt_hold'and row['layer_measurement']['assembled_for_grasp']else 0;longest=max(longest,clean_hold)
    grasp_passed=bool(teacher.accepted and not failure and measured['accepted']and wrist['accepted']and longest>=90)
    result={'seed':a.seed,'kind':'dynamic_cube_left_grasp_transfer','split':'development','full_episode':True,'dynamic_grasp_passed':grasp_passed,'face_turn_validated':False,'failure':failure,'steps':len(rows),'elapsed_s':time.monotonic()-started,'measured':measured,'wrist':wrist,'assembled_hold_s':longest/30,'final_layers':rows[-1]['layer_measurement']if rows else None,'cube_pose_writes_during_control':env.cube.pose_reset_count-initial_reset_count,'checkpoint':None,'neural_policy_trained':False,'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse']for r in rows),'self_contact_impulse_sum_Ns':sum(r['self_contact_impulse']for r in rows)}
    write('trajectory.json',rows);write('result.json',result)
    if teacher.path is not None:write('left_plan.json',teacher.path.tolist())
    if frames:
        frames[-1].save(run/'final.png')
        from side_video import encode_mp4
        encode_mp4(run/'demo.mp4',frames+[frames[-1]]*15)
    print('RESULT',json.dumps(result),flush=True)
except Exception as e:
    traceback.print_exc();write('error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
finally:
    code=1 if sys.exc_info()[0]else 0;threading.Timer(15,lambda:os._exit(code)).start()
    if env:env.close()
    close_application(app)
