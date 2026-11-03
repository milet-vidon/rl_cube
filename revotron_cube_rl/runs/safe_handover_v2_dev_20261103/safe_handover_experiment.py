"""Whole-cube tilt validation. This is not a Rubik's layer-turn success test."""
from pathlib import Path
import argparse,json,shutil,hashlib,time,sys,os,threading,traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--view',choices=['front','side'],default='front');p.add_argument('--name',required=True);p.add_argument('--seeds',required=True);p.add_argument('--shift',default='0,0,0');p.add_argument('--record',action='store_true');p.add_argument('--regrasp',action='store_true');p.add_argument('--development-resume',type=Path);p.add_argument('--contact-only',action='store_true');p.add_argument('--max-steps',type=int,default=5000);a=p.parse_args()
if a.development_resume and not a.regrasp:p.error('--development-resume requires --regrasp')
if a.contact_only:p.error('Contact-only restoration is not supported for corrected handover; use a full episode or --development-resume from tilt_hold')
from opposition_gate import require_opposition_gate
require_opposition_gate()
manifest=json.loads((ROOT/'wrist_collision_manifest.json').read_text())
for key in ['original_asset','override_asset']:
    asset=ROOT.parent/manifest[key]
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=manifest[key+'_sha256']:raise RuntimeError('Wrist asset hash changed')
if Path(os.environ.get('REVOTRON_USD_PATH','')).resolve()!=(ROOT.parent/manifest['override_asset']).resolve():raise RuntimeError('Safe handover requires the explicit repaired wrist USD')
run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False)
def write(name,obj):(run/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
sources=['safe_contact_candidate_v2.json','safe_path_audit.py','safe_handover.py','safe_collision_plan.py','safe_handover_experiment.py','wrist_collision_schema.py','wrist_collision_manifest.json','elbow_handover.py','elbow_audit.py','elbow_handover_experiment.py','evidence_paths.py','portability_migration.json','palm_contact_audit.py','palm_handover.py','palm_handover_audit.py','palm_handover_experiment.py','tilt_regrasp.py','tilt_collision_plan.py','tilt_experiment.py','tilt_audit.py','opposition_teacher.py','opposition_feedback.py','proximal_planner.py','opposition_grasp_v20.json','config.json','environment.py','joint_safety.py','kinematics.py','isaac_compat.py','finger_contact_audit.py','self_contact_audit.py']
for name in sources:shutil.copy2(ROOT/name,run/name)
arguments=vars(a)|{'development_resume':str(a.development_resume) if a.development_resume else None}
if a.development_resume:shutil.copy2(a.development_resume,run/'resume_state.json')
write('experiment.json',arguments|{'kind':'development_resume' if a.development_resume else ('whole_cube_tilt_regrasp' if a.regrasp else 'whole_cube_tilt_only'),'eligible_independent_evaluation':not bool(a.development_resume),'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},'right_regrasp_validated':False})
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import tensor,quaternion_from_native,close_application
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None;started=time.monotonic()
try:
    from environment import CubePickupEnv
    from safe_handover import SafeHandoverTeacher as TiltTeacher
    from tilt_regrasp import set_right_arm_gains
    from elbow_audit import measured_elbow,assess_elbow
    from finger_contact_audit import FingerContactAudit
    from palm_contact_audit import DetailedSelfContactAudit as SelfContactAudit
    from palm_handover_audit import assess_handover,hand_geometry,tilt_error,measured_hand_frames
    from kinematics import RobotKinematics
    geometry_kinematics=RobotKinematics()
    from PIL import Image,ImageDraw
    env=CubePickupEnv(vision=a.record,preview=a.record,auto_reset=False,arm_damping=8.);env.max_episode_length=a.max_steps;env.cfg['success']['hold_seconds']=3.
    fingers=FingerContactAudit();audit=SelfContactAudit();results=[]
    env.cfg['elbow_outward']={'shift_m':[-.045,.20,.055],'target_m':.18,'hold_min_outward_m':.16,'hold_min_outward_degrees':35.}
    env.cfg['post_tilt_left_finger_damping']=1.
    env.cfg['post_tilt_left_finger_stiffness']=20.
    env.contract['post_tilt_left_finger_damping']=1.
    env.contract['post_tilt_left_finger_stiffness']=20.
    env.cfg['post_tilt_right_arm_gains']={'stiffness':300.,'damping':20.,'effort_and_velocity_limits':'unchanged'}
    env.contract['post_tilt_right_arm_gains']=env.cfg['post_tilt_right_arm_gains']
    write('runtime_config.json',env.cfg);write('asset_contract.json',env.contract)
    for seed in map(int,a.seeds.split(',')):
        set_right_arm_gains(env,False)
        import torch
        finger_ids=[env.ids[i] for i,n in enumerate(env.cfg['joint_names']) if n.startswith(('left_','right_'))]
        env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(finger_ids)),30.,dtype=torch.float32),joint_ids=finger_ids)
        env.robot.write_joint_damping_to_sim_index(damping=torch.ones((1,len(finger_ids)),dtype=torch.float32),joint_ids=finger_ids)
        env.rng=np.random.default_rng(seed);env.reset();env.metrics={};env.pair_impulses={}
        if a.record:
            import torch
            eye=[1.65,0.,1.45] if a.view=='front' else [.75,1.60,1.45]
            env.overview.set_world_poses_from_view(torch.tensor([eye]),torch.tensor([[.40,.06,1.16]]));env._render()
        teacher=TiltTeacher(env,ROOT/'opposition_grasp_v20.json',tuple(map(float,a.shift.split(','))),regrasp=a.regrasp);rows=[];frames=[]
        if a.development_resume:
            import torch
            from isaac_compat import quaternion_to_native
            from tilt_regrasp import MirrorArrival
            initial=json.loads(a.development_resume.read_text())
            q=tensor(env.robot.data.default_joint_pos).clone();q[0,env.ids]=torch.tensor(initial['q'])
            env.robot.write_joint_state_to_sim(q,torch.zeros_like(q));env.targets=torch.tensor([initial['applied_target']])
            env.robot.set_joint_position_target(env.targets,joint_ids=env.ids)
            state=tensor(env.cube.data.default_root_state).clone();state[0,:3]=torch.tensor(initial['cube_position']);state[0,3:7]=quaternion_to_native(torch.tensor(initial['cube_quaternion_wxyz']));state[:,7:]=0.
            env.cube.write_root_state_to_sim(state);env.metrics=initial.copy()
            set_right_arm_gains(env,True)
            left_ids=[env.ids[i] for i,n in enumerate(env.cfg['joint_names']) if n.startswith('left_')]
            env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(left_ids)),20.,dtype=torch.float32),joint_ids=left_ids)
            for _ in range(2):env._physics()
            teacher.reference=np.array(initial['tilt_reference_wxyz']);teacher.path=np.array([initial['nominal_target']],dtype=np.float32);teacher.mode='regrasp';teacher.tick=1;teacher.planner.tilt_steps=1;teacher.planner.clear_steps=0
            teacher.finger_contacts=initial['finger_contact_impulses'];teacher.contact_points=initial['finger_contact_points']
            measured=dict(zip(env.cfg['joint_names'],initial['q']));nominal=dict(zip(env.cfg['joint_names'],initial['nominal_target']))
            if a.contact_only:
                teacher.right_phases=[('bilateral_hold',np.repeat(np.array([initial['nominal_target']],dtype=np.float32),181,axis=0))]
                teacher.regrasp_geometry={'Q':[[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]],'right_closed':initial.get('right_grasp_reference',{n:v for n,v in zip(env.cfg['joint_names'],initial['applied_target']) if n.startswith('right_')})}
            else:
                teacher.right_phases,teacher.regrasp_geometry=teacher.planner.regrasp_path(nominal,measured,initial['cube_position'],initial['cube_quaternion_wxyz'],teacher.contact_points)
            teacher.right_arrival=MirrorArrival(teacher.planner.k,env.cfg,teacher.planner.grasp,measured,initial['cube_position'],initial['cube_quaternion_wxyz'],teacher.regrasp_geometry,teacher.contact_points)
            for n,v in zip(env.cfg['joint_names'],initial['applied_target']):
                if n.startswith('left_'):teacher.base.finger_arrival.targets['_applied_'+n]=v
                if a.contact_only and n.startswith('right_'):teacher.right_arrival.targets['_applied_'+n]=v
        original_render=env._render
        if a.record:
            render_step=[0]
            def sampled_render():
                if render_step[0]%3==0:original_render()
                render_step[0]+=1
            env._render=sampled_render
        for step in range(a.max_steps):
            fingers.reset_step();audit.reset_step()
            try:action=teacher.action()
            except ValueError as error:
                teacher.failure_reason=str(error)
                if not rows:raise
                break
            env.step(action)
            row=dict(step=step,phase=teacher.name,**env.metrics);row.update(fingers.metrics());row.update(audit.metrics())
            row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();row['cube_quaternion_wxyz']=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).numpy().tolist()
            row['tilt_degrees']=-90;row['right_contact_axis']=teacher.planner.right_contact_axis;row['right_thumb_sign']=-1
            row['measured_hand_frames']=measured_hand_frames(env)
            row['measured_elbow']=measured_elbow(env)
            row['hand_geometry']=hand_geometry(row,env.cfg['joint_names'],geometry_kinematics)
            row['tilt_reference_wxyz']=teacher.reference.tolist() if teacher.reference is not None else None
            row['nominal_target']=teacher.nominal_target[0].tolist();row['tilt_error_deg']=teacher.tilt_error_deg;row['tilt_hold_steps']=teacher.hold;row['right_hold_steps']=teacher.right_hold;rows.append(row)
            row['tilt_error_deg']=tilt_error(row)
            teacher.finger_contacts=row['finger_contact_impulses']
            teacher.contact_points=row['finger_contact_points']
            unsafe=row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
            if row['phase'] in ('right_preshape','right_route') and row['contact_impulse'][1]>1e-5:
                teacher.failure_reason='unexpected_right_cube_contact_before_approach'
            if row['phase'] not in ('ready','preshape','lower','approach','squeeze','primary_lift') and (row['hand_geometry']['wrist_y_separation_m']<.03 or row['hand_geometry']['elbow_y_separation_m']<.04):teacher.failure_reason='arms_crossed'
            stopping=unsafe or teacher.accepted or teacher.failure_reason or row['drop']
            if step%120==0 or stopping:print('STEP',seed,step,row['phase'],'height',round(row['height_m'],3),'tilt_error',round(row['tilt_error_deg'],2),'self',row['self_contact_impulse'],flush=True)
            if a.record and (step%3==0 or stopping):
                im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,28),fill='black');d.text((6,8),f'ELBOW OUTWARD | seed {seed} | {teacher.name}',fill='white');d.rectangle((0,480,640,512),fill='black');d.text((6,490),f'ELBOW {row["measured_elbow"]["upper_arm_outward_deg"]:.1f} deg | ERROR {teacher.tilt_error_deg:.1f} deg | HOLD {teacher.hold/30:.2f}s',fill='white');frames.append(im)
            if stopping:break
        env._render=original_render
        if not teacher.accepted and not teacher.failure_reason and len(rows)>=a.max_steps:teacher.failure_reason='episode_step_budget'
        measured_audit=assess_handover(rows,env.cfg['joint_names'],regrasp=a.regrasp,full_episode=not bool(a.development_resume))
        elbow_audit=assess_elbow(rows,regrasp=a.regrasp)
        measured_audit['elbow_posture']=elbow_audit
        measured_audit['accepted']=measured_audit['accepted'] and elbow_audit['accepted']
        measured_audit['failures']+=elbow_audit['failures']
        accepted=bool(teacher.accepted and not unsafe and not teacher.failure_reason and measured_audit['accepted']);result={'seed':seed,'development_resume':bool(a.development_resume),'full_episode_verified':accepted and not bool(a.development_resume),'accepted':accepted,'tilt_only_accepted':accepted and not a.regrasp,'right_regrasp_validated':accepted and a.regrasp,'steps':len(rows),'failure':teacher.failure_reason or ('safety_gate' if unsafe else None),'self_contact_impulse_sum_Ns':sum(r['self_contact_impulse'] for r in rows),'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse'] for r in rows),'final_tilt_error_deg':rows[-1]['tilt_error_deg'],'tilt_hold_seconds':teacher.hold/30,'right_hold_seconds':teacher.right_hold/30,'final_self_pairs':rows[-1]['self_contact_pairs'],'measured_audit':measured_audit}
        results.append(result);write(f'trajectory_{seed}.json',rows);write('result.json',{'trials':results,'accepted':sum(r['accepted'] for r in results),'total':len(results),'elapsed_s':time.monotonic()-started});print('EPISODE',json.dumps(result),flush=True)
        if hasattr(teacher.planner,'last_path_validation'):write(f'path_audit_{seed}.json',teacher.planner.last_path_validation)
        if teacher.path is not None:write(f'tilt_plan_{seed}.json',teacher.path.tolist())
        if teacher.right_phases:write(f'right_plan_{seed}.json',{'geometry':teacher.regrasp_geometry,'phases':[(n,p.tolist()) for n,p in teacher.right_phases]})
        if a.record:
            for view,eye in [('front',[1.65,0.,1.45]),('side',[.75,1.60,1.45])]:
                env.overview.set_world_poses_from_view(torch.tensor([eye]),torch.tensor([[.40,.06,1.16]]));env._render()
                Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy()).save(run/f'final_{view}_{seed}.png')
        if frames:
            d=ImageDraw.Draw(frames[-1]);d.text((540,490),'PASS' if accepted else 'FAIL',fill='white')
            frames[0].save(run/f'rollout_{seed}.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
            from side_video import encode_mp4
            encode_mp4(run/f'rollout_{seed}.mp4',frames+[frames[-1]]*15)
except Exception as error:
    traceback.print_exc();write('error.json',{'type':type(error).__name__,'message':str(error),'seed':locals().get('seed'),'traceback':traceback.format_exc()});raise
finally:
    code=1 if sys.exc_info()[0] else 0;threading.Timer(15,lambda:os._exit(code)).start()
    if env:env.close()
    close_application(app)
