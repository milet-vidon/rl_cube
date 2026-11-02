"""Physics audit of experimental proximal-arm thumb-opposition teacher."""
from pathlib import Path
import argparse,json,hashlib,shutil,time,sys,os,threading
import numpy as np

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seeds',required=True)
p.add_argument('--grasp-config',required=True);p.add_argument('--record',action='store_true');p.add_argument('--closeup',action='store_true');a=p.parse_args()
run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj):(run/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
sources=['opposition_teacher.py','opposition_feedback.py','proximal_planner.py','opposition_geometry.py','opposition_experiment.py','finger_contact_audit.py','opposition_acceptance.py','self_contact_audit.py','side_grasp.py','config.json','kinematics.py','environment.py','joint_safety.py','acceptance.py','isaac_compat.py']
for name in sources:shutil.copy2(ROOT/name,run/name)
shutil.copy2(a.grasp_config,run/'grasp_config.json')
write('experiment.json',vars(a)|{'kind':'experimental_traditional_opposition_teacher','source_sha256':{n:sha(ROOT/n) for n in sources},'grasp_config_sha256':sha(a.grasp_config),'main_hand':'left','support_hand':'right','promoted_for_learning':False})
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import tensor,close_application,quaternion_from_native
app=AppLauncher(headless=True,enable_cameras=a.record,kit_args=platform_kit_args('--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env=None;started=time.monotonic()
try:
    from environment import CubePickupEnv
    from opposition_teacher import OppositionTeacher
    from self_contact_audit import SelfContactAudit
    from finger_contact_audit import FingerContactAudit,assess_opposition
    from acceptance import assess_pickup
    from PIL import Image,ImageDraw
    grasp_settings=json.loads(Path(a.grasp_config).read_text())
    env=CubePickupEnv(vision=a.record,preview=a.record,auto_reset=False,arm_damping=grasp_settings.get('arm_damping'));env.max_episode_length=1800;env.cfg['success']['hold_seconds']=3.
    env.cfg['arm_damping']=env.arm_damping
    audit=SelfContactAudit();fingers=FingerContactAudit();write('runtime_config.json',env.cfg);write('asset_contract.json',env.contract);results=[]
    for seed in map(int,a.seeds.split(',')):
        env.rng=np.random.default_rng(seed);env.reset();env.metrics={};env.pair_impulses={}
        if a.record and a.closeup:
            import torch
            env.overview.set_world_poses_from_view(torch.tensor([[1.35,-.85,1.55]]),torch.tensor([[.40,0.,1.12]]))
            env._render()
        teacher=OppositionTeacher(env,a.grasp_config);rows=[];frames=[]
        write(f'planned_{seed}.json',{n:q.tolist() for n,q in teacher.phases})
        for step in range(1800):
            audit.reset_step();fingers.reset_step();_,_,done,_=env.step(teacher.action())
            row=dict(step=step,phase=teacher.name,**env.metrics);row.update(audit.metrics());row.update(fingers.metrics());row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();rows.append(row)
            row['cube_quaternion_wxyz']=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).numpy().tolist()
            row['nominal_target']=teacher.nominal_target[0].tolist() if teacher.nominal_target is not None else None
            teacher.finger_contacts=row['finger_contact_impulses']
            unsafe=row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
            stopping=bool(done[0]) or unsafe or bool(teacher.failure_reason)
            if step%120==0 or stopping:print('STEP',seed,step,teacher.name,round(row['height_m'],4),'fingers',row['finger_contact_impulses']['left'],'self',row['self_contact_impulse'],flush=True)
            if a.record and (step%3==0 or stopping):
                im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im)
                d.rectangle((0,0,640,28),fill='black');d.text((8,8),f'EXPERIMENTAL THUMB OPPOSITION | seed {seed} | {teacher.name}',fill='white')
                d.rectangle((0,480,640,512),fill='black');d.text((8,490),f'LIFT {row["height_m"]*100:.1f} cm | HOLD {row["hold_steps"]/30:.2f} s',fill='white');frames.append(im)
            if stopping:break
        base=assess_pickup(rows,env.cfg);topology=assess_opposition(rows)
        if sum(r['self_contact_impulse'] for r in rows)>1e-6:base['accepted']=False;base['failure_reasons'].append('robot_self_contact')
        if not topology['passed']:base['accepted']=False;base['failure_reasons'].append('thumb_four_finger_topology')
        result={'seed':seed,'acceptance':base,'opposition':topology,'controller_failure':teacher.failure_reason,'max_height_m':max(r['height_m'] for r in rows),'steps':len(rows),'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse'] for r in rows),'self_contact_impulse_sum_Ns':sum(r['self_contact_impulse'] for r in rows),'contact_pairs':env.pair_impulses}
        results.append(result);write(f'trajectory_{seed}.json',rows);write(f'executed_plan_{seed}.json',{n:q.tolist() for n,q in teacher.phases});write('result.json',{'kind':'experimental_traditional_teacher','trials':results,'accepted':sum(r['acceptance']['accepted'] for r in results),'total':len(results),'elapsed_s':time.monotonic()-started,'promoted_for_learning':False});print('EPISODE',json.dumps(result),flush=True)
        if frames:
            d=ImageDraw.Draw(frames[-1]);d.text((480,490),'PASS' if base['accepted'] else 'FAIL',fill='white')
            frames[0].save(run/f'rollout_{seed}.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
            from side_video import encode_mp4
            encode_mp4(run/f'rollout_{seed}.mp4',frames+[frames[-1]]*15);env.save_rgb(run/f'chest_{seed}.png')
except Exception as error:
    import traceback
    traceback.print_exc()
    write('error.json',{'type':type(error).__name__,'message':str(error),'seed':locals().get('seed'),'traceback':traceback.format_exc()})
    raise
finally:
    code=1 if sys.exc_info()[0] else 0;threading.Timer(15,lambda:os._exit(code)).start()
    if env:env.close()
    close_application(app)
