"""Execute a prechecked robot-pad push; never write cube state or joint targets."""
from pathlib import Path
import json,os,subprocess,time
import numpy as np,torch
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw
import omni.physx
from pxr import PhysicsSchemaTools
from isaac_compat import tensor,quaternion_from_native
from layer_turn_metrics import measure_layers
from wrist_audit import assess_wrist,wrist_geometry
from elbow_audit import measured_elbow
from palm_handover_audit import measured_hand_frames

ROOT=Path(__file__).resolve().parent

class LayerContacts:
    def __init__(self):
        self.points=[];self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.callback)
    def callback(self,headers,data):
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(x))for x in (h.actor0,h.actor1)]
            robot=next((p for p in paths if p.startswith('/World/Robot/right_')),None);cube=next((p for p in paths if p.startswith('/World/Cube/')),None)
            if robot is None or cube is None:continue
            for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                impulse=float(np.linalg.norm(np.asarray(d.impulse)))
                if impulse>1e-7:self.points.append({'robot_body':robot,'cubelet':cube.rsplit('/',1)[-1],'impulse_Ns':impulse,'position_world_m':list(d.position)})


def execute(env,teacher,fingers,audit,rows,frames,run,args,stopped=lambda:False):
    names=env.cfg['joint_names'];initial=rows[-1];core0=np.asarray(initial['cube_position']);rot0=R.from_quat(np.asarray(initial['cube_quaternion_wxyz'])[[1,2,3,0]])
    request={'initial':dict(zip(names,initial['q'])),'joint_names':names,'cube_position':initial['cube_position'],'cube_quaternion':initial['cube_quaternion_wxyz'],'face':'nZ','target_deg':args.push_degrees}
    (run/'push_request.json').write_text(json.dumps(request,indent=2))
    host=os.environ.copy();host['PYTHONPATH']='/opt/ros/jazzy/lib/python3.12/site-packages';host['LD_LIBRARY_PATH']='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu'
    print('PUSH_PREFLIGHT_BEGIN',flush=True)
    with (run/'push_preflight.log').open('w')as log:
        try:process=subprocess.run(['/usr/bin/python3',str(ROOT/'face_push_plan.py'),str(run/'push_request.json'),str(run/'push_plan.json')],env=host,stdout=log,stderr=subprocess.STDOUT,timeout=190)
        except subprocess.TimeoutExpired:return {'executed':False,'qualified':False,'failure':'preflight_timeout'}
    plan=json.loads((run/'push_plan.json').read_text())
    if process.returncode or not plan['passed']:return {'executed':False,'qualified':False,'failure':'preflight_rejected','detail':plan}
    if stopped():return {'executed':False,'qualified':False,'failure':'execution_interrupted'}
    print('PUSH_PREFLIGHT_PASSED',plan['checks'],flush=True)
    fixed=env.targets[0].numpy().copy();right=[i for i,n in enumerate(names)if n.endswith('_R')or n.startswith('right_')];first=len(rows);contact=LayerContacts();lost=0;failure=None;push_contact_steps=0
    original_pose_writes=env.cube.pose_reset_count;started=time.monotonic()
    for phase,path in plan['phases']:
        for tick,planned in enumerate(path):
            fingers.reset_step();audit.reset_step();contact.points.clear()
            target=fixed.copy();target[right]=np.asarray(planned)[right];env.step((torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales)
            row=dict(step=len(rows),phase=phase,**env.metrics);row.update(fingers.metrics());row.update(audit.metrics())
            row['q']=tensor(env.robot.data.joint_pos)[0,env.ids].tolist();row['cube_quaternion_wxyz']=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).tolist()
            row.update(tilt_degrees=-90,right_contact_axis=0,right_thumb_sign=-1,measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env),tilt_reference_wxyz=teacher.reference.tolist())
            row['measured_wrist']=wrist_geometry(row,names);row['nominal_target']=target.tolist();row['cubelets']=env.cube.states();row['layer_measurement']=measure_layers(row['cubelets'],target_deg=args.push_degrees)
            row['pose_reset_count']=env.cube.pose_reset_count;row['right_cubelet_contacts']=list(contact.points);rows.append(row)
            with (run/'trajectory.jsonl').open('a')as journal:journal.write(json.dumps(row,allow_nan=False)+'\n')
            unsafe=row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])
            if stopped():failure='execution_interrupted'
            if unsafe:failure='robot_safety_gate'
            if row['drop']or row['cube_table_contact']:failure='cube_dropped_or_returned_to_table'
            primary=row['finger_contact_impulses']['left'];retained=primary['thumb']>1e-5 and sum(primary[f]>1e-5 for f in ('index','middle','ring','little'))>=3
            lost=0 if retained else lost+1
            if lost>=10:failure='left_grasp_lost_during_face_push'
            moved=np.linalg.norm(np.asarray(row['cube_position'])-core0);rot=R.from_quat(np.asarray(row['cube_quaternion_wxyz'])[[1,2,3,0]])
            if moved>.025 or np.degrees((rot*rot0.inv()).magnitude())>10:failure='held_cube_pose_envelope_exceeded'
            if phase in ('right_index_preshape','right_index_route')and contact.points:failure='premature_right_cube_contact'
            if any('/right_index_DIP'not in c['robot_body']or 'nZ'not in c['cubelet'].split('_')for c in contact.points):failure='unplanned_right_contact_body_or_layer'
            if phase=='right_face_push'and contact.points:push_contact_steps+=1
            if row['layer_measurement']['maximum_other_cubelet_rotation_deg']>15:failure='non_target_layers_rotating'
            if row['layer_measurement']['maximum_joint_origin_separation_m']>.002:failure='cube_joint_separation'
            if env.cube.pose_reset_count!=original_pose_writes:raise RuntimeError('Cube state write during face push')
            if tick%90==0 or failure:print('PUSH',row['step'],phase,'angle',round(row['layer_measurement']['face_angle_deg'],2),'error',round(row['layer_measurement']['maximum_face_error_deg'],2),'failure',failure,flush=True)
            if args.record and (row['step']%3==0 or failure):
                im=Image.fromarray(tensor(env.overview.data.output['rgb'])[0,:,:,:3].numpy());d=ImageDraw.Draw(im);d.rectangle((0,0,640,28),fill='black');d.text((6,8),f'nZ FACE PUSH | {phase} | seed {args.seed}',fill='white');d.rectangle((0,480,640,512),fill='black');d.text((6,490),f'TARGET {args.push_degrees:.0f} | ANGLE {row["layer_measurement"]["face_angle_deg"]:.1f} | ERROR {row["layer_measurement"]["maximum_face_error_deg"]:.1f}',fill='white');frames.append(im)
            if failure:break
        if failure:break
    checked=[r|{'phase':'tilt_hold'if r['phase']=='face_hold'and r['layer_measurement']['geometry_qualified']and r['layer_measurement']['maximum_selected_angular_velocity_rad_s']<.20 else 'not_face_hold'}for r in rows[first:]]
    hold=assess_wrist(checked,names)if checked else {'accepted':False}
    qualified=bool(not failure and hold['accepted']and push_contact_steps>=3)
    if not qualified and failure is None:failure='face_target_or_hold_gate_not_met'
    return {'executed':True,'qualified':qualified,'target_face':'nZ','target_degrees':args.push_degrees,'failure':failure,'post_push_hold':hold,'push_contact_steps':push_contact_steps,'control_steps':len(rows)-first,'elapsed_s':time.monotonic()-started,'final_layers':rows[-1]['layer_measurement'],'cube_pose_writes_during_push':env.cube.pose_reset_count-original_pose_writes}
