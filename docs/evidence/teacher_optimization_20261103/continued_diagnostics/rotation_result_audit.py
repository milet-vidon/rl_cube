from pathlib import Path
import json,time,numpy as np
from scipy.spatial.transform import Rotation as R
from kinematics import RobotKinematics
ROOT=Path('/home/brainco/rl_cube/github_project/revotron_cube_rl')
name='teacher_joint_right_traction_v4_20261103';run=ROOT/'runs'/name
while not (run/'result.json').exists() and Path('/proc/330284').exists():time.sleep(5)
if not (run/'result.json').exists():raise RuntimeError('Rotation process ended without result.json; inspect error/log')
result=json.loads((run/'result.json').read_text());rows=json.loads((run/'trajectory.json').read_text());names=json.loads((run/'config.json').read_text())['joint_names'];report={'run':name,'result_summary':{k:result.get(k) for k in ('failure','steps','dynamic_grasp_passed','center_transport_passed','face_turn_validated','cube_pose_writes_during_control')},'push':result.get('push'),'phases':{}}
for phase in ('opposition_hold','opposed_face_turn','opposed_face_hold'):
 seq=[r for r in rows if r['phase']==phase]
 if not seq:continue
 data={'count':len(seq),'right_all_five_fraction':sum(r['opposition_audit']['right_contacts']['all_five'] for r in seq)/len(seq),'qualified_fraction':sum(r['opposition_audit']['qualified'] for r in seq)/len(seq),'last_face_angle_deg':seq[-1]['layer_measurement']['face_angle_deg'],'maximum_selected_angular_velocity_rad_s':max(r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for r in seq),'final_right_fingers':seq[-1]['opposition_audit']['right_contacts']['fingers']}
 report['phases'][phase]=data
holds=[r for r in rows if r['phase']=='opposition_hold']
if holds:
 k=RobotKinematics();ref=holds[-1];f0=k.fk_link(dict(zip(names,ref['q'])),'right_palm_link');C=R.from_quat(np.array(ref['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();last=rows[-1]
 for label,field in (('actual','q'),('command','nominal_target')):
  frame=k.fk_link(dict(zip(names,last[field])),'right_palm_link');report[label+'_palm_rotation_cube_deg']=(R.from_matrix(C.T@frame[:3,:3]@f0[:3,:3].T@C).as_rotvec()*180/np.pi).tolist()
out=ROOT.parent/'work/teacher_optimization_20261103/right_traction_v4_analysis.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
