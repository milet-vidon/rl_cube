import json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from kinematics import RobotKinematics
p=Path('runs/teacher_joint_opposition_ring_v2_20261103');rows=json.loads((p/'trajectory.json').read_text());names=json.loads((p/'config.json').read_text())['joint_names'];k=RobotKinematics()
first=next(r for r in reversed(rows) if r['phase']=='opposition_hold');C=R.from_quat(np.array(first['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();palm=k.fk_link(dict(zip(names,first['q'])),'right_palm_link')
turn=[r for r in rows if r['phase']=='opposed_face_turn']
for row in turn[::15]:
 t=k.fk_link(dict(zip(names,row['nominal_target'])),'right_palm_link');angle=-R.from_matrix(C.T@t[:3,:3]@palm[:3,:3].T@C).as_rotvec()[2]*180/np.pi
 print(row['step'],'command',round(angle,2),'face',round(row['layer_measurement']['face_angle_deg'],2),'fingers',row['opposition_audit']['right_contacts']['fingers'])
for phase in ('opposition_hold','opposed_face_turn','opposed_face_hold'):
 row=next(r for r in reversed(rows) if r['phase']==phase);C=R.from_quat(np.array(row['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();report={}
 for f in ('thumb','index','middle','ring','little'):
  points=[p for p in row['finger_contact_points'] if p['side']=='left' and p['finger']==f]
  zs=[float((C.T@(np.array(p['position_world_m'])-row['cube_position']))[2]) for p in points]
  report[f]={'z_range_m':[min(zs),max(zs)] if zs else [],'target_layer_impulse':sum(p['impulse_Ns'] for p,z in zip(points,zs) if z<-.0095)}
 print('LEFT_TARGET_LAYER',phase,report)
