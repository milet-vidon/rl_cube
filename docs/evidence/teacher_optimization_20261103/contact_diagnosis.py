import json,sys,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from kinematics import RobotKinematics
p=Path('runs/teacher_joint_opposition_v1_20261103');rows=json.loads((p/'trajectory.json').read_text());row=rows[-1]
plan=json.loads((p/'opposition_plan.json').read_text());names=json.loads((p/'config.json').read_text())['joint_names'];k=RobotKinematics();q=dict(zip(names,row['q']));f=k.fk(q);C=R.from_quat(np.array(row['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
for finger,a in plan['contact_anchors'].items():
 t=f[a['link']];position=t[:3,3]+t[:3,:3]@a['point_local']+[0,0,1.15]
 print(finger,'anchor cube',C.T@(position-row['cube_position']))
print('audit',json.dumps(row.get('opposition_audit')))
