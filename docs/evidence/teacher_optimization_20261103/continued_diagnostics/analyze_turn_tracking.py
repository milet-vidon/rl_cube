from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
from kinematics import RobotKinematics
p=Path('runs/teacher_joint_opposition_ring_v2_20261103');rows=json.loads((p/'trajectory.json').read_text());cfg=json.loads((p/'config.json').read_text());names=cfg['joint_names'];k=RobotKinematics()
reference=next(r for r in reversed(rows) if r['phase']=='opposition_hold');f0=k.fk_link(dict(zip(names,reference['q'])),'right_palm_link');C=R.from_quat(np.array(reference['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
summaries=[]
for phase in ('opposition_hold','opposed_face_turn','opposed_face_hold'):
 seq=[r for r in rows if r['phase']==phase];last=seq[-1];actual=k.fk_link(dict(zip(names,last['q'])),'right_palm_link');command=k.fk_link(dict(zip(names,last['nominal_target'])),'right_palm_link')
 report={'phase':phase,'count':len(seq),'actual_palm_rotation_cube_deg':(R.from_matrix(C.T@actual[:3,:3]@f0[:3,:3].T@C).as_rotvec()*180/np.pi).tolist(),
 'command_palm_rotation_cube_deg':(R.from_matrix(C.T@command[:3,:3]@f0[:3,:3].T@C).as_rotvec()*180/np.pi).tolist(),
 'right_arm_target_errors_rad':{n:float(last['nominal_target'][names.index(n)]-last['q'][names.index(n)]) for n in names if n.endswith('_R')},
 'all_right_fingers_fraction':sum(r.get('opposition_audit',{}).get('right_contacts',{}).get('all_five',False) for r in seq)/len(seq),
 'final_audit':last.get('opposition_audit'),'face_angle_deg':last['layer_measurement']['face_angle_deg'],
 'max_layer_omega':max(r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for r in seq)}
 summaries.append(report)
print('arm stiffness',cfg.get('arm_stiffness'));print(json.dumps(summaries,indent=2))
Path('../work/teacher_optimization_20261103/turn_tracking_audit.json').write_text(json.dumps(summaries,indent=2))
