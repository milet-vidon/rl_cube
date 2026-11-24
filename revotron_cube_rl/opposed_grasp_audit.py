"""Five-finger opposition and target-layer contact checks from measured rows."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from tilt_audit import contact_topology
from wrist_audit import wrist_geometry
from elbow_audit import elbow_metrics
FINGERS=('thumb','index','middle','ring','little')

def right_contacts(points,position,quaternion,layer_angle=0.):
 C=R.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix();unturn=R.from_euler('z',layer_angle,degrees=True).as_matrix();impulses={f:0. for f in FINGERS};unexpected=[]
 for p in points:
  finger=next((f for f in FINGERS if '/right_'+f+'_DIP' in p['robot_body']),None)
  if finger is None or 'nZ' not in p['cubelet'].split('_'):
   unexpected.append(p);continue
  local=unturn@C.T@(np.asarray(p['position_world_m'])-position);sign=-1 if finger=='thumb'else 1
  valid=sign*local[1]>=.0215 and np.max(np.abs(local))<=.036 and local[2]<=-.0095
  if finger=='thumb':valid&=max(abs(local[0]),abs(local[2]))>=.010
  if valid:impulses[finger]+=p['impulse_Ns']
 return {'fingers':{f:impulses[f]>1e-5 for f in FINGERS},'impulses_Ns':impulses,'unexpected_contacts':unexpected,'all_five':all(v>1e-5 for v in impulses.values())}

def posture(row,names):
 left=np.asarray(row['measured_hand_frames']['left_palm_link']);right=np.asarray(row['measured_hand_frames']['right_palm_link']);delta=right[:3,3]-left[:3,3];direction=delta/max(np.linalg.norm(delta),1e-9);e=row['measured_elbow'];elbow=elbow_metrics(e['shoulder_root_m'],e['elbow_root_m'],e['wrist_root_m']);wrist=wrist_geometry(row,names)
 right_elbow=np.asarray(row['measured_hand_frames']['Link4_R'])[:3,3]
 valid=(abs(row['cube_position'][1])<=.03 and left[1,0]<=-.6 and left[:3,0]@direction>=.65 and -right[:3,0]@direction>=.65 and left[:3,0]@right[:3,0]<=-.3 and left[1,3]-right[1,3]>=.03 and elbow['elbow_root_m'][1]-right_elbow[1]>=.04 and elbow['outward_from_shoulder_m']>=.16 and elbow['upper_arm_outward_deg']>=35 and wrist['joint6_deg']>=5 and wrist['side_bend_deg']<=-5 and wrist['wrist_inside_elbow_m']>=.05 and wrist['forearm_direction_palm_frame'][2]>.3)
 return {'qualified':bool(valid),'wrist':wrist,'elbow':elbow,'palm_normal_dot':float(left[:3,0]@right[:3,0]),'left_facing_right_cos':float(left[:3,0]@direction),'right_facing_left_cos':float(-right[:3,0]@direction)}

def qualified(row,names,turned=False):
 geometry=posture(row,names);layers=row['layer_measurement'];right=right_contacts(row['right_cubelet_contacts'],row['cube_position'],row['cube_quaternion_wxyz'],layers['face_angle_deg']if turned else 0.)
 stable=row['height_m']>=.08 and not row['cube_table_contact']and not row['drop']and row['linear_speed']<.03 and row['angular_speed']<.20
 layers_ok=layers['geometry_qualified']and layers['maximum_selected_angular_velocity_rad_s']<.20 if turned else layers['assembled_for_grasp']
 safe=row['self_contact_impulse']<=1e-6 and row['robot_table_impulse']<=1e-6 and not(row['arm_soft_limit_hit']or row['actual_arm_soft_limit_violation']or any(row['hard_limit_hit']))
 return {'qualified':bool(stable and layers_ok and safe and geometry['qualified']and right['all_five']and not right['unexpected_contacts']and all(contact_topology(row,'left').values())),'posture':geometry,'right_contacts':right}
