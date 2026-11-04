import json,sys,numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/home/brainco/rl_cube/github_project/revotron_cube_rl')
from kinematics import RobotKinematics
from safe_collision_plan import CollisionScene
from torso_collision import collision_objects
p=Path('/tmp/j56_center_20261103');request=json.loads((p/'j56_request.json').read_text());plan=json.loads((p/'j56_plan.json').read_text());names=request['joint_names'];q=dict(zip(names,plan['phases'][-1][1][-1]));k=RobotKinematics();cube=np.eye(4);cube[:3,:3]=R.from_quat(np.array(plan['cube_target_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();cube[:3,3]=np.array(plan['center_target_world_m'])-[0,0,1.15];left=k.fk_link(q,'left_palm_link');local=np.linalg.inv(cube)@left;arm=[f'Joint{i}_R'for i in range(1,8)];limits=np.array([k.limits[n]for n in arm]);limits[1,0]=max(limits[1,0],np.radians(-100));lo=limits[:,0]+.035;hi=limits[:,1]-.035;rng=np.random.default_rng(1945);results=[]
q|={n.replace('left_','right_'):v for n,v in q.items()if n.startswith('left_')};scene=CollisionScene(q,{},plan['center_target_world_m'],plan['cube_target_quaternion_wxyz'],True);scene.fixed=[a for a in scene.fixed if a[0]!='cube_route_envelope']+collision_objects(request['torso_colliders'])
for roll in [0,30,-30,90,-90]:
 rz=np.eye(4);rz[:3,:3]=R.from_euler('z',roll,degrees=True).as_matrix();target=cube@rz@np.diag([1,1,-1,1])@local@np.diag([1,-1,1,1]);direction=target[:3,3]-left[:3,3];direction/=np.linalg.norm(direction);palms=bool(left[:3,0]@direction>=.65 and -target[:3,0]@direction>=.65)
 for attempt in range(5):
  ref=np.clip([q[n]for n in arm]if attempt==0 else rng.uniform(lo,hi),lo+1e-7,hi-1e-7)
  def residual(x):
   palm=k.fk_link(q|dict(zip(arm,x)),'right_palm_link');return np.r_[100*(palm[:3,3]-target[:3,3]),10*R.from_matrix(target[:3,:3]@palm[:3,:3].T).as_rotvec(),.005*(x-ref)]
  fit=least_squares(residual,ref,bounds=(lo,hi),max_nfev=160);state=q|dict(zip(arm,fit.x));palm=k.fk_link(state,'right_palm_link');error=np.linalg.norm(palm[:3,3]-target[:3,3]);angle=np.degrees(R.from_matrix(target[:3,:3]@palm[:3,:3].T).magnitude())
  if error>.002 or angle>1:continue
  scene.q=state;scene.last_pair=None;clear=scene.valid(fit.x);item={'roll_deg':roll,'attempt':attempt,'IK_error_m':float(error),'palm_direction_passed':palms,'robot_collision_clear':clear,'pair':scene.last_pair,'q':state};results.append(item);print({key:value for key,value in item.items()if key!='q'},flush=True)
  if clear:break
(p/'right_feasibility.json').write_text(json.dumps({'scope':'offline endpoint robot/robot feasibility only; no target-layer contact, route, or physics validation','candidates':results},indent=2))
