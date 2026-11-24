from pathlib import Path
import json,sys,numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
sys.path.insert(0,'/home/brainco/rl_cube/github_project/revotron_cube_rl')
from kinematics import RobotKinematics
root=Path('/tmp/j56_center_20261103');req=json.loads((root/'request.json').read_text());k=RobotKinematics();q=req['initial'];arm=[f'Joint{i}_L'for i in range(1,8)];ref=np.array([q[n]for n in arm]);limits=np.array([k.limits[n]for n in arm]);lo=limits[:,0]+.035;hi=limits[:,1]-.035;f=k.fk(q);palm=f['left_palm_link'];cube=np.eye(4);cube[:3,:3]=R.from_quat(np.array(req['cube_quaternion'])[[1,2,3,0]]).as_matrix();cube[:3,3]=np.array(req['cube_position'])-[0,0,1.15];local=np.linalg.inv(palm)@cube;target=np.array([.5,0,.15]);results=[]
def geometry(x):
 frames=k.fk(q|dict(zip(arm,x)));p=frames['left_palm_link'];s,e,w=[frames[n][:3,3]for n in ('Link2_L','Link4_L','Link7_L')];v=p[:3,:3].T@((w-e)/np.linalg.norm(w-e));return frames,p,s,e,w,v
# Establish whether the two wrist rotations alone can reach the existing target.
indices=[4,5]
def only(v):
 x=ref.copy();x[indices]=v;return 150*((geometry(x)[1]@local)[:3,3]-target)
fit=least_squares(only,ref[indices],bounds=(lo[indices],hi[indices]),max_nfev=200);x=ref.copy();x[indices]=fit.x;two={'q':dict(zip(arm,x.tolist())),'position_root_m':(geometry(x)[1]@local)[:3,3].tolist(),'error_m':float(np.linalg.norm(only(fit.x)/150)),'fixed_wrist_root_m':f['Link7_L'][:3,3].tolist(),'wrist_to_cube_distance_m':float(np.linalg.norm(cube[:3,3]-f['Link7_L'][:3,3]))};print('TWO_ONLY',two,flush=True)
for j5 in [-65,-50,-35,-20,0,20]:
 for j6 in [20,30,40]:
  wanted=np.radians([j5,j6,35.])
  def fun(x):
   frames,p,s,e,w,v=geometry(x)
   return np.r_[150*((p@local)[:3,3]-target),15*max(p[1,0]+.75,0),.25*R.from_matrix(palm[:3,:3]@p[:3,:3].T).as_rotvec(),55*max(.18-(e-s)[1],0),55*max(.06-(e-w)[1],0),25*max(.45-v[2],0),10*max(np.radians(38)-np.arctan2((e-s)[1],-(e-s)[2]),0),30*max(.24-frames['Link5_L'][0,3],0),np.array([1.5,3.,2.])*(x[4:]-wanted),.10*(x[:4]-ref[:4])]
  fit=least_squares(fun,np.clip(ref,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=240);frames,p,s,e,w,v=geometry(fit.x);pose=p@local;error=float(np.linalg.norm(pose[:3,3]-target));bend=float(np.degrees(np.arctan2(v[1],v[2])));angle=float(np.degrees(np.arctan2((e-s)[1],-(e-s)[2])));good=error<=.002 and e[1]-s[1]>=.16 and e[1]-w[1]>=.05 and np.degrees(fit.x[5])>=5 and bend<=-5 and p[1,0]<=-.6 and v[2]>.3 and angle>=35
  row={'requested_j5_j6_deg':[j5,j6],'position_error_m':error,'joint_deg':np.degrees(fit.x).tolist(),'joint_delta_deg':np.degrees(fit.x-ref).tolist(),'elbow_out_m':float(e[1]-s[1]),'upper_arm_deg':angle,'forearm_local_z':float(v[2]),'palm_y':float(p[1,0]),'side_bend_deg':bend,'wrist_inside_m':float(e[1]-w[1]),'pose_qualified':bool(good),'q':q|dict(zip(arm,fit.x.tolist())),'cube_pose_root':pose.tolist()};results.append(row);print('POSE',j5,j6,good,round(error,4),np.round(row['joint_deg'],1).tolist(),round(float(v[2]),3),flush=True)
(root/'candidates.json').write_text(json.dumps({'request':req,'two_joints_only':two,'candidates':results},indent=2))
