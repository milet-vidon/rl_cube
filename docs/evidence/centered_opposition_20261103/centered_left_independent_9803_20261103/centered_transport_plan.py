"""Move the held cube toward the torso midline, preserving wrist bend and elbow-out."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from safe_collision_plan import CollisionScene
from face_push_plan import timed
import coal
from torso_collision import collision_objects

class CenterScene(CollisionScene):
 def __init__(self,q,position,quaternion,torso):
  super().__init__(q,{},position,quaternion,False);self.torso=collision_objects(torso);frames=self.k.fk(q);self.geometry=[]
  for n,m,t,c,r in self.fixed:
   if n=='cube_route_envelope':continue
   self.geometry.append((n,m,np.linalg.inv(frames[n])@t,c,r))
  self.geometry+=self.moving;self.left_moving=set()
  for name,*_ in self.geometry:
   cursor=name
   while cursor in self.k.by_child:
    item=self.k.by_child[cursor]
    if item[0] in [f'Joint{i}_L'for i in range(1,8)]:self.left_moving.add(name);break
    cursor=item[1]
  self.cube_local=np.linalg.inv(frames['left_palm_link'])@np.block([[R.from_quat(np.array(quaternion)[[1,2,3,0]]).as_matrix(),(np.array(position)-[0,0,1.15])[:,None]],[np.array([[0,0,0,1.]])]])
 def valid_full(self,q):
  self.calls+=1;f=self.k.fk(q);left=[];fixed=list(self.torso)
  for n,m,t,c,r in self.geometry:
   a=(n,m,f[n]@t,c,r)
   (left if n in self.left_moving else fixed).append(a)
  for a in left:
   n,m,t,c,r=a
   pts=np.array(m.points())@t[:3,:3].T+t[:3,3]
   if pts[:,2].min()<-.246:self.last_pair=(n,'table',float(pts[:,2].min()+.25));return False
   for b in fixed:
    if n=='Link1_L'and not b[0].startswith(('right_','Link')):continue
    if self.distance(a,b)<.003:return False
  for i,a in enumerate(left):
   for b in left[i+1:]:
    na,nb=a[0],b[0];aa=na.startswith('Link')and na.endswith('_L');bb=nb.startswith('Link')and nb.endswith('_L')
    if not(aa or bb):continue # Finger configuration unchanged during arm-only transport.
    if aa and bb:
     ia=int(na[4]);ib=int(nb[4])
     if abs(ia-ib)<=1 or{ia,ib}=={5,7}:continue
    elif int((na if aa else nb)[4])>=5:continue
    if self.distance(a,b)<.003:return False
  T=f['left_palm_link']@self.cube_local;cube=('held_cube',coal.Box(.057,.057,.057),T,np.zeros(3),.05)
  for a in left+fixed:
   if a[0].startswith('left_'):continue
   if self.distance(a,cube)<.003:return False
  return True

def plan(request):
 q=request['initial'];names=request['joint_names'];scene=CenterScene(q,request['cube_position'],request['cube_quaternion'],request['torso_colliders']);k=scene.k
 palm=k.fk_link(q,'left_palm_link');local=scene.cube_local;arm=[f'Joint{i}_L'for i in range(1,8)]
 limits=np.array([k.limits[n]for n in arm]);lo=limits[:,0]+.035;hi=limits[:,1]-.035;ref=np.array([q[n]for n in arm]);target=np.asarray(request.get('center_root',[.50,0.,.15]))
 def residual(x):
  f=k.fk(q|dict(zip(arm,x)));p=f['left_palm_link'];s,e,w=[f[n][:3,3]for n in ('Link2_L','Link4_L','Link7_L')]
  forearm=p[:3,:3].T@((w-e)/np.linalg.norm(w-e));angle=np.arctan2((e-s)[1],-(e-s)[2])
  return np.r_[150*((p@local)[:3,3]-target),12*max(p[1,0]+.75,0),.3*R.from_matrix(palm[:3,:3]@p[:3,:3].T).as_rotvec(),50*max(.18-(e-s)[1],0),50*max(.06-(e-w)[1],0),20*max(np.radians(8)-x[5],0),20*max(.45-forearm[2],0),10*max(np.radians(38)-angle,0),30*max(.24-f['Link5_L'][0,3],0),.01*(x-ref)]
 fit=least_squares(residual,np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=180)
 final=q|dict(zip(arm,fit.x));frames=k.fk(final);cube=frames['left_palm_link']@local
 if np.linalg.norm(cube[:3,3]-target)>.002:raise ValueError('Centered cube IK error exceeds 2 mm')
 e,w,s=[frames[n][:3,3]for n in ('Link4_L','Link7_L','Link2_L')];forearm=frames['left_palm_link'][:3,:3].T@((w-e)/np.linalg.norm(w-e));bend=np.degrees(np.arctan2(forearm[1],forearm[2]))
 if e[1]-s[1]<.16 or e[1]-w[1]<.05 or np.degrees(fit.x[5])<5 or bend>-5 or frames['left_palm_link'][1,0]>-.6 or forearm[2]<=.3 or np.degrees(np.arctan2((e-s)[1],-(e-s)[2]))<35:raise ValueError('Centered posture violates accepted elbow/wrist/palm direction')
 start=np.array([q[n]for n in names]);finish=np.array([final[n]for n in names]);path=timed([start,finish],8.,names,k);previous=start;checks=0
 for step,v in enumerate(path):
  for u in np.linspace(0,1,max(1,int(np.ceil(np.max(np.abs(v-previous))/.01)))+1)[1:]:
   state=dict(zip(names,(1-u)*previous+u*v));checks+=1
   if not scene.valid_full(state):return {'passed':False,'phase':'left_center_transport','step':step,'pair':scene.last_pair,'checks':checks,'physics_validated':False}
  previous=v
 return {'passed':True,'phases':[('left_center_transport',path.tolist()),('center_hold',np.repeat(finish[None],121,axis=0).tolist())],'checks':checks,'center_target_world_m':(cube[:3,3]+[0,0,1.15]).tolist(),'cube_target_quaternion_wxyz':R.from_matrix(cube[:3,:3]).as_quat()[[3,0,1,2]].tolist(),'palm_target_root':frames['left_palm_link'].tolist(),'initial_cube_world_m':request['cube_position'],'elbow_outward_m':float(e[1]-s[1]),'side_bend_deg':float(bend),'forearm_local_z':float(forearm[2]),'usd_torso_colliders':len(scene.torso),'finger_configuration_changed':False,'physics_validated':False}

if __name__=='__main__':
 try:result=plan(json.loads(Path(sys.argv[1]).read_text()))
 except Exception as e:result={'passed':False,'error':str(e),'physics_validated':False}
 Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print('CENTER_PLAN',result['passed'],result.get('checks'),flush=True);raise SystemExit(0 if result['passed']else 1)
