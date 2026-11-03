"""Mirror the accepted thumb/four-finger shape onto the right-facing layer."""
from pathlib import Path
import json,sys,time,itertools
import numpy as np,coal
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from torso_collision import collision_objects
from opposed_grasp_audit import posture
from safe_collision_plan import CollisionScene,search
from face_push_plan import timed,FingerRoute
FINGERS=('thumb','index','middle','ring','little')

class OppositionScene(CollisionScene):
 def __init__(self,request):
  super().__init__(request['initial'],{},request['cube_position'],request['cube_quaternion'],True)
  self.fixed=[b for b in self.fixed if b[0]!='cube_route_envelope']+collision_objects(request['torso_colliders']);self.C=R.from_quat(np.array(request['cube_quaternion'])[[1,2,3,0]]).as_matrix();self.center=np.array(request['cube_position'])-[0,0,1.15];self.contact=False;self.theta=0.
 def valid(self,x):
  if not super().valid(x):return False
  f=self.k.fk(self.q|dict(zip(self.names,x)));turn=self.C@R.from_euler('z',-self.theta,degrees=True).as_matrix();bodies=[]
  for name,size,pos,rot in [('stationary_layers',(.057,.057,.038),self.center+self.C@[0,0,.0095],self.C),('target_nZ_layer',(.057,.057,.019),self.center+self.C@[0,0,-.019],turn)]:
   t=np.eye(4);t[:3,3]=pos;t[:3,:3]=rot;bodies.append((name,coal.Box(*size),t,np.zeros(3),np.linalg.norm(np.array(size)/2)))
  for n,m,t,c,r in self.moving:
   a=(n,m,f[n]@t,c,r)
   for b in bodies:
    allowed=self.contact and b[0]=='target_nZ_layer'and any(n.startswith('right_'+finger+'_DIP')for finger in FINGERS)
    distance=self.distance(a,b)
    if distance<(-.002 if allowed else .002):self.last_pair=(n,b[0],distance);return False
  if self.theta:
   for a in self.fixed:
    if a[0].startswith('left_')or a[0].endswith('_L'):
     distance=self.distance(a,bodies[1])
     if distance<.001:self.last_pair=(a[0],'rotating_layer',distance);return False
  return True

def closure_path(scene,opened,end,names,k):
 start=np.array([opened[n]for n in names]);finish=np.array([end[n]for n in names]);arm=scene.names;scene.contact=True
 schedules=[{}, {'middle':.2},{'ring':.2},{'middle':.2,'ring':-.2},{'ring':.2,'middle':-.2},{'little':.2,'ring':.1},{'index':-.2,'middle':-.1}]
 for delays in schedules:
  points=[]
  for u in np.linspace(0,1,17):
   v=start.copy()
   for i,n in enumerate(names):
    finger=next((f for f in FINGERS if n.startswith('right_'+f+'_')),None)
    shift=delays.get(finger,0.);fraction=np.clip((u-max(shift,0.))/(1-abs(shift)),0,1);fraction=fraction*fraction*(3-2*fraction);v[i]+=fraction*(finish[i]-start[i])
   points.append(v)
  path=timed(points,5.,names,k);valid=True
  for point in path:
   scene.q=dict(zip(names,point))
   if not scene.valid(np.array([scene.q[n]for n in arm])):valid=False;break
  if valid:print('CLOSURE_SCHEDULE',delays,flush=True);return path,delays
  print('CLOSURE_REJECT',delays,scene.last_pair,flush=True)
 variables=[n for n in names if n.startswith('right_')and 'MPR' in n];limits=np.array([k.limits[n]for n in variables]);points=[start]
 for u in np.linspace(0,1,17)[1:-1]:
  base=dict(zip(names,(1-u)*start+u*finish));ref=np.array([base[n]for n in variables]);low=np.maximum(limits[:,0]+.005,ref-.035);high=np.minimum(limits[:,1]-.005,ref+.035)
  def residual(v):
   state=base|dict(zip(variables,v));frames=k.fk(state);groups={f:[]for f in FINGERS}
   for n,m,t,c,r in scene.moving:
    finger=next((f for f in FINGERS if n.startswith('right_'+f+'_')),None)
    if finger:groups[finger].append((n,m,frames[n]@t,c,r))
   distances=[min(scene.distance(a,b)for a in groups[f]for b in groups[g])for i,f in enumerate(FINGERS)for g in FINGERS[i+1:]]
   return np.r_[500*np.maximum(.0035-np.array(distances),0),.05*(v-ref)]
  fit=least_squares(residual,np.clip(ref,low+1e-7,high-1e-7),bounds=(low,high),max_nfev=60);state=base|dict(zip(variables,fit.x));scene.q=state
  if not scene.valid(np.array([state[n]for n in arm])):raise ValueError('Closure waypoint repair failed: '+str(scene.last_pair))
  points.append(np.array([state[n]for n in names]))
 points.append(finish);path=timed(points,5.,names,k)
 for point in path:
  scene.q=dict(zip(names,point))
  if not scene.valid(np.array([scene.q[n]for n in arm])):raise ValueError('Repaired closure interpolation failed: '+str(scene.last_pair))
 print('CLOSURE_MPR_REPAIR_PASSED',flush=True);return path,{'method':'collision_aware_MPR_waypoints','maximum_adjustment_rad':.035}


def plan(request):
 started=time.monotonic();initial=request['initial'];names=request['joint_names'];scene=OppositionScene(request);k=scene.k
 cube=np.eye(4);cube[:3,:3]=scene.C;cube[:3,3]=scene.center;mirror=np.diag([1.,-1.,1.,1.]);flip=np.diag([1.,1.,-1.,1.]);roll=np.eye(4);roll[:3,:3]=R.from_euler('z',90,degrees=True).as_matrix()
 lp=np.linalg.inv(cube)@k.fk_link(initial,'left_palm_link');relative=roll@flip@lp@mirror
 arm=[f'Joint{i}_R'for i in range(1,8)];limits=np.array([k.limits[n]for n in arm]);limits[1,0]=max(limits[1,0],np.radians(-100));lo=limits[:,0]+.035;hi=limits[:,1]-.035
 closed={n.replace('left_','right_'):v for n,v in initial.items()if n.startswith('left_')};seed=initial|closed
 if 'right_ik_reference'in request:seed.update(dict(zip(arm,request['right_ik_reference'])))
 calibration={};grasp=json.loads(Path(__file__).with_name('opposition_grasp_v20.json').read_text())
 for finger in FINGERS:
  anchor=grasp['contact_anchors'][finger];link=anchor['link'];points=[p for p in request['left_contacts']if p['side']=='left'and p['finger']==finger]
  if not points:raise ValueError('Missing measured left contact for '+finger)
  point=max(points,key=lambda p:p['impulse_Ns']);world=np.array(point['position_world_m'])-[0,0,1.15];f=k.fk_link(initial,link);local=mirror[:3,:3]@f[:3,:3].T@(world-f[:3,3]);calibration[finger]={'link':link.replace('left_','right_'),'point_local':local.tolist()}
 def fit_arm(q,target):
  ref=np.clip([q[n]for n in arm],lo+1e-6,hi-1e-6)
  def residual(v):
   p=k.fk_link(q|dict(zip(arm,v)),'right_palm_link');return np.r_[100*(p[:3,3]-target[:3,3]),10*R.from_matrix(target[:3,:3]@p[:3,:3].T).as_rotvec(),.008*np.sqrt([1,1,1,1,4,4,4])*(v-ref)]
  fit=least_squares(residual,ref,bounds=(lo,hi),max_nfev=160);q=q|dict(zip(arm,fit.x));p=k.fk_link(q,'right_palm_link')
  if np.linalg.norm(p[:3,3]-target[:3,3])>.002 or np.degrees(R.from_matrix(target[:3,:3]@p[:3,:3].T).magnitude())>1:raise ValueError('Right opposition palm IK failed')
  return q
 def fit_finger(q,finger,goal,opening=False):
  ns=[n for n in names if n.startswith('right_'+finger+'_')];ref=np.array([q[n]for n in ns]);limits=np.array([k.limits[n]for n in ns]);low=np.maximum(limits[:,0]+.02,ref-(.9 if opening else .65));high=np.minimum(limits[:,1]-.02,ref+(.9 if opening else .65))
  for i,n in enumerate(ns):
   if 'MPR' in n:low[i]=max(low[i],ref[i]-.03);high[i]=min(high[i],ref[i]+.03)
  a=calibration[finger]
  def position(v):
   t=k.fk_link(q|dict(zip(ns,v)),a['link']);return scene.C.T@(t[:3,3]+t[:3,:3]@a['point_local']-scene.center)
  weights=np.array([20,150,20])if opening else np.array([100,150,100])
  def clearance(v):
   state=q|dict(zip(ns,v));frames=k.fk(state);shapes=[(n,m,frames[n]@t,c,r)for n,m,t,c,r in scene.moving if n.startswith('right_')];active=[a for a in shapes if a[0].startswith('right_'+finger+'_')];others=[a for a in shapes if any(a[0].startswith('right_'+f+'_')for f in FINGERS if f!=finger)]
   return [600*max(.0035-min((scene.distance(a,b)for a in active for b in others if b[0].startswith('right_'+f+'_')),default=1.),0.)for f in FINGERS if f!=finger]
  def residual(v):
   pos=position(v)
   if opening:return np.r_[weights*(pos-goal),.04*(v-ref),clearance(v)]
   return np.r_[150*(pos[1]-goal[1]),150*max(pos[2]+.017,0),150*max(-.027-pos[2],0),100*max(abs(pos[0])-.027,0),3*(pos[[0,2]]-goal[[0,2]]),.04*(v-ref),clearance(v)]
  fit=least_squares(residual,np.clip(ref,low+1e-6,high-1e-6),bounds=(low,high),max_nfev=150)
  err=np.linalg.norm(position(fit.x)-goal)
  pos=position(fit.x)
  if (not opening and (abs(pos[1]-goal[1])>.0015 or pos[2]>-.015 or pos[2]<-.029 or abs(pos[0])>.029))or(opening and abs(pos[1])<.039):raise ValueError('Finger goal unreachable: '+finger+' '+str(err))
  return q|dict(zip(ns,fit.x))
 failures=[]
 for offset, xs in itertools.product(([0.,.003,-.004],[0.,0.,-.004],[0.,.006,-.004],[0.,.003,-.008],[0.,.003,0.]), ((.025,.009,-.009,-.025),(.027,.009,-.009,-.027),(.028,.008,-.012,-.028))):
  try:
   local=relative.copy();local[:3,3]+=offset;target=cube@local;q=fit_arm(seed,target)
   goals={'thumb':[.013,-.0282,-.023],'index':[.028,.0282,-.026],'middle':[.005,.0282,-.0265],'ring':[-.021,.0282,-.0265],'little':[-.028,.0282,-.0145]}
   for i,finger in enumerate(FINGERS[1:]):goals[finger][0]=xs[i]
   for finger in FINGERS:q=fit_finger(q,finger,np.array(goals[finger]))
   end=q.copy();scene.q=end;scene.contact=True
   if not scene.valid(np.array([end[n]for n in arm])):raise ValueError('Closed grasp collision: '+str(scene.last_pair))
   measured=k.fk(end);check={'q':[end[n]for n in names],'cube_position':request['cube_position'],'measured_hand_frames':{n:t.tolist()for n,t in measured.items()},'measured_elbow':{'shoulder_root_m':measured['Link2_L'][:3,3].tolist(),'elbow_root_m':measured['Link4_L'][:3,3].tolist(),'wrist_root_m':measured['Link7_L'][:3,3].tolist()}}
   geometry=posture(check,names)
   if not geometry['qualified']:raise ValueError('Mirrored palm posture rejected: '+json.dumps(geometry))
   print('CLOSED_CANDIDATE',offset,flush=True)
   opened=end.copy()
   for finger in FINGERS:
    goal=np.array(goals[finger]);goal[1]=-.046 if finger=='thumb'else .046;opened=fit_finger(opened,finger,goal,True)
   scene.q=opened;scene.contact=False
   if not scene.valid(np.array([opened[n]for n in arm])):raise ValueError('Open grasp collision: '+str(scene.last_pair))
   closure,closure_schedule=closure_path(scene,opened,end,names,k)
   approach=[];q=opened
   for u in np.linspace(0,1,25):
    t=target.copy();t[:3,3]+=scene.C@np.array([0,0,-.12*u]);q=fit_arm(q,t);approach.append(np.array([q[n]for n in names]))
   approach=approach[::-1];pre=dict(zip(names,approach[0]));open_fingers={n:pre[n]for n in names if n.startswith('right_')}
   scene.q=initial|open_fingers;scene.contact=False
   route=search(scene,np.array([initial[n]for n in arm]),np.array([pre[n]for n in arm]),timeout=25.)
   finger_names=[n for n in names if n.startswith('right_')];fs=FingerRoute(scene,initial,finger_names);fp=search(fs,np.array([initial[n]for n in finger_names]),np.array([pre[n]for n in finger_names]),timeout=30.)
   finger_points=[np.array([(initial|dict(zip(finger_names,x)))[n]for n in names])for x in fp];preshape=np.concatenate([timed([a,b],1.5,names,k)for a,b in zip(finger_points,finger_points[1:])])
   route_points=[np.array([(initial|open_fingers|dict(zip(arm,x)))[n]for n in names])for x in route]
   phases=[('right_preshape',preshape),('right_route',timed(route_points,4.,names,k)),('right_approach',timed(approach,4.,names,k)),('right_close',closure),('opposition_hold',np.repeat(np.array([[end[n]for n in names]]),121,axis=0))]
   q=end;turn=[]
   for angle in np.linspace(0,request.get('target_deg',15.),16):
    t=np.eye(4);t[:3,:3]=R.from_euler('z',-angle,degrees=True).as_matrix();q=fit_arm(q,cube@t@local);turn.append(np.array([q[n]for n in names]))
   phases.append(('opposed_face_turn',timed(turn,5.,names,k)));phases.append(('opposed_face_hold',np.repeat(turn[-1][None],121,axis=0)))
   previous=np.array([initial[n]for n in names]);checks=0
   for phase,path in phases:
    for step,point in enumerate(path):
     for u in np.linspace(0,1,max(1,int(np.ceil(np.max(np.abs(point-previous))/.01)))+1)[1:]:
      state=(1-u)*previous+u*point;scene.q=dict(zip(names,state));scene.contact=phase in ('right_close','opposition_hold','opposed_face_turn','opposed_face_hold');scene.theta=0.
      if phase in ('opposed_face_turn','opposed_face_hold'):
       p=k.fk_link(scene.q,'right_palm_link');delta=scene.C.T@p[:3,:3]@target[:3,:3].T@scene.C;scene.theta=float(-R.from_matrix(delta).as_rotvec()[2]*180/np.pi)
      checks+=1
      if not scene.valid(np.array([scene.q[n]for n in arm])):raise ValueError(f'{phase} step {step}: {scene.last_pair}')
     previous=point
   return {'passed':True,'phases':[(p,x.tolist())for p,x in phases],'checks':checks,'contact_anchors':calibration,'right_contact_axis':1,'right_thumb_sign':-1,'hand_style':'thumb_opposed_to_four_fingers','target_face':'nZ','target_deg':request.get('target_deg',15.),'offset_cube':offset,'closure_schedule':closure_schedule,'failures_before_selection':failures,'elapsed_s':time.monotonic()-started,'physics_validated':False}
  except ValueError as e:failures.append({'offset':offset,'finger_x':xs,'failure':str(e)});print('REJECT',failures[-1],flush=True)
 return {'passed':False,'failures':failures,'elapsed_s':time.monotonic()-started,'physics_validated':False}

if __name__=='__main__':
 try:result=plan(json.loads(Path(sys.argv[1]).read_text()))
 except Exception as e:result={'passed':False,'error':str(e),'physics_validated':False}
 Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print('OPPOSITION_PLAN',result['passed'],flush=True);raise SystemExit(0 if result['passed']else 1)
