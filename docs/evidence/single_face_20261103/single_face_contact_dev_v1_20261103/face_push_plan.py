"""Host-side IK and full swept-path checks for an index-pad nZ face push.

Run with the ROS/Coal Python ABI. The cube remains purely contact driven.
"""
import json,sys,time
from pathlib import Path
import numpy as np,coal
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from safe_collision_plan import CollisionScene,search
from kinematics import RobotKinematics


def timed(points,seconds,names,k):
    curve=PchipInterpolator(np.linspace(0,1,len(points)),points,axis=0)
    velocity=np.array([.25 if n.startswith('Joint')else .8 for n in names]);acceleration=2*velocity
    for _ in range(8):
        u=np.linspace(0,1,int(np.ceil(seconds*30))+1);path=curve(10*u**3-15*u**4+6*u**5)
        v=np.max(np.abs(np.diff(path,axis=0)),axis=0)*30;a=np.max(np.abs(np.diff(path,n=2,axis=0)),axis=0)*900
        ratio=max(np.max(v/velocity),np.sqrt(np.max(a/acceleration)))
        if ratio<=1:return path
        seconds*=ratio*1.03
    raise ValueError('Cannot time push path')


class PushScene(CollisionScene):
    def __init__(self,request):
        super().__init__(request['initial'],{},request['cube_position'],request['cube_quaternion'],True)
        self.fixed=[b for b in self.fixed if b[0]!='cube_route_envelope'];self.C=R.from_quat(np.array(request['cube_quaternion'])[[1,2,3,0]]).as_matrix();self.center=np.asarray(request['cube_position'])-[0,0,1.15]
        self.theta=0.;self.contact=False;self.last_pair=None
    def valid(self,x):
        if not super().valid(x):return False
        frames=self.k.fk(self.q|dict(zip(self.names,x)))
        turn=self.C@R.from_euler('z',-self.theta,degrees=True).as_matrix()
        bodies=[]
        for name,size,center,rot in [('stationary_layers',(.057,.057,.038),self.center+self.C@np.array([0,0,.0095]),self.C),('moving_nZ_layer',(.057,.057,.019),self.center+self.C@np.array([0,0,-.019]),turn)]:
            t=np.eye(4);t[:3,3]=center;t[:3,:3]=rot;bodies.append((name,coal.Box(*size),t,np.zeros(3),np.linalg.norm(np.asarray(size)/2)))
        for n,m,local,c,r in self.moving:
            a=(n,m,frames[n]@local,c,r)
            for b in bodies:
                if self.contact and n.startswith('right_index_DIP')and b[0]=='moving_nZ_layer':continue
                if self.distance(a,b)<.002:
                    self.last_pair=(n,b[0],self.distance(a,b));return False
        # Check swept turning-layer corners against the stationary left robot.
        if self.contact:
            for b in self.fixed:
                if b[0].startswith('left_')or b[0].endswith('_L'):
                    if self.distance(b,bodies[1])<.001:
                        self.last_pair=(b[0],'moving_nZ_layer',self.distance(b,bodies[1]));return False
        return True



class FingerRoute:
    def __init__(self,scene,initial,names):
        self.scene=scene;self.initial=initial;self.names=names;self.k=scene.k
    @property
    def calls(self):return self.scene.calls
    @property
    def last_pair(self):return self.scene.last_pair
    def valid(self,x):
        self.scene.q=self.initial|dict(zip(self.names,x))
        return self.scene.valid(np.array([self.initial[n]for n in self.scene.names]))
    def edge(self,a,b,resolution=.025):
        return all(self.valid((1-u)*a+u*b)for u in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/resolution))+1))[1:])

def plan(request):
    if request.get('face','nZ')!='nZ':raise ValueError('First-stage planner supports the right-facing nZ layer only')
    k=RobotKinematics();names=request['joint_names'];initial=request['initial'];scene=PushScene(request)
    C=scene.C;center=scene.center;right=[f'Joint{i}_R'for i in range(1,8)];variables=right+[n for n in names if n.startswith('right_index_')]
    limits=np.array([k.limits[n]for n in variables]);limits[1,0]=max(limits[1,0],np.radians(-100));lo=limits[:,0]+.035;hi=limits[:,1]-.035
    anchor=json.loads((Path(__file__).parent/'opposition_grasp_v20.json').read_text())['contact_anchors']['index'];point=np.diag([1,-1,1])@np.array(anchor['point_local']);link='right_index_DIP_rubber_link'
    working=initial.copy()
    for n,v in [('right_middle_MPR_joint',-.15),('right_middle_MCP_joint',1.2),('right_middle_PIP_joint',1.4),('right_middle_DIP_joint',.6)]:working[n]=v
    for i,n in enumerate(variables):
        if n=='right_index_MPR_joint':lo[i]=-.03;hi[i]=.03;working[n]=0.
        if n.startswith('right_index_') and 'MPR' not in n:
            lo[i]=max(lo[i],.02);hi[i]=min(hi[i],.40);working[n]=.15
    first=np.clip([working[n]for n in variables],lo+1e-6,hi-1e-6);rng=np.random.default_rng(731)
    def fit(q,local,random_start=None):
        start=np.clip([q[n]for n in variables],lo+1e-6,hi-1e-6);target=center+C@local
        def residual(x):
            frames=k.fk(q|dict(zip(variables,x)));tip=frames[link];p=tip[:3,3]+tip[:3,:3]@point;w=frames['Link7_R'][:3,3];wc=C.T@(w-center)
            return np.r_[100*(p-target),50*max(wc[2]+.075,0),25*max(w[1]-(center[1]-.04),0),.02*(x-start)]
        solved=least_squares(residual,start if random_start is None else random_start,bounds=(lo,hi),max_nfev=150)
        q=q|dict(zip(variables,solved.x));t=k.fk_link(q,link)
        if np.linalg.norm(t[:3,3]+t[:3,:3]@point-target)>.002:raise ValueError('Index pad IK residual exceeds 2 mm')
        return q
    failures=[];target_degrees=float(request.get('target_deg',90.));began=time.monotonic()
    for corner in ([-.018,.020,-.027],[.020,-.018,-.027],[-.020,-.018,-.027],[.018,.020,-.027]):
        for attempt in range(4):
            if time.monotonic()-began>150:raise ValueError('Push planning budget exceeded; '+str(failures[-3:]))
            try:
                random_start=first if attempt==0 else np.clip(first+rng.normal(0,.65,len(first)),lo+1e-6,hi-1e-6)
                endpoint=fit(working,np.array(corner),random_start);scene.q=endpoint;scene.contact=True;scene.theta=0.
                if not scene.valid(np.array([endpoint[n]for n in right])):raise ValueError('Contact endpoint: '+str(scene.last_pair))
                pre=np.array(corner)+[0,0,-.065];q=endpoint;approach=[]
                # Solve backwards to preserve a contact-feasible wrist branch.
                for u in np.linspace(0,1,26):
                    q=fit(q,(1-u)*np.array(corner)+u*pre);approach.append(np.array([q[n]for n in names]))
                approach=approach[::-1];pregrasp=dict(zip(names,approach[0]));opened={n:pregrasp[n]for n in names if n.startswith('right_')}
                scene.q=initial|opened;scene.contact=False;scene.theta=0.
                arm_route=search(scene,np.array([initial[n]for n in right]),np.array([pregrasp[n]for n in right]),timeout=15.)
                start=np.array([initial[n]for n in names]);opened_q=np.array([(initial|opened)[n]for n in names])
                route=[np.array([(initial|opened|dict(zip(right,x)))[n]for n in names])for x in arm_route]
                finger_names=[n for n in names if n.startswith(('right_index_','right_middle_'))]
                finger_scene=FingerRoute(scene,initial,finger_names)
                finger_route=search(finger_scene,np.array([initial[n]for n in finger_names]),np.array([pregrasp[n]for n in finger_names]),timeout=15.)
                finger_points=[np.array([(initial|dict(zip(finger_names,x)))[n]for n in names])for x in finger_route]
                preshape=np.concatenate([timed([x,y],1.,names,k)for x,y in zip(finger_points,finger_points[1:])])
                phases=[('right_index_preshape',preshape),('right_index_route',timed(route,3.,names,k)),('right_index_approach',timed(approach,3.,names,k))]
                q=endpoint;turn=[]
                for angle in np.linspace(0,target_degrees,max(3,int(abs(target_degrees)/2)+1)):
                    local=R.from_euler('z',-angle,degrees=True).apply(corner);q=fit(q,local);turn.append(np.array([q[n]for n in names]))
                phases.append(('right_face_push',timed(turn,max(3.,abs(target_degrees)/12),names,k)))
                # Retract along the face normal, then hold without right-hand support.
                retreat=[];last=R.from_euler('z',-target_degrees,degrees=True).apply(corner)
                for u in np.linspace(0,1,20):
                    q=fit(q,last+[0,0,-.05*u]);retreat.append(np.array([q[n]for n in names]))
                phases.extend([('right_index_retreat',timed(retreat,3.,names,k)),('face_hold',np.repeat(retreat[-1][None],91,axis=0))])
                # Every nominal target plus fine joint interpolation is checked before any right motion.
                previous=start;checks=0
                for phase,path in phases:
                    for step,target in enumerate(path):
                        samples=max(1,int(np.ceil(np.max(np.abs(target-previous))/.01)))
                        for u in np.linspace(0,1,samples+1)[1:]:
                            row=(1-u)*previous+u*target;scene.q=dict(zip(names,row));scene.contact=phase in ('right_index_approach','right_face_push','right_index_retreat','face_hold')
                            scene.theta=(target_degrees*step/max(1,len(path)-1)if phase=='right_face_push'else target_degrees if phase in ('right_index_retreat','face_hold')else 0.)
                            # Time parameter is quintic: use measured IK pad azimuth to match each target.
                            if phase=='right_face_push':
                                t=k.fk_link(scene.q,link);p_local=C.T@(t[:3,3]+t[:3,:3]@point-center);scene.theta=float(np.degrees(np.arctan2(corner[1],corner[0])-np.arctan2(p_local[1],p_local[0]))%360)
                            checks+=1
                            if not scene.valid(np.array([scene.q[n]for n in right])):raise ValueError(f'{phase} step {step}: {scene.last_pair}')
                        previous=target
                return {'passed':True,'face':'nZ','target_deg':target_degrees,'contact_pad':link,'point_local':point.tolist(),'contact_start_cube':corner,'phases':[(name,path.tolist())for name,path in phases],'checks':checks,'failures_before_selection':failures,'elapsed_s':time.monotonic()-began,'physics_validated':False}
            except ValueError as e:failures.append({'corner':corner,'attempt':attempt,'failure':str(e)});print('REJECT',failures[-1],flush=True)
    return {'passed':False,'failures':failures,'physics_validated':False}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    try:result=plan(request)
    except ValueError as e:result={'passed':False,'failure':str(e),'physics_validated':False}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print('PLAN',result['passed'],flush=True);raise SystemExit(0 if result['passed']else 1)
