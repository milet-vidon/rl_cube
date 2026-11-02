"""Offline, deterministic joint-space route search with convex mesh checks."""
from pathlib import Path
import sys,json,time,xml.etree.ElementTree as ET
import numpy as np,coal
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kinematics import RobotKinematics

class CollisionScene:
 def __init__(self,initial,opened,cube_position,cube_quaternion,keep_right_side=False):
  self.keep_right_side=keep_right_side;self.cube_y=float(cube_position[1])
  self.k=RobotKinematics();self.q=initial|opened;self.names=[f'Joint{i}_R' for i in range(1,8)];self.request=coal.DistanceRequest();self.request.enable_signed_distance=True
  root=Path(__file__).resolve().parent.parent/'revotron_revo3/source/revotron_system';tree=ET.parse(root/'urdf/revotron_bimanual_revo3_56dof.urdf').getroot();self.moving=[];self.fixed=[];frames=self.k.fk(self.q)
  for link in tree.findall('link'):
   name=link.get('name');moving=False;cursor=name
   while cursor in self.k.by_child:
    item=self.k.by_child[cursor]
    if item[0] in self.names:moving=True;break
    cursor=item[1]
   for collision in link.findall('collision'):
    mesh=collision.find('geometry/mesh')
    if mesh is None:continue
    shape=coal.MeshLoader().load(str(root/'urdf'/mesh.get('filename')),np.ones(3));shape.buildConvexRepresentation(True);shape=shape.convex
    local=np.eye(4);o=collision.find('origin')
    if o is not None:local[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');local[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
    vertices=np.asarray(shape.points());center=vertices.mean(0);radius=float(np.linalg.norm(vertices-center,axis=1).max())
    # Conservative planning proxy for observed wrist-connector contacts.
    # The simulated collision meshes and contact reporting are unchanged.
    if keep_right_side and name in ('left_connector_link','right_connector_link'):
     lo,hi=vertices.min(0),vertices.max(0);local[:3,3]+=local[:3,:3]@((lo+hi)/2);shape=coal.Box(*(hi-lo));center=np.zeros(3);radius=float(np.linalg.norm(hi-lo)/2)
    obj=(name,shape,local,center,radius)
    if moving:self.moving.append(obj)
    else:self.fixed.append((name,shape,frames[name]@local,center,radius))
  C=np.eye(4);C[:3,:3]=R.from_quat(np.asarray(cube_quaternion)[[1,2,3,0]]).as_matrix();C[:3,3]=np.asarray(cube_position)-[0,0,1.15]
  # Route planning keeps every right-hand/arm shape 30 mm outside the cube.
  # Actual fingertip contact is only allowed in the separate slow approach.
  self.fixed.append(('cube_route_envelope',coal.Box(.117,.117,.117),C,np.zeros(3),.102))
  self.calls=0;self.last_pair=None
 def distance(self,a,b):
  n,m,t,c,r=a;nn,mm,tt,cc,rr=b
  ca=t[:3,:3]@c+t[:3,3];cb=tt[:3,:3]@cc+tt[:3,3]
  if np.linalg.norm(ca-cb)>r+rr+.004:return 1.
  result=coal.DistanceResult();d=float(coal.distance(m,coal.Transform3s(t[:3,:3],t[:3,3]),mm,coal.Transform3s(tt[:3,:3],tt[:3,3]),self.request,result))
  if d<.004:self.last_pair=(n,nn,d)
  return d
 def valid(self,x):
  self.calls+=1
  limits=np.array([self.k.limits[n] for n in self.names])
  if np.any(x<limits[:,0]) or np.any(x>limits[:,1]):self.last_pair=('right_arm','soft_limits',0.);return False
  q=self.q|dict(zip(self.names,x));frames=self.k.fk(q);moving=[(n,m,frames[n]@t,c,r) for n,m,t,c,r in self.moving]
  if self.keep_right_side:
   if frames['right_palm_link'][1,3]>self.cube_y-.02 or frames['right_palm_link'][1,3]>frames['left_palm_link'][1,3]-.05 or frames['Link4_R'][1,3]>frames['Link4_L'][1,3]-.04:
    self.last_pair=('right_arm','noncrossing_side_envelope',0.);return False
  for a in moving:
   n,m,t,c,r=a
   # Table top is -0.25 m in robot-root coordinates.
   if (t[:3,:3]@c+t[:3,3])[2]-r<-.25:
    v=(np.asarray(m.points()) if hasattr(m,'points') else np.array([[x,y,z] for x in [-m.halfSide[0],m.halfSide[0]] for y in [-m.halfSide[1],m.halfSide[1]] for z in [-m.halfSide[2],m.halfSide[2]]]))@t[:3,:3].T+t[:3,3]
    if v[:,2].min()<-.246:self.last_pair=(n,'table',float(v[:,2].min()+.25));return False
   for b in self.fixed:
    if n=='Link1_R' and not b[0].startswith(('left_','Link')):continue
    if self.distance(a,b)<.003:return False
  # Hand posture is fixed and audited separately. Check its clearance from
  # non-adjacent arm links, and arm links against non-adjacent arm links.
  for i,a in enumerate(moving):
   for b in moving[i+1:]:
    na,nb=a[0],b[0]
    aa=na.startswith('Link') and na.endswith('_R');bb=nb.startswith('Link') and nb.endswith('_R')
    if not (aa or bb):continue
    if aa and bb:
     ia=int(na[4]);ib=int(nb[4]);
     if abs(ia-ib)<=1 or {ia,ib}=={5,7}:continue
    elif int((na if aa else nb)[4])>=5:continue
    if self.distance(a,b)<.003:return False
  return True
 def edge(self,a,b,resolution=.03):
  for u in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/resolution))+1))[1:]:
   if not self.valid((1-u)*a+u*b):return False
  return True

def search(scene,start,goal,timeout=80.):
 limits=np.array([scene.k.limits[n] for n in scene.names]);limits[:,0]+=.035;limits[:,1]-=.035
 for label,x in [('start',start),('goal',goal)]:
  if not scene.valid(x):raise ValueError(f'{label} collision: {scene.last_pair}')
 if scene.edge(start,goal):return [start,goal]
 rng=np.random.default_rng(1945);trees=[([start],[-1]),([goal],[-1])];begin=time.monotonic()
 def extend(tree,target):
  nodes,parents=tree;idx=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)));old=nodes[idx];delta=target-old;length=np.linalg.norm(delta);new=old+delta*min(1.,.25/max(length,1e-12))
  if not scene.edge(old,new):return None,False
  nodes.append(new);parents.append(idx);return len(nodes)-1,length<=.25
 def chain(tree,idx):
  result=[]
  while idx>=0:result.append(tree[0][idx]);idx=tree[1][idx]
  return result[::-1]
 for iteration in range(10000):
  if time.monotonic()-begin>timeout:break
  side=iteration%2;a,b=trees[side],trees[1-side]
  sample=b[0][0] if rng.random()<.15 else rng.uniform(limits[:,0],limits[:,1])
  idx,_=extend(a,sample)
  if idx is None:continue
  target=a[0][idx]
  for _ in range(100):
   other,reached=extend(b,target)
   if other is None:break
   if reached:
    first=chain(a,idx);second=chain(b,other);path=first+second[-2::-1]
    if side:path=path[::-1]
    for _ in range(120):
     i,j=sorted(rng.integers(0,len(path),2))
     if j>i+1 and scene.edge(path[i],path[j],.025):path=path[:i+1]+path[j:]
    print('SEARCH',iteration,'checks',scene.calls,'seconds',time.monotonic()-begin,file=sys.stderr)
    return path
 raise ValueError(f'No route within {timeout} s; nodes {[len(t[0]) for t in trees]}, checks {scene.calls}')

if __name__=='__main__':
 request=json.loads(Path(sys.argv[1]).read_text());scene=CollisionScene(request['initial'],request['opened'],request['cube_position'],request['cube_quaternion'],request.get('keep_right_side',False));names=scene.names
 path=search(scene,np.array([request['initial'][n] for n in names]),np.array([request['goal'][n] for n in names]))
 Path(sys.argv[2]).write_text(json.dumps({'right_names':names,'waypoints':[p.tolist() for p in path],'collision_checks':scene.calls},indent=2))
