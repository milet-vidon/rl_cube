import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/brainco/rl_cube/github_project/revotron_cube_rl')
from centered_transport_plan import CenterScene
p=Path('/tmp/j56_center_20261103');r=json.loads((p/'j56_request.json').read_text());plan=json.loads((p/'j56_plan.json').read_text());s=CenterScene(r['initial'],r['cube_position'],r['cube_quaternion'],r['torso_colliders']);names=r['joint_names'];end=dict(zip(names,plan['phases'][-1][1][-1]));fingers=('thumb','index','middle','ring','little')
def check(q):
 if not s.valid_full(q):return False,s.last_pair
 frames=s.k.fk(q);shapes=[(n,m,frames[n]@t,c,rad)for n,m,t,c,rad in s.geometry if n.startswith('left_')]
 for i,a in enumerate(shapes):
  fa=next((f for f in fingers if a[0].startswith('left_'+f+'_')),None)
  if not fa:continue
  for b in shapes[i+1:]:
   fb=next((f for f in fingers if b[0].startswith('left_'+f+'_')),None)
   if fb and fb!=fa and s.distance(a,b)<.003:return False,s.last_pair
 return True,None
for offset in [0,-.005,-.010,-.012,-.015]:
 for label,q in [('start',r['initial']),('end',end)]:
  state=q|{'left_index_MPR_joint':q['left_index_MPR_joint']+offset,'left_index_PIP_joint':q['left_index_PIP_joint']+(.003 if offset else 0)};print(offset,label,check(state),flush=True)
