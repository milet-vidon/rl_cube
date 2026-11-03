import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,'revotron_cube_rl')
from safe_collision_plan import CollisionScene
r=json.loads(Path('/tmp/wrist_direction_20261103/planned_candidate.json').read_text());s=CollisionScene(r['initial'],{},r['cube_position'],r['cube_quaternion']);f0=s.k.fk(r['initial']);moving=[];fixed=[]
for n,m,t,c,rad in s.fixed:
 if n=='cube_route_envelope':continue
 if n.startswith('left_') or n in [f'Link{i}_L'for i in range(1,8)]:moving.append((n,m,np.linalg.inv(f0[n])@t,c,rad))
 else:fixed.append((n,m,t,c,rad))
fixed += [(n,m,f0[n]@t,c,rad)for n,m,t,c,rad in s.moving]
checks=0;closest=(1.,None)
for step in range(r['wrist_start']-1,len(r['path']),5):
 q=dict(zip(r['joint_names'],r['path'][step]));f=s.k.fk(q);shapes=[(n,m,f[n]@t,c,rad)for n,m,t,c,rad in moving]
 for a in shapes:
  n=a[0]
  for b in fixed:
   if n=='Link1_L' and not b[0].startswith(('right_','Link')):continue
   d=s.distance(a,b);checks+=1
   if d<closest[0]:closest=(d,[step,n,b[0]])
  for b in shapes:
   if not (n.startswith('Link') and n.endswith('_L')):continue
   other=b[0]
   if other==n:continue
   if other.startswith('Link')and other.endswith('_L'):
    if abs(int(n[4])-int(other[4]))<=1 or {int(n[4]),int(other[4])}=={5,7}:continue
   elif int(n[4])>=5:continue
   d=s.distance(a,b);checks+=1
   if d<closest[0]:closest=(d,[step,n,b[0]])
report={'passed':closest[0]>=.003,'minimum_observed_m':closest[0],'closest_pair':closest[1],'checks':checks,'sampling_stride_control_steps':5,'scope':'left_arm_new_adjustment_vs_body_right_arm_and_nonadjacent_left_arm','not_a_physics_or_contact_certificate':True}
Path('/tmp/wrist_direction_20261103/geometry_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));raise SystemExit(0 if report['passed']else 1)
