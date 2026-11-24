import json,sys,numpy as np
from pathlib import Path
from opposed_grasp_plan import OppositionScene
from opposed_grasp_audit import posture
r=json.loads(Path(sys.argv[1]).read_text());p=json.loads(Path(sys.argv[2]).read_text());scene=OppositionScene(r);names=r['joint_names']
out={}
for phase,path in p['phases']:
 if phase not in ('opposition_hold','opposed_face_turn','opposed_face_hold'):continue
 reports=[]
 for point in path[::10]+[path[-1]]:
  q=dict(zip(names,point));f=scene.k.fk(q)
  row={'q':point,'cube_position':r['cube_position'],'measured_hand_frames':f,'measured_elbow':{'shoulder_root_m':f['Link2_L'][:3,3],'elbow_root_m':f['Link4_L'][:3,3],'wrist_root_m':f['Link7_L'][:3,3]}}
  reports.append(posture(row,names))
 out[phase]={'all_qualified':all(x['qualified'] for x in reports),'minimum_right_facing':min(x['right_facing_left_cos'] for x in reports)}
print(json.dumps(out));Path(sys.argv[3]).write_text(json.dumps(out,indent=2))
