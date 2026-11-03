import json,sys,numpy as np
from pathlib import Path
from opposed_grasp_plan import OppositionScene,FINGERS
from opposed_grasp_audit import posture
request=json.loads(Path(sys.argv[1]).read_text());plan=json.loads(Path(sys.argv[2]).read_text())
scene=OppositionScene(request);names=request['joint_names']
q=dict(zip(names,next(path[-1] for phase,path in plan['phases'] if phase=='opposition_hold')))
frames=scene.k.fk(q);contacts={}
for f in FINGERS:
 a=plan['contact_anchors'][f];t=frames[a['link']];point=scene.C.T@(t[:3,3]+t[:3,:3]@a['point_local']-scene.center);contacts[f]=point.tolist()
row={'q':[q[n] for n in names],'cube_position':request['cube_position'],'measured_hand_frames':{n:t.tolist() for n,t in frames.items()},'measured_elbow':{'shoulder_root_m':frames['Link2_L'][:3,3].tolist(),'elbow_root_m':frames['Link4_L'][:3,3].tolist(),'wrist_root_m':frames['Link7_L'][:3,3].tolist()}}
scene.q=q;scene.contact=True
report={'posture':posture(row,names),'collision_free':scene.valid(np.array([q[n] for n in scene.names])),'contact_anchors_cube_m':contacts,'path_checks':plan['checks'],'physics_validated':False}
Path(sys.argv[3]).write_text(json.dumps(report,indent=2));print(json.dumps(report))
