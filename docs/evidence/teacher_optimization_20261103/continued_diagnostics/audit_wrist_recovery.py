from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
from wrist_direction import WristDirectionPlanner
from adaptive_wrist_teacher import AdaptiveWristPlanner
reports=[]
for seed in (9801,9802,9803):
    run=Path(f'runs/centered_left_independent_{seed}_20261103');rows=json.loads((run/'trajectory.json').read_text())
    cfg=json.loads((run/'config.json').read_text());names=cfg['joint_names'];q=dict(zip(names,json.loads((run/'left_plan.json').read_text())[-1]))
    row=rows[-1];desired=R.from_euler('x',-90,degrees=True)*R.from_quat(np.array(row['tilt_reference_wxyz'])[[1,2,3,0]])
    actual=R.from_quat(np.array(row['cube_quaternion_wxyz'])[[1,2,3,0]])
    error=(desired*actual.inv()).as_rotvec();angle=np.linalg.norm(error)
    delta=R.from_rotvec(error*min(.5,np.radians(1.)/angle)).as_matrix()
    planner=WristDirectionPlanner('opposition_grasp_v20.json');planner.wrist_goal=np.radians(12.)
    palm=planner.k.fk_link(q,'left_palm_link')
    holds=[r for r in rows if r['phase']=='tilt_hold'];center=np.array(holds[0]['cube_position'] if holds else row['cube_position'])-[0,0,cfg['robot_z']]
    position=center+delta@(palm[:3,3]-center);rotation=delta@palm[:3,:3]
    report={'seed':seed,'request':{'q':q,'position':position.tolist(),'rotation':rotation.tolist()},'reconstructed_request':True}
    for name,cls in [('baseline',WristDirectionPlanner),('adaptive',AdaptiveWristPlanner)]:
        pl=cls('opposition_grasp_v20.json');pl.wrist_goal=np.radians(12.)
        try:
            result=pl.fit_tilt(q,'left',position,rotation);pose=pl.k.fk_link(result,'left_palm_link')
            report[name]={'passed':True,'position_error_m':float(np.linalg.norm(pose[:3,3]-position)),
                          'angle_error_deg':float(np.degrees(R.from_matrix(rotation@pose[:3,:3].T).magnitude())),
                          'diagnostics':pl.diagnostics}
        except ValueError as e:report[name]={'passed':False,'failure':str(e)}
    reports.append(report);print(seed,{k:report[k] for k in ('baseline','adaptive')},flush=True)
Path('../work/teacher_optimization_20261103/wrist_recovery_audit.json').write_text(json.dumps(reports,indent=2))
