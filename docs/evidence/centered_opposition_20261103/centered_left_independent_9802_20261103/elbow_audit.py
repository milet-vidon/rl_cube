"""Physical shoulder/elbow/wrist audit in the fixed robot root frame."""
import numpy as np

MIN_OUTWARD_M=.16
MIN_OUTWARD_DEG=35.


def measured_elbow(env):
    from isaac_compat import tensor
    ids=[env.robot.find_bodies(n)[0][0] for n in ['Link2_L','Link4_L','Link7_L']]
    s,e,w=tensor(env.robot.data.body_link_pos_w)[0,ids].numpy()-[0,0,env.cfg['robot_z']]
    return elbow_metrics(s,e,w)


def elbow_metrics(s,e,w):
    s,e,w=map(np.asarray,(s,e,w));upper=e-s;axis=w-s
    radial=e-(s+axis*np.dot(e-s,axis)/max(np.dot(axis,axis),1e-12))
    return {'shoulder_root_m':s.tolist(),'elbow_root_m':e.tolist(),'wrist_root_m':w.tolist(),
            'outward_from_shoulder_m':float(upper[1]),
            'upper_arm_outward_deg':float(np.degrees(np.arctan2(upper[1],-upper[2]))),
            'outward_from_shoulder_wrist_axis_m':float(radial[1])}


def assess_elbow(rows,regrasp=False):
    phase='bilateral_hold' if regrasp else 'tilt_hold'
    streak=longest=0;missing=any('measured_elbow' not in r for r in rows)
    for row in rows:
        m=row.get('measured_elbow',{})
        good=(row['phase']==phase and m.get('outward_from_shoulder_m',-1)>=MIN_OUTWARD_M
              and m.get('upper_arm_outward_deg',-180)>=MIN_OUTWARD_DEG
              and row['height_m']>=.08 and not row['cube_table_contact']
              and row['linear_speed']<.03 and row['angular_speed']<.20
              and row['tilt_error_deg']<=5.)
        streak=streak+1 if good else 0;longest=max(longest,streak)
    failures=[]
    if missing:failures.append('missing_measured_shoulder_elbow_wrist')
    if longest<90:failures.append('no_3s_outward_upper_arm_hold')
    return {'accepted':not failures,'failures':failures,'longest_outward_hold_seconds':longest/30,
            'minimum_outward_m':MIN_OUTWARD_M,'minimum_outward_degrees':MIN_OUTWARD_DEG,
            'final':rows[-1].get('measured_elbow') if rows else None}
