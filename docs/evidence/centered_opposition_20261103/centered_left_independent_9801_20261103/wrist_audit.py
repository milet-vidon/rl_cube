"""Signed wrist side bend from measured forearm and palm frames."""
import numpy as np
from elbow_gate import combined_hold


def wrist_geometry(row,names):
    e=np.asarray(row['measured_elbow']['elbow_root_m']);w=np.asarray(row['measured_elbow']['wrist_root_m'])
    length=np.linalg.norm(w-e)
    if length<.1:raise ValueError('Invalid measured forearm length')
    palm=np.asarray(row['measured_hand_frames']['left_palm_link']);local=palm[:3,:3].T@((w-e)/length)
    return {'joint6_deg':float(np.degrees(row['q'][names.index('Joint6_L')])),
            'side_bend_deg':float(np.degrees(np.arctan2(local[1],local[2]))),
            'flexion_deg':float(np.degrees(np.arctan2(local[0],local[2]))),
            'forearm_direction_palm_frame':local.tolist(),
            'wrist_inside_elbow_m':float(e[1]-w[1])}


def assess_wrist(rows,names):
    checked=[]
    for row in rows:
        g=wrist_geometry(row,names)
        valid=(g['joint6_deg']>=5 and g['side_bend_deg']<=-5 and g['wrist_inside_elbow_m']>=.05
               and g['forearm_direction_palm_frame'][2]>.3)
        checked.append(row if valid else row|{'phase':'wrist_not_qualified'})
    report=combined_hold(checked,names)
    report.update(final=wrist_geometry(rows[-1],names) if rows else None,
                  scope='J6_side_bend_reversed_with_unchanged_palm_orientation',
                  failures=[] if report['accepted']else['no_3s_jointly_qualified_reversed_wrist_hold'])
    return report
