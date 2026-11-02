"""Independent contact-role and object-face checks for the asymmetric grasp."""
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS=('thumb','index','middle','ring','little')

def contact_roles(row):
    result={f:False for f in FINGERS};corner=False
    if 'cube_quaternion_wxyz' not in row:return result,corner
    q=np.asarray(row['cube_quaternion_wxyz']);inv=Rotation.from_quat(q[[1,2,3,0]]).inv()
    for contact in row['finger_contact_points']:
        if contact['side']!='left' or contact['impulse_Ns']<=1e-7:continue
        f=contact['finger'];point=inv.apply(np.asarray(contact['position_world_m'])-row['cube_position'])
        face=-.0285 if f=='thumb' else .0285
        on_face=abs(point[0]-face)<=.006 and np.max(np.abs(point[1:]))<=.0345
        result[f]|=bool(on_face)
        if f=='thumb' and on_face:corner|=bool(np.linalg.norm(np.abs(point[1:])-.0285)<=.018)
    return result,corner

def assess_opposition(rows,hz=30,hold_seconds=3.):
    window=rows[-round(hz*hold_seconds):];n=max(1,len(window))
    coverage={f:sum(r['finger_contact_impulses']['left'][f]>1e-5 for r in window)/n for f in FINGERS}
    roles=[contact_roles(r) for r in window]
    face_coverage={f:sum(x[0][f] for x in roles)/n for f in FINGERS}
    corner_coverage=sum(x[1] for x in roles)/n
    passed=len(window)==round(hz*hold_seconds) and coverage['thumb']>=.9 and face_coverage['thumb']>=.9 and corner_coverage>=.9
    passed=passed and all(coverage[f]>=.5 and face_coverage[f]>=.5 for f in FINGERS[1:])
    return {'passed':passed,'main_hand':'left','support_hand':'right','hold_contact_coverage':coverage,
            'correct_face_contact_coverage':face_coverage,'thumb_corner_contact_coverage':corner_coverage,
            'required_thumb_coverage':.9,'required_each_four_finger_coverage':.5,
            'expected_faces':'thumb on cube -X corner, four fingers on cube +X',
            'corner_distance_max_m':.018,'face_position_tolerance_m':.006}
