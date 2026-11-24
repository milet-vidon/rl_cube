"""Development score: one coherent face turn while the hand keeps the cube aloft.

Body attitudes are expressed in the moving Core frame; rotations of the whole cube
cancel. All six faces are eligible, in either direction. No active face-turn
command is implied by observing a turn.
"""
import numpy as np
from scipy.spatial.transform import Rotation

FACES = ('pX', 'nX', 'pY', 'nY', 'pZ', 'nZ')
ANGLE_DEG = 15.0
COHERENCE_DEG = 5.0
MAX_JOINT_SEPARATION_M = .002

def face_samples(rows, face):
    """Return one signed layer angle and assembly-consistency flag per frame."""
    if face not in FACES: raise ValueError(face)
    names=sorted(set(rows[0]['cubelets'])-{'Core'})
    group=np.array([face in n.split('_') for n in names])
    if len(names)!=26 or group.sum()!=9: raise ValueError('Expected 9 of 26 cubelets on face')
    keys=['Core']+names
    q=np.asarray([[r['cubelets'][n]['quaternion_wxyz'] for n in keys] for r in rows])
    if not np.all(np.isfinite(q)) or not np.allclose(np.linalg.norm(q,axis=-1),1,atol=.002): raise ValueError('Invalid cubelet orientation')
    # Inverse unit core quaternion times each cubelet; SciPy handles normalisation.
    raw=Rotation.from_quat(q[:,:, [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,27,3,3)
    rel=raw[:,:1].swapaxes(-2,-1)@raw[:,1:]
    rot=Rotation.from_matrix(rel.reshape(-1,3,3))
    quat=rot.as_quat().reshape(-1,26,4)
    quat=np.where(quat[:,:,3:]<0,-quat,quat)
    axis=np.eye(3)['XYZ'.index(face[1])] * (1 if face[0]=='p' else -1)
    signed=2*np.arctan2(quat[:,:,:3]@axis,quat[:,:,3])
    # Express the nine angles near the first to avoid discontinuities at +/-180.
    selected=signed[:,group]
    aligned=selected[:,:1]+np.angle(np.exp(1j*(selected-selected[:,:1])))
    angle=np.rad2deg(np.median(aligned,axis=1))
    twist=Rotation.from_rotvec((signed[:,:,None]*axis).reshape(-1,3))
    off=(twist.inv()*rot).magnitude().reshape(-1,26)
    sep=np.asarray([r['layer_measurement']['maximum_joint_origin_separation_m'] for r in rows])
    span=np.rad2deg(np.ptp(aligned,axis=1))
    ok=((span<=COHERENCE_DEG)&(np.rad2deg(off[:,group].max(axis=1))<=COHERENCE_DEG)
        &(np.rad2deg(rot.magnitude().reshape(-1,26)[:,~group].max(axis=1))<=COHERENCE_DEG)
        &(sep>=0)&(sep<=MAX_JOINT_SEPARATION_M))
    return angle,ok

def summarize_face(rows,cfg):
    from contact_hold_simple_rotation import build_evidence
    res={};first=rows[0]['step']
    for face in FACES:
        angle,ok=face_samples(rows,face)
        ev=build_evidence(rows,cfg,assembly_check=lambda r: bool(ok[r['step']-first]))
        prior=ev['start_by_end']; hold=ev['hold_frames']; post=ev['post_frames']
        peak=0.; best=0.; win=None
        for end,start in enumerate(prior):
            if end+1-start<hold: continue
            pivot=end-post
            if pivot-start<ev['confirm_frames']: continue
            # Before -> first attainment, followed by a half second held on the
            # turned side of the threshold. Only net signed change is credited.
            pre=angle[start:pivot]; last=angle[pivot:end+1]
            delta=(last[None,:]-pre[:,None]+180.)%360.-180.
            sustained=np.maximum(delta.min(axis=1),(-delta).min(axis=1))
            confirmed=np.lib.stride_tricks.sliding_window_view(sustained,ev['confirm_frames']).min(axis=1)
            idx=int(np.argmax(confirmed));score=float(confirmed[idx])
            peak=max(peak,float(np.abs(delta).max()))
            if score>best:
                best=score;win=[rows[start+idx]['step'],rows[pivot]['step'],rows[end]['step']]
        res[face]=dict(sustained_turn_deg=best,max_in_grasp_deg=peak,steps_start_reached_held=win)
    face=max(res,key=lambda f:res[f]['sustained_turn_deg'])
    return dict(qualified=res[face]['sustained_turn_deg']>=ANGLE_DEG, best_face=face if res[face]['sustained_turn_deg'] else None,
                sustained_turn_deg=res[face]['sustained_turn_deg'],threshold_deg=ANGLE_DEG,per_face=res,
                definition='one face: all 9 agree within 5 degrees, other 17 stay within 5 degrees of Core, joint origins within 2 mm; either turn direction; sustained after turning; no dropped/tumbling cube')
