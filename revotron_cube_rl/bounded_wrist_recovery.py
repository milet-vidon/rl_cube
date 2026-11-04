"""Recover the requested palm pose within unchanged measured wrist/elbow gates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from elbow_audit import elbow_metrics

def recover_bounded_pose(planner,q,position,rotation,original):
    names=[f'Joint{i}_L' for i in range(1,8)];limits=np.array([planner.k.limits[n] for n in names])
    lo=limits[:,0]+np.radians(2.);hi=limits[:,1]-np.radians(2.);lo[5]=max(lo[5],np.radians(6.));ref=np.clip([q[n] for n in names],lo+1e-7,hi-1e-7)
    def measured(v):
        state=q|dict(zip(names,v));frames=planner.k.fk(state);palm=frames['left_palm_link'];s,e,w=[frames[n][:3,3] for n in ('Link2_L','Link4_L','Link7_L')]
        local=palm[:3,:3].T@((w-e)/np.linalg.norm(w-e));elbow=elbow_metrics(s,e,w)
        return state,palm,s,e,w,local,elbow
    def residual(v):
        _,palm,s,e,w,local,elbow=measured(v)
        return np.r_[300*(palm[:3,3]-position),60*R.from_matrix(rotation@palm[:3,:3].T).as_rotvec(),
                     300*max(.165-(e-s)[1],0),300*max(.055-(e-w)[1],0),
                     30*max(np.radians(7.)-v[5],0),10*max(v[5]-np.radians(16.),0),
                     10*max(.32-local[2],0),10*max(np.arctan2(local[1],local[2])+np.radians(6.),0),
                     1.*max(36.-elbow['upper_arm_outward_deg'],0),.02*(v-ref)]
    for wrist in (float(ref[5]),np.radians(10.),np.radians(8.)):
        start=ref.copy();start[5]=np.clip(wrist,lo[5]+1e-7,hi[5]-1e-7)
        fit=least_squares(residual,start,bounds=(lo,hi),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        state,palm,s,e,w,local,elbow=measured(fit.x)
        pos=float(np.linalg.norm(palm[:3,3]-position));angle=float(np.degrees(R.from_matrix(rotation@palm[:3,:3].T).magnitude()))
        if (pos<=.002 and angle<=1. and (e-s)[1]>=.162 and (e-w)[1]>=.055 and elbow['upper_arm_outward_deg']>=35.5
            and fit.x[5]>=np.radians(6.) and local[2]>.32 and np.arctan2(local[1],local[2])<=np.radians(-6.)):
            record={'kind':'bounded_wrist_pose_recovery','position_error_m':pos,'orientation_error_deg':angle,
                    'joint6_deg':float(np.degrees(fit.x[5])),'elbow_outward_m':float((e-s)[1]),'upper_arm_outward_deg':elbow['upper_arm_outward_deg'],'original_error':str(original)}
            planner.diagnostics.append(record);print('BOUNDED_WRIST_RECOVERY',record,flush=True);return state
    raise original
