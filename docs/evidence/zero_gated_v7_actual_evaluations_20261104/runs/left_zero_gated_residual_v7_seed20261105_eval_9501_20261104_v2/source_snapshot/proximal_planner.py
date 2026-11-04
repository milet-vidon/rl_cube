"""Proximal-motion preference adapted from Noro's per-joint posture-task design.

Noro PinkW protects shoulder/elbow motion with [5,5,3,3,.3,.3,.3].
This task intentionally reverses that preference; its hardware limits and TCP
calibration are not imported. Pose accuracy and the Revo3 soft limits stay gates.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from side_grasp import SidePlanner

PROXIMAL_WEIGHTS = np.array([.3,.3,.5,.8,4.,10.,10.])

class ProximalPlanner(SidePlanner):
    def fit(self,q,side,pos,rot,wrist=None):
        names=[f'Joint{i}_{"L" if side=="left" else "R"}' for i in range(1,8)]
        limits=np.array([self.k.limits[n] for n in names])
        q0=np.clip([q[n] for n in names],limits[:,0]+.002,limits[:,1]-.002)
        def residual(x):
            t=self.k.fk_link(q|dict(zip(names,x)),side+'_palm_link')
            return np.r_[30*(t[:3,3]-pos),4*Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec(),
                         .008*np.sqrt(PROXIMAL_WEIGHTS)*(x-q0),
                         .08*(x[6]-(self.wrist if wrist is None else wrist))]
        fit=least_squares(residual,q0,bounds=(limits[:,0]+.002,limits[:,1]-.002),
                          max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fit.x));t=self.k.fk_link(result,side+'_palm_link')
        error=np.linalg.norm(t[:3,3]-pos)
        angle=np.linalg.norm(Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec())
        if error>.002 or angle>np.radians(1):
            raise ValueError(f'Proximal IK infeasible: {side}, position={error}, angle={np.degrees(angle)}')
        return result

def motion_summary(segments,names):
    q=np.vstack([segments[0][1]]+[x[2] for x in segments])
    result={}
    for side in ['L','R']:
        columns=[names.index(f'Joint{i}_{side}') for i in range(1,8)]
        travel=np.abs(np.diff(q[:,columns],axis=0)).sum(axis=0)
        lift=[i for i,x in enumerate(segments) if x[0] in ['lift','hold']]
        lift_q=q[[min(lift)]+[i+1 for i in lift]][:,columns]
        lift_travel=np.abs(np.diff(lift_q,axis=0)).sum(axis=0)
        result[side]={'total_travel_deg':np.degrees(travel).tolist(),
                      'lift_travel_deg':np.degrees(lift_travel).tolist(),
                      'lift_distal_share':float(lift_travel[4:].sum()/max(lift_travel.sum(),1e-9))}
    return result

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    results=[]
    for xyz in [[.48,0.,.9285],[.465,-.015,.9285],[.495,.015,.9285]]:
        row={'xyz':xyz}
        for label,cls in [('baseline',SidePlanner),('proximal',ProximalPlanner)]:
            planner=cls();segments=planner.plan(np.array(xyz),.096,.025)
            row[label]=motion_summary(segments,planner.cfg['joint_names'])
        results.append(row);print(json.dumps(row),flush=True)
    Path(a.output).write_text(json.dumps(results,indent=2))
