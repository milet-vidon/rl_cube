"""Local Tianji hand feasibility for thumb-corner versus four-finger opposition.

Uses this project's URDF and named joints. No Noro hardware code is executed.
Fingertip FK is a candidate contact construction, not a physics success claim.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinematics import RobotKinematics

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')

class HandGeometry:
    def __init__(self, side='left'):
        self.side = side
        self.k = RobotKinematics()
        self.palm_from_base = np.linalg.inv(self.k.fk_link({},f'{side}_palm_link')) @ self.k.fk_link({},f'{side}_hand_base_link')
        self.chains = {}
        self.names = {}
        for finger in FINGERS:
            chain = []
            link = f'{side}_{finger}_tip_Link'
            while link != f'{side}_hand_base_link':
                item = self.k.by_child[link]
                chain.append(item)
                link = item[1]
            self.chains[finger] = list(reversed(chain))
            self.names[finger] = [x[0] for x in self.chains[finger] if x[4] is not None]

    def fk(self, finger, values):
        q = dict(zip(self.names[finger], values))
        t = self.palm_from_base.copy()
        for name, _, _, origin, axis in self.chains[finger]:
            joint = origin.copy()
            if axis is not None:
                joint[:3,:3] = joint[:3,:3] @ Rotation.from_rotvec(axis*q[name]).as_matrix()
            t = t @ joint
        return t

    def solve(self, finger, target, normal, initial=None):
        names = self.names[finger]
        limits = np.array([self.k.limits[n] for n in names])
        if initial is None:
            initial = np.array([1.0,.7,.3,.3,.1] if finger=='thumb' else [0.,.9,.5,.2])
        initial = np.clip(initial, limits[:,0]+.006, limits[:,1]-.006)
        def residual(x):
            t = self.fk(finger,x)
            pad = -t[:3,1] if finger=='thumb' else t[:3,0]
            if self.side=='right' and finger=='thumb': pad = -pad
            return np.r_[80*(t[:3,3]-target), .10*(pad-normal), .002*(x-initial)]
        fit = least_squares(residual, initial, bounds=(limits[:,0]+.005,limits[:,1]-.005), max_nfev=180)
        t = self.fk(finger,fit.x)
        return {'joints':dict(zip(names,fit.x.tolist())), 'tip':t[:3,3].tolist(),
                'target':np.asarray(target).tolist(), 'error_m':float(np.linalg.norm(t[:3,3]-target))}

    def candidate(self, center, clearance=.006, rotation=None):
        center = np.asarray(center)
        rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        half = .0285 + clearance
        # Two staggered rows fit four fingertips into a 57 mm face.
        contacts = {'thumb':np.array([-.018,-.020,-half]),
                    'index':np.array([-.021,-.024,half]),
                    'middle':np.array([.021,-.024,half]),
                    'ring':np.array([-.021,.024,half]),
                    'little':np.array([.021,.024,half])}
        fingers = {}
        for finger, local in contacts.items():
            normal = rotation @ np.array([0.,0.,1. if finger=='thumb' else -1.])
            fingers[finger] = self.solve(finger, center+rotation@local, normal)
        return {'center_palm':center.tolist(),'cube_rotation_palm':rotation.tolist(),
                'clearance_m':clearance,'fingers':fingers,
                'max_tip_error_m':max(f['error_m'] for f in fingers.values()),
                'joint_targets':{n:v for f in fingers.values() for n,v in f['joints'].items()}}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    hand=HandGeometry();candidates=[]
    for cx in [.045,.055,.065]:
        for cy in [-.005,0.,.005]:
            for cz in [.115,.13,.14]:
                for angle in [35.,45.,55.]:
                    c=hand.candidate([cx,cy,cz],rotation=Rotation.from_euler('z',angle,degrees=True).as_matrix())
                    c['face_roll_deg']=angle;candidates.append(c)
        print('GRID',cx,'best error',min(x['max_tip_error_m'] for x in candidates),flush=True)
    candidates.sort(key=lambda x:x['max_tip_error_m'])
    Path(a.output).write_text(json.dumps({'kind':'kinematic_candidates_not_physics_acceptance','candidates':candidates},indent=2))
    print(json.dumps(candidates[0],indent=2))
