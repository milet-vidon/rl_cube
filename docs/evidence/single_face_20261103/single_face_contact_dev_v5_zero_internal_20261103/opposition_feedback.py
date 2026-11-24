"""Bounded, state-based fingertip arrival correction for the simulation teacher."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class FingerArrival:
    def __init__(self,kinematics,grasp,joint_names,robot_z):
        self.k=kinematics;self.anchors=grasp.get('contact_anchors',{})
        self.names=joint_names;self.robot_z=robot_z;self.targets={};self.tick=0
        self.reference=grasp['joint_targets']

    def update(self,measured,nominal,contacts,cube_position,cube_quaternion):
        q=dict(zip(self.names,measured));out=nominal.copy()
        rotation=Rotation.from_quat(np.asarray(cube_quaternion)[[1,2,3,0]]).as_matrix()
        position=np.asarray(cube_position)-[0.,0.,self.robot_z]
        for f,anchor in self.anchors.items():
            names=[n for n in self.reference if n.startswith('left_'+f+'_')]
            if self.tick%3==0 and contacts.get(f,0.)<=1e-5:
                ref=np.array([self.reference[n] for n in names]);lim=np.array([self.k.limits[n] for n in names])
                # Lateral spread is already constrained by neighbouring fingers.
                # Arrival correction primarily bends the MCP/PIP/DIP joints.
                radius=np.array([.005 if 'MPR' in n else .30 for n in names])
                lo=np.maximum(lim[:,0]+.02,ref-radius);hi=np.minimum(lim[:,1]-.02,ref+radius)
                target=position+rotation@np.array(anchor['target_cube'])
                def residual(x):
                    t=self.k.fk_link(q|dict(zip(names,x)),anchor['link'])
                    point=t[:3,3]+t[:3,:3]@anchor['point_local']
                    local_error=rotation.T@(point-target)
                    return np.r_[np.array([60.,15.,60.])*local_error,.015*(x-ref)]
                fit=least_squares(residual,np.clip([q[n] for n in names],lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=30)
                self.targets.update(zip(names,fit.x))
            for n in names:
                i=self.names.index(n);previous=self.targets.get('_applied_'+n,float(nominal[i]))
                # Once a finger touches, hold its applied target instead of
                # continuing towards an unreachable surface inside the object.
                target=previous if contacts.get(f,0.)>1e-5 else self.targets.get(n,previous)
                out[i]=previous+np.clip(target-previous,-.005,.005)
                self.targets['_applied_'+n]=float(out[i])
        self.tick+=1
        return out
