"""Core-relative, signed layer measurements. Public quaternions are wxyz."""
import numpy as np
from scipy.spatial.transform import Rotation

def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    return Rotation.from_quat(q[[1,2,3,0]])

def layer_delta(initial_core, initial_face, core, face, axis):
    axis=np.asarray(axis,dtype=float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:
        raise ValueError('Invalid layer axis')
    axis=axis/np.linalg.norm(axis)
    start=rotation(initial_core).inv()*rotation(initial_face)
    current=rotation(core).inv()*rotation(face)
    delta=current*start.inv()
    q=delta.as_quat();projection=float(np.dot(q[:3],axis))
    angle=(np.degrees(2*np.arctan2(projection,q[3]))+180)%360-180
    twist=Rotation.from_rotvec(axis*np.radians(angle))
    swing=np.degrees((delta*twist.inv()).magnitude())
    return {'signed_angle_deg':float(angle),'off_axis_error_deg':float(swing)}
