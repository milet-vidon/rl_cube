"""Map measured layer orientation to the original turn planner's reference frame.

The planner reports -log(C0.T @ P @ P_ref.T @ C0).z in degrees.
Reconstructing a layer orientation as L = C @ Rz(-face_angle) gives the
same representation for its world rotation relative to initial Core C0:
-log(C0.T @ C @ Rz(-face_angle)).z.

This reconstruction uses the measured median layer z-angle. It does not recover
individual cubelet off-axis rotations or inter-cubelet dispersion. All original
geometry, contact and raw velocity acceptance gates must remain separate.
It returns a principal rotation-vector projection, not an unwrapped angle.
Initial nonzero face angle is deliberately retained; zeroing it without also
changing the planner reference would introduce another frame mismatch.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Expected a finite Core quaternion in wxyz order')
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Core quaternion has invalid norm')
    q = q / norm
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def planning_frame_layer_angle(initial_core_wxyz, current_core_wxyz, face_angle_deg):
    """Return signed z(log) layer angle in the planner's initial Core frame.

    Quaternion inputs are world orientations in wxyz order. face_angle_deg is
    the existing measured relative-to-current-Core layer metric. A positive
    face angle corresponds to Rz(-face_angle), matching the existing planner.
    """
    face = float(face_angle_deg)
    if not math.isfinite(face):
        raise ValueError('Nonfinite measured layer angle')
    initial = _rotation(initial_core_wxyz)
    current = _rotation(current_core_wxyz)
    layer_in_initial = initial.T @ current @ Rotation.from_euler('z', -face, degrees=True).as_matrix()
    angle = float(-np.degrees(Rotation.from_matrix(layer_in_initial).as_rotvec()[2]))
    if not math.isfinite(angle):
        raise ValueError('Nonfinite planning-frame layer angle')
    return angle
