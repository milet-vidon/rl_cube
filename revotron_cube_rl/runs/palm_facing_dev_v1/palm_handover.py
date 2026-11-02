"""Corrected handover: left palm turns toward the right hand, wrists stay ordered."""
import numpy as np
from tilt_regrasp import TiltPlanner, TiltTeacher, MirrorArrival, set_right_arm_gains


class PalmFacingPlanner(TiltPlanner):
    tilt_degrees = -90.
    grasp_symmetry = np.diag([1., -1., -1.])
    right_contact_axis = 0
    right_thumb_sign = -1
    pregrasp_offset_cube = np.array([0., 0., -.10])
    little_contact_z = -.024
    keep_right_side = True


class PalmFacingTeacher(TiltTeacher):
    planner_type = PalmFacingPlanner
