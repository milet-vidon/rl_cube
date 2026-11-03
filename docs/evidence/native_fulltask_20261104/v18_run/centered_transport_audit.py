"""Measured qualification for a centered, natural-wrist left-hand hold."""
import numpy as np
from scipy.spatial.transform import Rotation
from tilt_audit import contact_topology
from wrist_audit import wrist_geometry
from elbow_audit import elbow_metrics


def assess_center_row(row, names, goal_position, goal_quaternion):
    wrist = wrist_geometry(row, names)
    measured = row['measured_elbow']
    elbow = elbow_metrics(measured['shoulder_root_m'], measured['elbow_root_m'], measured['wrist_root_m'])
    actual = Rotation.from_quat(np.asarray(row['cube_quaternion_wxyz'])[[1, 2, 3, 0]])
    desired = Rotation.from_quat(np.asarray(goal_quaternion)[[1, 2, 3, 0]])
    orientation_error = float(np.degrees((actual * desired.inv()).magnitude()))
    position_error = float(np.linalg.norm(np.asarray(row['cube_position']) - goal_position))
    conditions = {
        'five_left_contacts': all(contact_topology(row, 'left').values()),
        'stable_lift': row['height_m'] >= .08 and row['linear_speed'] < .03 and row['angular_speed'] < .20,
        'cube_assembled': row['layer_measurement']['assembled_for_grasp'],
        'center_position': abs(row['cube_position'][1]) < .03 and position_error < .015,
        'cube_orientation': orientation_error < 5,
        'wrist_direction': wrist['joint6_deg'] >= 5 and wrist['side_bend_deg'] <= -5,
        'natural_wrist': wrist['forearm_direction_palm_frame'][2] > .3 and wrist['wrist_inside_elbow_m'] >= .05,
        'elbow_outward': elbow['outward_from_shoulder_m'] >= .16 and elbow['upper_arm_outward_deg'] >= 35,
        'palm_toward_right': np.asarray(row['measured_hand_frames']['left_palm_link'])[1, 0] <= -.6,
        'safe': row['self_contact_impulse'] <= 1e-6 and row['robot_table_impulse'] <= 1e-6 and not (row['drop'] or row['cube_table_contact'] or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])),
    }
    return {'qualified': bool(all(conditions.values())), 'failed_conditions': [key for key, value in conditions.items() if not value], 'position_error_m': position_error, 'orientation_error_deg': orientation_error, 'wrist': wrist, 'elbow': elbow}
