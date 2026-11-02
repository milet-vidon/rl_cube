"""Post-step acceptance for a whole-cube tilt and optional mirrored regrasp.

Uses measured poses and contact points, independently of controller counters.
This does not test any internal Rubik's-cube layer motion.
"""
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')


def rotation_wxyz(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def tilt_error(row):
    reference = row.get('tilt_reference_wxyz')
    if reference is None:
        return 90.0
    target = Rotation.from_euler('x', row.get('tilt_degrees', 90), degrees=True).as_matrix() @ rotation_wxyz(reference)
    actual = rotation_wxyz(row['cube_quaternion_wxyz'])
    return float(np.degrees(Rotation.from_matrix(target @ actual.T).magnitude()))


def contact_topology(row, side):
    """Cube-frame opposition: left X faces, mirrored right Y faces."""
    axis, thumb_sign = (0, -1) if side == 'left' else (row.get('right_contact_axis', 1), row.get('right_thumb_sign', 1))
    rotation = rotation_wxyz(row['cube_quaternion_wxyz'])
    center = np.asarray(row['cube_position'])
    impulses = {finger: 0.0 for finger in FINGERS}
    for contact in row['finger_contact_points']:
        if contact['side'] != side:
            continue
        finger = contact['finger']
        local = rotation.T @ (np.asarray(contact['position_world_m']) - center)
        sign = thumb_sign if finger == 'thumb' else -thumb_sign
        on_face = sign * local[axis] >= .0215 and np.max(np.abs(local)) <= .036
        at_corner = np.max(np.abs(np.delete(local, axis))) >= .010
        if on_face and (finger != 'thumb' or at_corner):
            impulses[finger] += contact['impulse_Ns']
    return {finger: impulses[finger] > 1e-5 for finger in FINGERS}


def assess_tilt(rows, regrasp=False, hz=30, full_episode=True):
    failures = []
    if not rows:
        return {'accepted': False, 'failures': ['empty_trajectory']}
    if full_episode:
        phases = {row['phase'] for row in rows}
        required = {'primary_lift', 'tilt_90', 'tilt_hold'}
        if regrasp:
            required |= {'right_close', 'bilateral_hold'}
        turned = [row for row in rows if row['phase'] == 'tilt_90']
        if not required <= phases or not turned or tilt_error(turned[0]) < 80.0:
            failures.append('missing_full_pickup_tilt_sequence')
    for key in ('self_contact_impulse', 'robot_table_impulse'):
        if any(row[key] > 1e-6 for row in rows):
            failures.append(key)
    for key in ('arm_soft_limit_hit', 'actual_arm_soft_limit_violation', 'drop'):
        if any(row[key] for row in rows):
            failures.append(key)
    if any(any(row['hard_limit_hit']) for row in rows):
        failures.append('hard_limit_hit')
    after_lift = [r for r in rows if r['phase'] in ('right_clear','tilt_90','align_right','tilt_hold','right_preshape','right_route','right_approach','right_close','bilateral_hold')]
    returned_to_table = any(r['cube_table_contact'] for r in after_lift)
    if returned_to_table:
        failures.append('cube_returned_to_table_after_lift')

    sides = ('left', 'right') if regrasp else ('left',)
    phase = 'bilateral_hold' if regrasp else 'tilt_hold'
    streak = []
    longest = 0
    qualifying_window = None
    for row in rows:
        topology = {side: contact_topology(row, side) for side in sides}
        contacts = all(t['thumb'] and sum(t[f] for f in FINGERS[1:]) >= 3 for t in topology.values())
        qualified = (row['phase'] == phase and tilt_error(row) <= 5.0
                     and row['height_m'] >= .08 and not row['cube_table_contact']
                     and row['linear_speed'] < .03 and row['angular_speed'] < .20 and contacts)
        streak = streak + [topology] if qualified else []
        longest = max(longest, len(streak))
        if len(streak) >= 3 * hz:
            window = streak[-3 * hz:]
            coverage = {side: {f: float(np.mean([t[side][f] for t in window])) for f in FINGERS} for side in sides}
            if all(v >= (.9 if f == 'thumb' else .5) for side in coverage.values() for f, v in side.items()):
                qualifying_window = coverage
    if qualifying_window is None:
        failures.append('no_3s_measured_opposition_hold_at_90deg')
    return {
        'accepted': not failures,
        'scope': 'whole_cube_tilt_and_mirrored_right_grasp' if regrasp else 'whole_cube_tilt_only',
        'full_episode': full_episode,
        'failures': failures,
        'final_tilt_error_deg': tilt_error(rows[-1]),
        'longest_measured_hold_seconds': longest / hz,
        'contact_coverage_in_qualifying_window': qualifying_window,
        'maximum_lift_m': max(row['height_m'] for row in rows),
        'final_lift_m': rows[-1]['height_m'],
        'cube_returned_to_table_after_lift': returned_to_table,
        'robot_table_impulse_sum_Ns': sum(row['robot_table_impulse'] for row in rows),
        'self_contact_impulse_sum_Ns': sum(row['self_contact_impulse'] for row in rows),
    }
