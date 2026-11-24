"""Measured geometry gates for the user's palm-facing, uncrossed handover."""
import numpy as np
from kinematics import RobotKinematics
from tilt_audit import assess_tilt, tilt_error, contact_topology
from scipy.spatial.transform import Rotation


def measured_hand_frames(env):
    from isaac_compat import tensor, quaternion_from_native
    frames={}
    for side in ('left','right'):
        body,offset,quat=env.frames[side]
        ids,_=env.robot.find_bodies(body)
        position=tensor(env.robot.data.body_link_pos_w)[0,ids[0]].numpy()
        q=quaternion_from_native(tensor(env.robot.data.body_link_quat_w)[0,ids[0]]).numpy()
        rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        matrix=np.eye(4)
        matrix[:3,:3]=rotation@Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        matrix[:3,3]=position+rotation@np.asarray(offset)-[0,0,env.cfg['robot_z']]
        frames[side+'_palm_link']=matrix.tolist()
        ids,_=env.robot.find_bodies('Link4_'+('L' if side=='left' else 'R'))
        elbow=np.eye(4);elbow[:3,3]=tensor(env.robot.data.body_link_pos_w)[0,ids[0]].numpy()-[0,0,env.cfg['robot_z']]
        frames['Link4_'+('L' if side=='left' else 'R')]=elbow.tolist()
    return frames


def hand_geometry(row, names, kinematics=None):
    kinematics = kinematics or RobotKinematics()
    frames = kinematics.fk(dict(zip(names, row['q'])))
    if 'measured_hand_frames' in row:
        frames.update({n:np.asarray(t) for n,t in row['measured_hand_frames'].items()})
    left, right = frames['left_palm_link'], frames['right_palm_link']
    direction = right[:3, 3] - left[:3, 3]
    direction /= max(np.linalg.norm(direction), 1e-9)
    return {
        'left_palm_position_root': left[:3, 3].tolist(),
        'right_palm_position_root': right[:3, 3].tolist(),
        'left_palm_normal_world': left[:3, 0].tolist(),
        'right_palm_normal_world': right[:3, 0].tolist(),
        'left_facing_right_cos': float(left[:3, 0] @ direction),
        'right_facing_left_cos': float(-right[:3, 0] @ direction),
        'palm_normal_dot': float(left[:3, 0] @ right[:3, 0]),
        'wrist_y_separation_m': float(left[1, 3] - right[1, 3]),
        'elbow_y_separation_m': float(frames['Link4_L'][1, 3] - frames['Link4_R'][1, 3]),
        'left_wrist_from_cube_y_m': float(left[1, 3] - row['cube_position'][1]),
        'right_wrist_from_cube_y_m': float(row['cube_position'][1] - right[1, 3]),
    }


def assess_handover(rows, names, regrasp=True, full_episode=True):
    report = assess_tilt(rows, regrasp=regrasp, full_episode=full_episode)
    failures = list(report['failures'])
    if any(row.get('tilt_degrees') != -90 for row in rows):
        failures.append('wrong_rotation_direction')
    kinematics = RobotKinematics()
    geometries = [hand_geometry(row, names, kinematics) for row in rows]
    active = [(r, g) for r, g in zip(rows, geometries) if r['phase'] not in
              ('ready', 'preshape', 'lower', 'approach', 'squeeze', 'primary_lift')]
    if any(g['wrist_y_separation_m'] < .03 or g['elbow_y_separation_m'] < .04 for _, g in active):
        failures.append('arms_crossed')
    hold = [(r, g) for r, g in active if r['phase'] == ('bilateral_hold' if regrasp else 'tilt_hold')]
    streak = longest = 0
    for row, geometry in hold:
        correct = (geometry['left_palm_normal_world'][1] <= -.6
                   and geometry['left_wrist_from_cube_y_m'] >= .02
                   and geometry['right_wrist_from_cube_y_m'] >= .02)
        if regrasp:
            correct &= (geometry['left_facing_right_cos'] >= .65
                        and geometry['right_facing_left_cos'] >= .65
                        and geometry['palm_normal_dot'] <= -.3)
        contacts = all(all(contact_topology(row, side).values())
                       for side in (('left', 'right') if regrasp else ('left',)))
        stable = (row['height_m'] >= .08 and not row['cube_table_contact']
                  and row['linear_speed'] < .03 and row['angular_speed'] < .20)
        streak = streak + 1 if correct and stable and contacts and tilt_error(row) <= 5 else 0
        longest = max(longest, streak)
    if longest < 90:
        failures.append('no_3s_palm_facing_uncrossed_hold')
    report.update(accepted=not failures, failures=failures,
                  semantic_task='left_palm_toward_right_hand_uncrossed_opposition_handover',
                  longest_palm_facing_hold_seconds=longest / 30,
                  final_hand_geometry=geometries[-1] if geometries else None)
    return report
