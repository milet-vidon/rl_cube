"""Offline actual-target/contact/frame audit; no Isaac imports or simulator calls."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.spatial.transform import Rotation as R

root = Path(__file__).resolve().parents[2] / 'revotron_cube_rl'
sys.path.insert(0, str(root))
from kinematics import RobotKinematics
from turn_feedback_frame import planning_frame_layer_angle

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
k = RobotKinematics()


def rot(quaternion):
    return R.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(label, directory):
    run = root / 'runs' / directory
    rows = json.loads((run / 'trajectory.json').read_text())
    plan = json.loads((run / 'opposition_plan.json').read_text())
    request = json.loads((run / 'opposition_request.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    names = config['joint_names']
    turn = [row for row in rows if row['phase'] == 'opposed_face_turn']
    hold = [row for row in rows if row['phase'] == 'opposed_face_hold']
    baseline = rows[turn[0]['step'] - 1]
    C0 = rot(request['cube_quaternion'])
    Cturn = rot(baseline['cube_quaternion_wxyz'])
    Pturn = np.asarray(baseline['measured_hand_frames']['right_palm_link'])[:3, :3]
    close = next(path[-1] for phase, path in plan['phases'] if phase == 'right_close')
    Pref = k.fk_link(dict(zip(names, close)), 'right_palm_link')[:3, :3]
    baseq = dict(zip(names, baseline['q']))
    tracked = {}
    for finger in FINGERS:
        points = [p for p in baseline['right_cubelet_contacts'] if 'right_' + finger + '_' in p['robot_body']]
        p = max(points, key=lambda x: x['impulse_Ns'])
        link = p['robot_body'].split('/')[-1]
        frame = k.fk_link(baseq, link).copy()
        frame[:3, 3] += [0, 0, config['robot_z']]
        body = baseline['cubelets'][p['cubelet']]
        world = np.asarray(p['position_world_m'])
        tracked[finger] = {
            'link': link, 'cubelet': p['cubelet'],
            'hand_local': frame[:3, :3].T @ (world - frame[:3, 3]),
            'cubelet_local': rot(body['quaternion_wxyz']).T @ (world - body['position_m']),
        }

    def metrics(row):
        C = rot(row['cube_quaternion_wxyz'])
        P = np.asarray(row['measured_hand_frames']['right_palm_link'])[:3, :3]
        layer = row['layer_measurement']['face_angle_deg']
        mapped = planning_frame_layer_angle(request['cube_quaternion'], row['cube_quaternion_wxyz'], layer)
        planned_index = (row.get('turn_pacing') or {}).get('path_index', min(max(row['step'] - turn[0]['step'], 0), len(plan['turn_path_angles_deg']) - 1))
        theta = plan['turn_path_angles_deg'][planned_index]
        measured_palm = float(-np.degrees(R.from_matrix(C0.T @ P @ Pref.T @ C0).as_rotvec()[2]))
        world_delta = float(-np.degrees(R.from_matrix(Cturn.T @ P @ Pturn.T @ Cturn).as_rotvec()[2]))
        core_delta = float(-np.degrees(R.from_matrix(Cturn.T @ C).as_rotvec()[2]))
        selected = row['layer_measurement']['selected_cubelets']
        omega = {name: float(np.linalg.norm(row['cubelets'][name]['angular_velocity_rad_s'])) for name in selected}
        core_omega = np.asarray(row['cubelets']['Core']['angular_velocity_rad_s'])
        relative_omega = {name: float(np.linalg.norm(np.asarray(row['cubelets'][name]['angular_velocity_rad_s']) - core_omega)) for name in selected}
        if abs(max(relative_omega.values()) - row['layer_measurement']['maximum_selected_angular_velocity_rad_s']) > 1e-10:
            raise ValueError('Logged raw-relative layer speed does not match body velocities')
        right = row['opposition_audit']['right_contacts']
        q = dict(zip(names, row['q']))
        target = dict(zip(names, row['applied_target']))
        geometry = {}
        for finger in FINGERS:
            anchor = plan['contact_anchors'][finger]
            measured_frame = k.fk_link(q, anchor['link'])
            target_frame = k.fk_link(target, anchor['link'])
            measured_point = measured_frame[:3, :3] @ anchor['point_local'] + measured_frame[:3, 3] + [0, 0, config['robot_z']]
            target_point = target_frame[:3, :3] @ anchor['point_local'] + target_frame[:3, 3] + [0, 0, config['robot_z']]
            t = tracked[finger]
            link_frame = k.fk_link(q, t['link'])
            hand_point = link_frame[:3, :3] @ t['hand_local'] + link_frame[:3, 3] + [0, 0, config['robot_z']]
            cubelet = row['cubelets'][t['cubelet']]
            body_rotation = rot(cubelet['quaternion_wxyz'])
            cube_point = body_rotation @ t['cubelet_local'] + cubelet['position_m']
            displacement = body_rotation.T @ (hand_point - cube_point)
            ids = [i for i, n in enumerate(names) if n.startswith('right_' + finger + '_')]
            error = np.asarray(row['applied_target'])[ids] - np.asarray(row['q'])[ids]
            velocity = np.asarray(row['joint_velocity'])[ids]
            torque = 10. * error - velocity
            geometry[finger] = {
                'impulse_Ns': right['impulses_Ns'][finger],
                'general_cube_impulse_Ns': row['finger_contact_impulses']['right'][finger],
                'classified_contact': right['fingers'][finger],
                'anchor_in_current_Core_m': (C.T @ (measured_point - row['cube_position'])).tolist(),
                'anchor_actual_minus_target_current_Core_m': (C.T @ (measured_point - target_point)).tolist(),
                'finger_joint_target_minus_actual_rad': {names[i]: float(e) for i, e in zip(ids, error)},
                'approx_PD_torque_Nm': {names[i]: float(tau) for i, tau in zip(ids, torque)},
                'tracked_baseline_pair': {'link': t['link'], 'cubelet': t['cubelet'],
                                          'hand_minus_cube_material_point_in_current_cubelet_m': displacement.tolist(),
                                          'outward_y_displacement_m': float(displacement[1] * (-1 if finger == 'thumb' else 1))},
            }
        return {
            'step': row['step'], 'turn_tick': row['step'] - turn[0]['step'], 'phase': row['phase'],
            'planned_path_index': planned_index, 'planned_palm_angle_deg': theta,
            'measured_palm_planning_frame_deg': measured_palm,
            'palm_target_tracking_error_deg': theta - measured_palm,
            'layer_angle_current_Core_deg': layer, 'layer_angle_planning_frame_deg': mapped,
            'planned_lead_over_planning_frame_layer_deg': theta - mapped,
            'measured_palm_lead_over_planning_frame_layer_deg': measured_palm - mapped,
            'world_palm_delta_from_turn_start_deg': world_delta,
            'Core_delta_from_turn_start_deg': core_delta,
            'Core_raw_angular_speed_rad_s': row['angular_speed'],
            'selected_raw_absolute_angular_speeds_rad_s': omega,
            'maximum_selected_raw_absolute_angular_speed_rad_s': max(omega.values()),
            'selected_raw_relative_to_Core_angular_speeds_rad_s': relative_omega,
            'maximum_selected_raw_relative_to_Core_angular_speed_rad_s': max(relative_omega.values()),
            'left_impulses_Ns': row['finger_contact_impulses']['left'], 'right_fingers': geometry,
        }

    losses = {finger: next((r['step'] for r in turn if not r['opposition_audit']['right_contacts']['fingers'][finger]), None) for finger in FINGERS}
    selected_steps = {baseline['step'], turn[0]['step'], turn[-1]['step'], rows[-1]['step']}
    selected_steps.update(r['step'] for r in turn[::10])
    selected_steps.update(i for step in losses.values() if step is not None for i in range(step - 2, step + 2))
    output = {
        'run': directory, 'sha256': {name: sha(run / name) for name in ('trajectory.json', 'opposition_plan.json', 'opposition_request.json', 'experiment.json')},
        'turn_first_step': turn[0]['step'], 'turn_frames': len(turn), 'hold_frames': len(hold),
        'first_right_contact_loss_step': losses,
        'right_missing_turn_frames': {f: sum(not r['opposition_audit']['right_contacts']['fingers'][f] for r in turn) for f in FINGERS},
        'right_missing_hold_frames': {f: sum(not r['opposition_audit']['right_contacts']['fingers'][f] for r in hold) for f in FINGERS},
        'raw_selected_relative_to_Core_speed_min_max_rad_s': [min(r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for r in turn), max(r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for r in turn)],
        'raw_selected_relative_to_Core_speed_above_point2_frames': sum(r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] >= .2 for r in turn),
        'left_first_zero_impulse_turn_step': {f: next((r['step'] for r in turn if r['finger_contact_impulses']['left'][f] == 0), None) for f in FINGERS},
        'first_Core_speed_above_point2_step': next((r['step'] for r in turn if r['angular_speed'] >= .2), None),
        'selected_steps': [metrics(rows[i]) for i in sorted(selected_steps)],
    }
    return output, rows, plan


v15, A, p15 = summarize('v15', 'teacher_joint_ring_support_v15_20261103')
v16, B, p16 = summarize('v16', 'teacher_joint_supported_open_v16_20261103')
keys = ('q', 'applied_target', 'cube_position', 'cube_quaternion_wxyz', 'finger_contact_impulses', 'cubelets')
first_difference = {}
for key in keys:
    first_difference[key] = next((i for i, (a, b) in enumerate(zip(A, B)) if a[key] != b[key]), None)
report = {
    'scope': 'read-only offline v16 first-contact-loss analysis; no new physics',
    'v15': v15, 'v16': v16,
    'same_prefix': {'first_difference_step': first_difference, 'plan_paths_and_angles_exactly_equal': all(p15[key] == p16[key] for key in ('phases', 'turn_path_angles_deg', 'contact_anchors'))},
    'interpretation_limits': [
        'Tracked contact material point displacement is not mesh minimum distance: surface contacts may migrate to other material points.',
        'Approx PD torque is K10*(target-q)-D1*qd from logged control boundary state, not measured solver actuation or substep impulse.',
        'Frame-consistent mapped median angle omits off-axis/dispersion; raw relative omega_cubelet minus omega_Core remains a separate unchanged gate. Absolute raw speeds are separately reported.',
        'This is evidence for single-variable test selection, not closed-loop counterfactual simulation.',
    ],
    'next_single_variable': {
        'baseline': 'v15, with original pacing enabled, K10 right fingers, source linear damping and bounded ring support',
        'change': 'Only map measured layer feedback into the original opposition_request Core frame before unchanged TurnPacer.next_index',
        'unchanged': ['pacer lead .75deg', 'contact and stall gates', '56D checked targets and collision checks',
                      'all hand gains/materials', 'cube physical parameters', 'raw-relative angular-speed and final acceptance gates'],
        'reason': 'v16 first contact loss follows actual palm-to-layer planning-frame lead of 3.30deg; unlimited advancement develops 11.95deg lead by hold.',
        'expected_limitation': 'Frame correction may still stall; it does not resolve existing selected raw-relative angular speed above .2rad/s on every v15/v16 turn frame.',
        'grip_stiffness_assessment': 'K20 is not the first isolated diagnosis: ring/little normal-direction target errors have already unloaded near zero at first contact loss, while palm/layer phase mismatch is large. This does not prove K20 cannot help.',
        'passive_physics_assessment': 'Keep passive articulation as separately validated physical-model candidate; do not combine it with frame correction in the same attribution run.',
    },
}
path = Path(__file__).with_name('v16_turn_contact_loss_diagnosis.json')
path.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'path': str(path), 'sha256': sha(path), 'same_prefix': report['same_prefix'],
                  'v16_losses': v16['first_right_contact_loss_step'],
                  'v16_relative_to_Core_speed_range': v16['raw_selected_relative_to_Core_speed_min_max_rad_s']}))
for r in v16['selected_steps']:
    if r['step'] in (3174, 3175, 3239, 3240, 3241, 3264, 3325, 3446):
        print(json.dumps({key: value for key, value in r.items() if key != 'right_fingers'}))
        print('FINGERS', json.dumps({f: {key: value for key, value in g.items() if key in ('impulse_Ns', 'anchor_actual_minus_target_current_Core_m', 'tracked_baseline_pair')} for f, g in r['right_fingers'].items()}))
