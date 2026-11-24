"""Read-only evidence audit; never imports or launches Isaac."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math
import statistics
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path('/home/brainco/rl_cube/github_project')
ROOT = PROJECT / 'revotron_cube_rl'
WORK = PROJECT / 'work/teacher_optimization_20261103'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
NATIVE = 'teacher_native_fulltask_v18_20261104'
OLD = 'teacher_joint_planning_frame_paced_v17_20261104'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def lines(path, running=False):
    result = []
    with path.open() as stream:
        for line in stream:
            try: result.append(json.loads(line))
            except json.JSONDecodeError:
                if not running: raise
    return result


def summary(values):
    return None if not values else {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values), 'last': values[-1]}


def longest(flags):
    count = best = 0
    for flag in flags:
        count = count + 1 if flag else 0
        best = max(best, count)
    return best


def native_verification(run):
    actual = load(run / 'passive_articulation_verification.json')
    baseline = load(run / 'passive_cube_baseline.json')
    assert sha(run / 'passive_cube_baseline.json') == actual['baseline_sha256']
    assert actual['backend_verified'] is True
    assert len(actual['body_paths']) == 27 and len(actual['dof_names']) == 66
    assert actual['body_paths'][0] == '/World/Core'
    props = actual['measured_mass_properties']
    assert set(props) == set(baseline['rigid_baseline'])
    for path, now in props.items():
        old = baseline['rigid_baseline'][path]
        assert np.isclose(now['mass'], old['mass'], rtol=1e-5)
        assert np.allclose(now['inertia'], old['inertia'], rtol=1e-5, atol=1e-12)
        assert np.allclose(now['local_com'][:3], old['local_com'][:3], rtol=1e-5, atol=1e-7)
        qa, qb = np.asarray(now['local_com'][3:]), np.asarray(old['local_com'][3:])
        assert np.isclose(np.linalg.norm(qa), 1., atol=1e-6)
        assert min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb)) < 1e-5
    backend = actual['backend_readback']
    assert np.asarray(backend['armature']).shape == (1, 66)
    assert np.allclose(backend['armature'], .0001, rtol=1e-6)
    assert np.asarray(backend['friction']).shape == (1, 66, 3)
    assert np.allclose(backend['friction'], [.001, .001, .0005], rtol=1e-6)
    for field in ('drive_stiffness', 'drive_damping', 'position_targets', 'velocity_targets', 'actuation_forces'):
        assert np.isfinite(backend[field]).all() and np.max(np.abs(backend[field])) == 0
    reset = actual['last_reset_readback']
    assert reset['passed'] and reset['all_66_q_and_qd_zero'] and reset['count'] == 2
    assert actual['solver_iterations']['usd_readback'] == [64, 4]
    experiment = load(run / 'experiment.json')
    assert experiment['full_task_episode'] and experiment['passive_cube_articulation']
    assert not experiment['source_core_inertia'] and not experiment['source_joint_damping']
    assert experiment['max_steps'] == 4500
    for name, digest in experiment['source_sha256'].items():
        archive = experiment['source_archive_paths'][name]
        assert sha(run / archive) == digest, 'Source snapshot changed: ' + name
    return {'passed': True, 'body_count': 27, 'DOF_count': 66,
        'all_27_mass_inertia_COM_frames_match_measured_baseline': True,
        'per_DOF_armature': backend['armature'][0][0], 'per_DOF_friction_static_dynamic_linear': backend['friction'][0][0],
        'zero_initial_active_drive_and_effort_readback': True,
        'Core_mass_kg': props['/World/Core']['mass'], 'Core_inertia_kg_m2': props['/World/Core']['inertia'],
        'solver': actual['solver_iterations'], 'reset': reset,
        'snapshot_hashes_match_experiment': True,
        'evidence_sha256': {name: sha(run / name) for name in ('passive_articulation_verification.json', 'passive_cube_baseline.json', 'experiment.json')}}


def task_events(run, running):
    events = lines(run / 'task_episode_transitions.jsonl', running)
    assert events[0]['record_type'] == 'episode_start'
    assert events[0]['maximum_control_steps'] == 4500
    count = 0; raw_done = 0; raw_success = 0; milestones = []; terminal = None; final = None; prefix = []
    prior_raw = events[0]['initial_raw_counters']
    for event in events[1:]:
        assert final is None, 'Event after episode_final'
        if event['record_type'] == 'episode_final':
            final = event
            assert event['task']['done'] and event['task']['task_episodes'] == 1
            assert event['after_completed_steps'] == count
            continue
        assert event['record_type'] == 'transition' and terminal is None, 'Physics transition after task terminal'
        assert event['step'] == count and event['row']['step'] == count
        assert event['phase'] == event['row']['phase']
        assert event['audit']['original_audits_completed']
        raw, task = event['raw'], event['task']
        assert raw['episode_length'] == count + 1
        assert raw['done_returned'] == bool(raw['pickup_success'] or raw['drop'] or raw['derived_timeout'])
        assert raw['derived_timeout'] == (count + 1 >= 4500)
        assert bool(raw['metrics']['terms']['terminal']) == raw['done_returned']
        assert event['raw_counters_before'] == prior_raw
        expected = {'episodes': prior_raw['episodes'] + int(raw['done_returned']),
                    'successes': prior_raw['successes'] + int(raw['pickup_success'] and raw['done_returned'])}
        assert raw['counters_after'] == expected
        prior_raw = expected
        raw_done += int(raw['done_returned']); raw_success += int(raw['pickup_success'])
        if event['pickup_milestone_first_observed']: milestones.append(count)
        assert task['raw_pickup_done_frames'] == raw_done and task['raw_pickup_success_frames'] == raw_success
        assert task['reward'] == 0 and not task['eligible_for_ppo'] and not task['step_audit_pending']
        assert task['completed_control_steps'] == count + 1
        assert task['pickup_milestone_step'] == (milestones[0] if milestones else None)
        if task['done']: terminal = {'step': count, 'phase': event['phase'], 'task': task}
        if raw['drop'] or raw['derived_timeout']: assert task['done']
        prefix.append(event['row'])
        count += 1
    assert len(milestones) <= 1
    if not running:
        assert final is not None, 'Completed run lacks final event'
        result = load(run / 'result.json')
        assert result['task_episode'] == final['task']
        assert result['steps'] == count
    return {'passed': True, 'counts': dict(Counter(e['record_type'] for e in events)),
        'transitions': count, 'raw_done_frames': raw_done, 'raw_pickup_success_frames': raw_success,
        'milestone_steps': milestones, 'first_task_terminal_transition': terminal,
        'final': final['task'] if final else None, 'no_steps_after_task_terminal': True,
        'contiguous_step_audit_and_raw_counter_contract': True}, prefix


def phase_summary(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row['phase'], [])
        group.append(row)
        states = row['cubelets']; selected = row['layer_measurement']['selected_cubelets']
        omega = np.asarray(states['Core']['angular_velocity_rad_s'])
        speed = max(float(np.linalg.norm(np.asarray(states[name]['angular_velocity_rad_s']) - omega)) for name in selected)
        assert np.isclose(speed, row['layer_measurement']['maximum_selected_angular_velocity_rad_s'], atol=1e-10)
        assert row['layer_measurement']['face'] == 'nZ'
        core = Rotation.from_quat(np.asarray(states['Core']['quaternion_wxyz'])[[1, 2, 3, 0]])
        body = Rotation.from_quat(np.asarray([states[name]['quaternion_wxyz'] for name in selected])[:, [1, 2, 3, 0]])
        relative = (core.inv() * body).as_quat()
        relative[relative[:, 3] < 0] *= -1
        angle = float(np.median(np.degrees(2 * np.arctan2(-relative[:, 2], relative[:, 3]))))
        assert np.isclose(angle, row['layer_measurement']['face_angle_deg'], atol=1e-8)
    result = {}
    for phase, group in groups.items():
        speed = [r['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for r in group]
        result[phase] = {'frames': len(group), 'first_step': group[0]['step'], 'last_step': group[-1]['step'],
            'height_m': summary([r['height_m'] for r in group]),
            'Core_absolute_angular_speed_rad_s': summary([float(np.linalg.norm(r['cubelets']['Core']['angular_velocity_rad_s'])) for r in group]),
            'selected_relative_angular_speed_rad_s': summary(speed),
            'selected_relative_speed_ge_point2_frames': sum(x >= .2 for x in speed),
            'true_Core_relative_face_angle_deg': summary([r['layer_measurement']['face_angle_deg'] for r in group]),
            'maximum_nonselected_rotation_deg': max(r['layer_measurement']['maximum_other_cubelet_rotation_deg'] for r in group),
            'maximum_joint_origin_separation_m': max(r['layer_measurement']['maximum_joint_origin_separation_m'] for r in group),
            'left_positive_impulse_frames': {f: sum(r['finger_contact_impulses']['left'][f] > 1e-7 for r in group) for f in FINGERS},
            'left_all_five_positive_frames': sum(all(r['finger_contact_impulses']['left'][f] > 1e-7 for f in FINGERS) for r in group),
            'right_target_all_five_frames': sum(r.get('opposition_audit', {}).get('right_contacts', {}).get('all_five', False) for r in group)}
    return result


def contact_distribution(rows, local_markers, phases=('opposition_hold', 'opposed_face_turn', 'opposed_face_hold')):
    """Geometric inference only: original left-contact records omit actor IDs."""
    answer = {}
    for phase in phases:
        group = [r for r in rows if r['phase'] == phase]
        if not group: continue
        stats = {f: {'selected_impulse_Ns': 0., 'other_impulse_Ns': 0., 'ambiguous_impulse_Ns': 0.,
            'total_impulse_Ns': 0., 'selected_frames': [], 'other_frames': [], 'ambiguous_frames': [],
            'contact_points': 0, 'nearest_marker_distance_m': [], 'selected_vs_other_distance_gap_m': [],
            'nearest_name_impulse_Ns': {}} for f in FINGERS}
        sensitivity = {str(gap): {f: {'selected': 0., 'other': 0., 'ambiguous': 0.} for f in FINGERS}
                       for gap in (.00025, .001, .002)}
        for row in group:
            centers = {}
            for name, state in row['cubelets'].items():
                if name == 'Core': continue
                rotation = Rotation.from_quat(np.asarray(state['quaternion_wxyz'])[[1, 2, 3, 0]])
                centers[name] = np.asarray(state['position_m']) + rotation.apply(local_markers[name])
            selected = set(row['layer_measurement']['selected_cubelets'])
            frame = {f: set() for f in FINGERS}
            for point in row['finger_contact_points']:
                if point['side'] != 'left' or point['impulse_Ns'] <= 1e-7: continue
                finger, impulse = point['finger'], point['impulse_Ns']
                position = np.asarray(point['position_world_m'])
                distances = {name: float(np.linalg.norm(position - center)) for name, center in centers.items()}
                nearest = min(distances, key=distances.get)
                target_d = min(value for name, value in distances.items() if name in selected)
                other_d = min(value for name, value in distances.items() if name not in selected)
                gap = abs(target_d - other_d)
                bucket = 'selected' if target_d < other_d else 'other'
                # These are local geometric center markers, not actor origins.
                # The .018m radius only rejects implausible outer cubie matches.
                for margin in (.00025, .001, .002):
                    classification = bucket if gap > margin and distances[nearest] < .018 else 'ambiguous'
                    sensitivity[str(margin)][finger][classification] += impulse
                classification = bucket if gap > .001 and distances[nearest] < .018 else 'ambiguous'
                s = stats[finger]; s[classification + '_impulse_Ns'] += impulse; s['total_impulse_Ns'] += impulse
                s['contact_points'] += 1; s['nearest_marker_distance_m'].append(distances[nearest])
                s['selected_vs_other_distance_gap_m'].append(gap)
                s['nearest_name_impulse_Ns'][nearest] = s['nearest_name_impulse_Ns'].get(nearest, 0.) + impulse
                frame[finger].add(classification)
            for finger in FINGERS:
                for classification in ('selected', 'other', 'ambiguous'):
                    stats[finger][classification + '_frames'].append(classification in frame[finger])
        for finger, s in stats.items():
            denominator = s['total_impulse_Ns']
            s['impulse_fraction'] = {k: s[k + '_impulse_Ns'] / denominator if denominator else None for k in ('selected', 'other', 'ambiguous')}
            for classification in ('selected', 'other', 'ambiguous'):
                flags = s.pop(classification + '_frames')
                s[classification + '_contact_frames'] = sum(flags)
                s[classification + '_longest_contiguous_frames'] = longest(flags)
            for field in ('nearest_marker_distance_m', 'selected_vs_other_distance_gap_m'):
                s[field] = summary(s[field])
        answer[phase] = {'frames': len(group), 'by_finger': stats, 'gap_sensitivity_impulse_Ns': sensitivity}
    return answer


def turn_diagnosis(run, rows):
    turns = [row for row in rows if row['phase'] in ('opposed_face_turn', 'opposed_face_hold')]
    if not turns: return {'turn_frames': 0, 'scope': 'No turn action has been observed in this prefix'}
    assert sha(ROOT / 'kinematics.py') == sha(run / 'kinematics.py')
    assert sha(ROOT / 'tilt_audit.py') == sha(run / 'tilt_audit.py')
    sys.path.insert(0, str(ROOT))
    from kinematics import RobotKinematics
    from tilt_audit import contact_topology
    plan, request = load(run / 'opposition_plan.json'), load(run / 'opposition_request.json')
    close = next(path[-1] for phase, path in plan['phases'] if phase == 'right_close')
    reference = RobotKinematics().fk_link(dict(zip(request['joint_names'], close)), 'right_palm_link')[:3, :3]
    initial_core = Rotation.from_quat(np.asarray(request['cube_quaternion'])[[1, 2, 3, 0]]).as_matrix()
    measurements = []; failures = Counter()
    for row in turns:
        palm = np.asarray(row['measured_hand_frames']['right_palm_link'])[:3, :3]
        current_core = Rotation.from_quat(np.asarray(row['cube_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        angle = row['layer_measurement']['face_angle_deg']
        mapped = -float(np.degrees(Rotation.from_matrix(initial_core.T @ current_core @ Rotation.from_euler('z', -angle, degrees=True).as_matrix()).as_rotvec()[2]))
        actual = -float(np.degrees(Rotation.from_matrix(initial_core.T @ palm @ reference.T @ initial_core).as_rotvec()[2]))
        paced = row.get('turn_pacing') or {}
        planned = paced.get('planned_palm_angle_deg')
        if planned is None:
            planned = plan['turn_path_angles_deg'][-1] if row['phase'] == 'opposed_face_hold' else plan['turn_path_angles_deg'][row['step'] - turns[0]['step']]
        layer = row['layer_measurement']; report = row['opposition_audit']
        checks = {
            'height_and_no_drop_or_table': row['height_m'] >= .08 and not row['cube_table_contact'] and not row['drop'],
            'Core_linear_speed_below_point03': row['linear_speed'] < .03,
            'Core_angular_speed_below_point2': row['angular_speed'] < .20,
            'layer_target_geometry': layer['geometry_qualified'],
            'selected_relative_speed_below_point2': layer['maximum_selected_angular_velocity_rad_s'] < .20,
            'robot_safety': row['self_contact_impulse'] <= 1e-6 and row['robot_table_impulse'] <= 1e-6 and not (row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])),
            'posture': report['posture']['qualified'], 'right_all_five_target_contacts': report['right_contacts']['all_five'],
            'no_wrong_right_cubelet_contacts': not report['right_contacts']['unexpected_contacts'],
            'left_all_five_topology_contacts': all(contact_topology(row, 'left').values())}
        assert bool(all(checks.values())) == report['qualified']
        failures.update(key for key, value in checks.items() if not value)
        measurements.append({'step': row['step'], 'phase': row['phase'], 'path_index': paced.get('path_index'),
            'held_ticks': paced.get('held_ticks'), 'planned_palm_angle_deg': planned,
            'actual_palm_angle_in_planner_frame_deg': actual,
            'actual_Core_relative_face_angle_deg': angle, 'layer_angle_in_planner_frame_deg': mapped,
            'actual_palm_minus_layer_deg': actual - mapped, 'planned_palm_minus_layer_deg': planned - mapped,
            'planned_minus_actual_palm_deg': planned - actual,
            'selected_raw_relative_angular_speed_rad_s': layer['maximum_selected_angular_velocity_rad_s'],
            'original_gate_checks': checks})
    first_stall = next((row for row in measurements if (row['held_ticks'] or 0) > 0), None)
    return {'turn_frames': len(turns), 'original_gate_false_frame_counts': dict(failures),
        'recomputed_original_qualification_matches_all_frames': True,
        'measures': {key: summary([row[key] for row in measurements]) for key in (
            'planned_palm_angle_deg', 'actual_palm_angle_in_planner_frame_deg', 'actual_Core_relative_face_angle_deg',
            'layer_angle_in_planner_frame_deg', 'actual_palm_minus_layer_deg', 'planned_palm_minus_layer_deg',
            'planned_minus_actual_palm_deg', 'selected_raw_relative_angular_speed_rad_s')},
        'first': measurements[0], 'first_pacer_hold': first_stall, 'last': measurements[-1],
        'selected_snapshots': measurements[::20],
        'interpretation': 'Actual and planned palm are compared against the original plan right-close reference in initial-Core coordinates. Layer mapping reconstructs median twist and cannot resolve off-axis dispersion. No torque is inferred.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--allow-running', action='store_true'); args = parser.parse_args()
    run = ROOT / 'runs' / NATIVE
    running = not (run / 'result.json').exists()
    if running and not args.allow_running: raise RuntimeError('Final result not written yet')
    verification = native_verification(run)
    events, current = task_events(run, running)
    baseline = load(run / 'passive_cube_baseline.json')
    markers = {path.rsplit('/', 1)[-1]: np.asarray(record['local_com'][:3])
               for path, record in baseline['rigid_baseline'].items() if path != '/World/Core'}
    old_run = ROOT / 'runs' / OLD; old_rows = load(old_run / 'trajectory.json')
    assert load(old_run / 'cube_asset.json')['source_sha256'] == load(run / 'cube_asset.json')['source_sha256']
    old_result = load(old_run / 'result.json')
    first_terminal = next((r['step'] for r in old_rows if bool(r['terms']['terminal'])), None)
    old_scope = {'scope': 'historical physical continuation diagnostic only; not a valid FullTask episode',
        'first_original_terminal_step': first_terminal,
        'post_original_terminal_frames': len(old_rows) - first_terminal - 1 if first_terminal is not None else 0}
    result = {'scope': 'read-only native FullTask evidence audit; no simulation or source changes',
        'running_prefix_only': running, 'native_verification': verification, 'full_task_events': events,
        'native': {'run': NATIVE, 'phases': phase_summary(current), 'result': load(run / 'result.json') if not running else None,
                   'left_contact_distribution_geometric_inference': contact_distribution(current, markers),
                   'turn_diagnosis': turn_diagnosis(run, current)},
        'D6_v17_diagnostic_only': {'run': OLD, 'episode_scope': old_scope, 'phases': phase_summary(old_rows),
            'result_subset': {k: old_result[k] for k in ('steps', 'failure', 'dynamic_grasp_passed', 'center_transport_passed', 'assembled_hold_s')},
            'left_contact_distribution_geometric_inference': contact_distribution(old_rows, markers),
            'turn_diagnosis': turn_diagnosis(old_run, old_rows)},
        'contact_inference_method': {
            'direct_left_cubelet_actor_identity_recorded': False,
            'method': 'World contact point nearest actual cubelet geometric center marker; each source local COM marker transformed by its actual actor pose. Compare nearest selected9 versus other17 external cubelets. Core is the 18th remaining body but has no collider.',
            'not_actor_origin_as_center': True,
            'nominal_selected_vs_other_distance_margin_m': .001,
            'maximum_marker_distance_m': .018,
            'sensitivity_margins_m': [.00025, .001, .002],
            'impulses': 'Scalar positive impulse magnitudes accumulated across contact callback points, not force or signed tangential impulse',
            'ambiguity': 'Points close to a selected/other nearest-center boundary or too far from all markers stay ambiguous',
            'limitations': 'Geometric association is estimated, not recovered actor identity. Contact distribution alone does not establish resisting torque; tangential impulse vectors were not recorded.'},
        'evidence_sha256': {'v17_trajectory': sha(old_run / 'trajectory.json'), 'v17_result': sha(old_run / 'result.json')},
        'no_general_six_face_or_solution_claim': True}
    if not running:
        result['evidence_sha256'].update({name: sha(run / name) for name in ('trajectory.json', 'result.json', 'task_episode_transitions.jsonl')})
    destination = WORK / ('native_fulltask_v18_prefix_audit.json' if running else 'native_fulltask_v18_final_audit.json')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('AUDIT', str(destination), sha(destination))
    print('EVENTS', json.dumps(events))
    for label in ('native', 'D6_v17_diagnostic_only'):
        print('PHASES', label, json.dumps(result[label]['phases']))
        for phase, data in result[label]['left_contact_distribution_geometric_inference'].items():
            print('LEFT_CONTACT', label, phase, json.dumps({f: {'fraction': d['impulse_fraction'], 'selected_frames': d['selected_contact_frames'], 'other_frames': d['other_contact_frames'], 'ambiguous_frames': d['ambiguous_contact_frames']} for f, d in data['by_finger'].items()}))


if __name__ == '__main__': main()
