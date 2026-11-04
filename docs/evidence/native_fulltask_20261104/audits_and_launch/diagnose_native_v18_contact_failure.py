"""Pure CPU analysis of the completed v18 episode and recorded contact sequence."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_fulltask_v18 import ROOT, WORK, NATIVE, OLD, sha, load, summary, contact_distribution

sys.path.insert(0, str(ROOT))
from tilt_audit import contact_topology


def analyze(run):
    rows = load(run / 'trajectory.json'); names = load(run / 'config.json')['joint_names']
    start = next(r['step'] for r in rows if r['phase'] == 'right_preshape')
    result = load(run / 'result.json')
    ranges = {}; first_right = {}
    for finger in ('thumb', 'index', 'middle', 'ring', 'little'):
        intervals = []
        for row in rows[start:]:
            if not contact_topology(row, 'left')[finger]:
                step = row['step']
                if intervals and intervals[-1][1] + 1 == step: intervals[-1][1] = step
                else: intervals.append([step, step])
        ranges[finger] = intervals
        first_right[finger] = next((r['step'] for r in rows[start:] if r['finger_contact_impulses']['right'][finger] > 1e-7), None)
    index = [i for i, n in enumerate(names) if n.startswith('left_index_')]
    ring = [i for i, n in enumerate(names) if n.startswith('left_ring_')]
    right = [i for i, n in enumerate(names) if n.startswith('right_') or n.endswith('_R')]
    observations = {}
    for label, indices in (('left_index', index), ('left_ring', ring), ('all_right_joints', right)):
        observations[label] = {'maximum_actual_minus_target_abs_rad': max(abs(r['q'][i] - r['applied_target'][i]) for r in rows[start:] for i in indices),
            'maximum_nominal_minus_applied_abs_rad': max(abs(r['nominal_target'][i] - r['applied_target'][i]) for r in rows[start:] for i in indices),
            'last_tracking': {names[i]: {'actual': rows[-1]['q'][i], 'applied_target': rows[-1]['applied_target'][i],
                                      'error': rows[-1]['q'][i] - rows[-1]['applied_target'][i]} for i in indices}}
    steps = {start - 1, start, rows[-1]['step']}
    for intervals in ranges.values():
        for first, last in intervals: steps.update((first - 1, first, last))
    for step in first_right.values():
        if step is not None: steps.update((step - 1, step))
    snapshots = []
    for step in sorted(steps):
        r = rows[step]
        snapshots.append({key: r.get(key) for key in ('step', 'phase', 'cube_position', 'cube_quaternion_wxyz',
            'linear_speed', 'angular_speed', 'finger_contact_impulses', 'left_ring_support', 'index_contact_feedback')} | {
                'left_topology': contact_topology(r, 'left'),
                'Core_relative_face_angle_deg': r['layer_measurement']['face_angle_deg'],
                'selected_relative_omega_rad_s': r['layer_measurement']['maximum_selected_angular_velocity_rad_s'],
                'left_index_and_ring_tracking': {names[i]: {'q': r['q'][i], 'target': r['applied_target'][i],
                    'q_minus_target': r['q'][i] - r['applied_target'][i]} for i in index + ring}})
    displacement = None
    if first_right['little'] is not None:
        reference = rows[first_right['little'] - 1]
        q0 = Rotation.from_quat(np.asarray(reference['cube_quaternion_wxyz'])[[1, 2, 3, 0]])
        q1 = Rotation.from_quat(np.asarray(rows[-1]['cube_quaternion_wxyz'])[[1, 2, 3, 0]])
        displacement = {'reference_step_before_first_right_little_contact': reference['step'],
            'last_step': rows[-1]['step'], 'Core_translation_m': float(np.linalg.norm(np.asarray(rows[-1]['cube_position']) - reference['cube_position'])),
            'Core_rotation_degrees': float(np.degrees((q1 * q0.inv()).magnitude()))}
    return {'run': run.name, 'opposition_start_step': start, 'left_topology_loss_intervals': ranges,
        'first_right_positive_impulse_step': first_right, 'tracking_during_opposition': observations,
        'Core_motion_after_first_right_little_contact': displacement, 'snapshots': snapshots,
        'result': {key: result.get(key) for key in ('steps', 'failure', 'dynamic_grasp_passed', 'center_transport_passed', 'assembled_hold_s', 'task_episode')},
        'artifact_sha256': {n: sha(run / n) for n in ('trajectory.json', 'result.json', 'experiment.json')}}, rows


def main():
    run = ROOT / 'runs' / NATIVE
    native, rows = analyze(run)
    old, _ = analyze(ROOT / 'runs' / OLD)
    props = load(run / 'passive_cube_baseline.json')['rigid_baseline']
    markers = {p.rsplit('/', 1)[-1]: np.asarray(v['local_com'][:3]) for p, v in props.items() if p != '/World/Core'}
    report = {'scope': 'Completed v18 physical-contact sequence diagnosis; read-only CPU; no changed thresholds or simulator run',
        'native': native, 'D6_v17_historical_diagnostic_only': old,
        'left_right_close_contact_distribution_GEOMETRIC_INFERENCE_NOT_ACTOR_OBSERVATION': contact_distribution(rows, markers, phases=('right_close',)),
        'contact_inference_limits': 'Uses actual contact positions and transformed source local COM center markers with .25/1/2mm group-distance sensitivity. Scalar impulse magnitudes do not establish resisting torque. v18 never reached opposition_hold or turn.',
        'causal_scope': [
            'Left index loss starts at 2063, before every right contact; right pressure cannot explain the first index loss.',
            'Only right little first contacts at 2997; left ring pressure declines thereafter and reaches zero at 3007. This temporal association alone does not isolate mechanical causation.',
            'Index and ring are both absent for ten consecutive frames 3007..3016; the unchanged original retained-contact safety rule terminates the task.',
            'The final face angle is drift, not a turn; no opposed_face_turn action was executed.'
        ],
        'minimal_next_candidate_for_parent_review': {
            'change': 'Continue the existing center left-index MPR contact feedback into the opposition stages only; keep its original range, increment and impulse references.',
            'reference': 'Recover un-offset reference as final applied MPR minus the offset actually applied in the last center row; carry the already-computed next_offset. Never add offset twice to fixed target.',
            'freeze': 'Keep native model, right trajectory/gains, ring support, all original acceptance/safety gates, and FullTask terminal contract unchanged.',
            'required_before_run': 'Host continuity/bounds tests plus joint index/ring envelope clearance preflight; then a new from-reset episode.',
            'claim': 'An experiment targeting loss before right contact, not a proven fix or a face-turn success.'},
        'full_task_contract_evidence': 'native_fulltask_v18_final_audit.json: 3017 contiguous audited transitions, one terminal/final, no post-terminal physics, raw_done=0, pickup_milestone absent'}
    path = WORK / 'native_fulltask_v18_contact_failure_diagnosis.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(path), 'sha256': sha(path), 'native_loss_intervals': native['left_topology_loss_intervals'],
        'right_first_contacts': native['first_right_positive_impulse_step'], 'Core_after_little': native['Core_motion_after_first_right_little_contact'],
        'tracking_max_abs_rad': {k: v['maximum_actual_minus_target_abs_rad'] for k, v in native['tracking_during_opposition'].items()}}))


if __name__ == '__main__': main()
