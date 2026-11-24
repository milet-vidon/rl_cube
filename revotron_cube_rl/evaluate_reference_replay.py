"""Diagnostic replay of recorded applied targets, never a learned-policy result.

Use the original environment step, reset, gains, contacts and acceptance gates.
The default headless mode isolates target approximation against the v3 evaluator;
source render mode also reproduces the original recorded camera/render schedule.
No teacher action is evaluated and no cube state is written after reset.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_target(row):
    target = np.asarray(row['applied_target'], dtype=np.float32)
    if target.shape != (56,) or not np.isfinite(target).all():
        raise ValueError('Reference applied target must contain 56 finite joints')
    return target.copy()


def replay_schedule(rows):
    if any(row['step'] != i for i, row in enumerate(rows)):
        raise ValueError('Reference control steps are not contiguous from zero')
    for row in rows:
        replay_target(row)
    gain_step = next(i for i, row in enumerate(rows) if row['phase'] == 'tilt_hold')
    reference_step = next(i for i, row in enumerate(rows)
                          if row.get('tilt_reference_wxyz') is not None)
    if reference_step < 1:
        raise ValueError('Missing pre-step measured tilt reference')
    reference = np.asarray(rows[reference_step]['tilt_reference_wxyz'])
    previous = np.asarray(rows[reference_step - 1]['cube_quaternion_wxyz'])
    if (reference.shape != (4,) or previous.shape != (4,)
            or not np.isfinite(reference).all() or not np.isfinite(previous).all()
            or not np.allclose(reference, previous, atol=1e-7, rtol=0)):
        raise ValueError('Reference capture schedule differs from original teacher')
    return gain_step, reference_step


def target_error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (actual.shape != (56,) or reference.shape != (56,)
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Invalid emitted/reference target')
    difference = actual - reference
    return {'max_abs_rad': float(np.max(np.abs(difference))),
            'rmse_rad': float(np.sqrt(np.mean(difference ** 2))),
            'worst_joint_index': int(np.argmax(np.abs(difference)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--render-mode', choices=('headless', 'source'), default='headless')
    args = parser.parse_args()

    import torch
    from phase_contact_policy import accepted_prefix
    rows, names, left, source = accepted_prefix(args.source)
    if source['seed'] != args.seed:
        raise ValueError('Replay schedule/reset seed mismatch')
    gain_step, reference_step = replay_schedule(rows)
    manifest = json.loads((args.source / 'artifact_manifest.json').read_text())
    if sha(args.source / 'center_request.json') != manifest['sha256']['center_request.json']:
        raise ValueError('Center coordination request hash mismatch')
    protocol = json.loads((args.source / 'center_request.json').read_text())['coordination_protocol']
    plan = json.loads((args.source / 'center_plan.json').read_text())
    experiment = json.loads((args.source / 'experiment.json').read_text())
    source_hashes = experiment['source_sha256']
    code = Path(__file__).resolve().parent
    required_unchanged = (
        'environment.py', 'articulated_cube_env.py', 'dynamic_cube.py', 'config.json',
        'isaac_compat.py', 'joint_safety.py', 'finger_contact_audit.py',
        'palm_contact_audit.py', 'self_contact_audit.py', 'layer_turn_metrics.py',
        'centered_transport_audit.py', 'center_j56_audit.py', 'wrist_audit.py',
        'palm_handover_audit.py', 'elbow_audit.py', 'tilt_audit.py', 'kinematics.py',
    )
    for name in required_unchanged:
        if sha(code / name) != source_hashes[name]:
            raise ValueError('Original replay environment/audit source changed: ' + name)
    if not experiment['center_only'] or experiment['center_style'] != 'j56':
        raise ValueError('Replay requires the original center-only j56 source')
    friction = tuple(map(float, experiment['friction'].split(',')))
    hand_friction = (tuple(map(float, experiment['hand_friction'].split(',')))
                     if experiment['hand_friction'] else None)
    cameras = args.render_mode == 'source' and bool(experiment['record'])
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    sources = set(source_hashes) | {
        Path(__file__).name, 'phase_contact_policy.py', 'runtime_platform.py',
    }
    evaluation_hashes = {}
    for name in sorted(sources):
        shutil.copy2(code / name, snapshot / name)
        evaluation_hashes[name] = sha(snapshot / name)
    provenance = {
        'scope': 'diagnostic_exact_applied_target_replay_not_neural_policy',
        'seed': args.seed, 'render_mode': args.render_mode, 'cameras': cameras,
        'learned_policy': False, 'left_teacher_targets_used': True,
        'source_manifest_sha256': source['manifest_sha256'],
        'source_trajectory_sha256': sha(args.source / 'trajectory.json'),
        'source_experiment_sha256': sha(args.source / 'experiment.json'),
        'evaluation_source_sha256': evaluation_hashes,
        'source_changes': {name: {'original': digest, 'current': evaluation_hashes[name]}
                           for name, digest in source_hashes.items()
                           if evaluation_hashes[name] != digest},
        'runtime_and_audit_hashes_match': list(required_unchanged),
        'left_finger_hold_gain_step': gain_step,
        'measured_tilt_reference_capture_step': reference_step,
        'target_error_tolerance_rad': 1e-6,
        'target_delivery': 'reference applied targets through unchanged env.step projection',
        'initialization': 'same DynamicPickupEnv construction, reseed and reset as source',
        'audit_subscription_order': ['environment', 'FingerContactAudit', 'DetailedSelfContactAudit'],
        'gain_schedule': 'source arm stiffness/damping 8; fingers K30 D1; left K20 at first tilt_hold',
        'teacher_constructor_audit': 'constructors read env state and build local plans; no simulator writes',
        'stage_gate_unlocked': False,
    }
    (args.output / 'replay_protocol.json').write_text(json.dumps(provenance, indent=2))
    from isaaclab.app import AppLauncher
    from runtime_platform import platform_kit_args
    kit = ('--/app/useFabricSceneDelegate=false '
           '--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false') if args.render_mode == 'source' else ''
    app = AppLauncher(headless=True, enable_cameras=cameras, kit_args=platform_kit_args(kit)).app
    env = None
    try:
        from articulated_cube_env import DynamicPickupEnv
        from isaac_compat import tensor, quaternion_from_native
        from finger_contact_audit import FingerContactAudit
        from palm_contact_audit import DetailedSelfContactAudit
        from palm_handover_audit import measured_hand_frames
        from elbow_audit import measured_elbow
        from centered_transport_audit import assess_center_row
        from center_j56_audit import joint_coordination
        from wrist_audit import assess_wrist
        from layer_turn_metrics import measure_layers
        env = DynamicPickupEnv(args.output, friction=friction, hand_friction=hand_friction,
                               seed=args.seed, vision=cameras, preview=cameras,
                               auto_reset=False, arm_damping=8.)
        env.max_episode_length = int(experiment['max_steps'])
        if env.max_episode_length <= len(rows):
            raise ValueError('Original timeout does not cover the complete prefix')
        env.cfg['success']['hold_seconds'] = 3.
        env.rng = np.random.default_rng(args.seed)
        env.reset()
        env.metrics = {}
        env.pair_impulses = {}
        if env.cfg['joint_names'] != names:
            raise ValueError('Runtime joint schema mismatch')
        for name, actual in (('config.json', env.cfg), ('asset_contract.json', env.contract)):
            (args.output / name).write_text(json.dumps(actual, indent=2))
            if actual != json.loads((args.source / name).read_text()):
                raise ValueError('Runtime ' + name + ' differs from original source')
        fingers = FingerContactAudit()
        audit = DetailedSelfContactAudit()
        reset_count = env.cube.pose_reset_count
        if cameras:
            env.overview.set_world_poses_from_view(torch.tensor([[1.65, 0., 1.45]]),
                                                   torch.tensor([[.40, .06, 1.16]]))
            env._render()
            original_render = env._render
            render_count = [0]

            def sampled_render():
                if render_count[0] % 3 == 0:
                    original_render()
                render_count[0] += 1

            env._render = sampled_render
        initial = {
            'q': tensor(env.robot.data.joint_pos)[0, env.ids].tolist(),
            'joint_velocity': tensor(env.robot.data.joint_vel)[0, env.ids].tolist(),
            'applied_target': env.targets[0].tolist(),
            'cube_root_state_native': tensor(env.cube.data.root_state_w)[0].tolist(),
            'cube_pose_reset_count': reset_count,
        }
        (args.output / 'initial_state.json').write_text(json.dumps(initial, indent=2, allow_nan=False))
        hold = longest = 0
        failure = None
        lifted = returned = False
        maximum_lift = maximum_target_error = 0.
        started = time.monotonic()
        center_initial_q = actual_tilt_reference = precenter_wrist_report = None
        audit_rows = []
        finger_ids = [env.ids[i] for i, name in enumerate(names) if name.startswith('left_')]
        with (args.output / 'trajectory.jsonl').open('w') as journal:
            for step, reference in enumerate(rows):
                if step == gain_step:
                    env.robot.write_joint_stiffness_to_sim_index(
                        stiffness=torch.full((1, len(finger_ids)), 20., dtype=torch.float32),
                        joint_ids=finger_ids)
                if reference['phase'] == 'left_center_transport' and precenter_wrist_report is None:
                    precenter_wrist_report = assess_wrist([
                        row if row['layer_measurement']['assembled_for_grasp']
                        else row | {'phase': 'cube_not_assembled'} for row in audit_rows], names)
                    if not precenter_wrist_report['accepted']:
                        failure = 'precenter_wrist_hold_gate_not_met'
                        break
                if step == reference_step:
                    actual_tilt_reference = quaternion_from_native(
                        tensor(env.cube.data.root_state_w)[0, 3:7]).tolist()
                if reference['phase'] == 'left_center_transport' and center_initial_q is None:
                    center_initial_q = tensor(env.robot.data.joint_pos)[0, env.ids].tolist()
                target = replay_target(reference)
                fingers.reset_step()
                audit.reset_step()
                env.step((torch.from_numpy(target).reshape(1, 56) - env.targets) / env.scales)
                row = dict(step=step, phase=reference['phase'], **env.metrics)
                row.update(fingers.metrics())
                row.update(audit.metrics())
                row.update(nominal_target=target.tolist(), applied_target=env.targets[0].tolist(),
                           q=tensor(env.robot.data.joint_pos)[0, env.ids].tolist(),
                           cube_quaternion_wxyz=quaternion_from_native(
                               tensor(env.cube.data.root_state_w)[0, 3:7]).tolist(),
                           measured_hand_frames=measured_hand_frames(env),
                           measured_elbow=measured_elbow(env),
                           tilt_reference_wxyz=actual_tilt_reference, tilt_degrees=-90,
                           right_contact_axis=0, right_thumb_sign=-1,
                           diagnostic_reference_replay=True)
                row['layer_measurement'] = measure_layers(env.cube.states())
                error = target_error(row['applied_target'], reference['applied_target'])
                error['worst_joint'] = names[error['worst_joint_index']]
                row['reference_target_error'] = error
                row['reference_state_difference'] = {
                    'q_max_abs_rad': float(np.max(np.abs(np.asarray(row['q']) - reference['q']))),
                    'cube_position_delta_m': (np.asarray(row['cube_position'])
                                              - reference['cube_position']).tolist(),
                }
                maximum_target_error = max(maximum_target_error, error['max_abs_rad'])
                audit_rows.append(row)
                if error['max_abs_rad'] > 1e-6:
                    failure = 'reference_applied_target_not_reproduced'
                if (row['self_contact_impulse'] > 1e-6 or row['robot_table_impulse'] > 1e-6
                        or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation']
                        or any(row['hard_limit_hit'])):
                    failure = 'robot_safety_gate'
                maximum_lift = max(maximum_lift, float(row['height_m']))
                lifted = lifted or row['height_m'] >= .08
                if lifted and row['height_m'] < .05:
                    returned = True
                    failure = 'cube_returned_to_table_after_lift'
                if row['drop']:
                    failure = 'cube_drop'
                if env.cube.pose_reset_count != reset_count:
                    raise RuntimeError('Unexpected cube pose write')
                if row['phase'] == 'center_hold':
                    if center_initial_q is None:
                        raise ValueError('Missing measured center initial state')
                    report = assess_center_row(row, names, plan['center_target_world_m'],
                                               plan['cube_target_quaternion_wxyz'])
                    coordination = joint_coordination(row['q'], center_initial_q, names, protocol)
                    hold = hold + 1 if report['qualified'] and coordination['qualified'] and not failure else 0
                    longest = max(longest, hold)
                    row['center_audit'] = report
                    row['joint_coordination'] = coordination
                journal.write(json.dumps(row, allow_nan=False) + '\n')
                if step % 120 == 0 or failure:
                    journal.flush()
                    print('REFERENCE_REPLAY', step, row['phase'], 'hold', hold,
                          'target_max_error', error['max_abs_rad'], 'failure', failure, flush=True)
                if failure:
                    break
        result = provenance | {
            'steps': len(audit_rows), 'source_prefix_steps': len(rows),
            'complete_prefix_replayed': len(audit_rows) == len(rows),
            'precenter_wrist_audit': precenter_wrist_report,
            'accepted': bool(not failure and longest >= 90 and len(audit_rows) == len(rows)),
            'failure': failure or ('center_hold_gate_not_met' if longest < 90 else None),
            'qualified_center_hold_s': longest / 30., 'maximum_lift_m': maximum_lift,
            'maximum_emitted_target_error_rad': maximum_target_error,
            'cube_returned_to_table_after_lift': returned,
            'elapsed_s': time.monotonic() - started,
            'phase_schedule_source': args.source.name, 'right_schedule_seed': source['seed'],
            'right_turn_trained': False,
            'cube_pose_writes_during_control': env.cube.pose_reset_count - reset_count,
            'acceptance_scope': 'diagnostic replay only; never learned-policy acceptance',
        }
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print('RESULT', json.dumps(result), flush=True)
    except Exception as error:
        import traceback
        (args.output / 'error.json').write_text(json.dumps(
            {'error': str(error), 'traceback': traceback.format_exc()}, indent=2))
        raise
    finally:
        from isaac_compat import close_application
        if env is not None:
            env.close()
        close_application(app)


if __name__ == '__main__':
    main()
