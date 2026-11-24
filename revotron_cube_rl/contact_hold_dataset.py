"""Audited successful pickup-to-wrist demonstrations for a new state BC policy.

This reader does not promote V11 evaluation/DAgger failure prefixes. It requires
a complete accepted teacher episode, verified bytes, and recomputed original
physical gates. All sources remain development data, never independent tests.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from phase_contact_policy import FEATURES, PHASES, features

SCHEMA = 'contact_hold_success_demo_state_bc_v1'
PICKUP_PHASES = PHASES[:10]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_json(run, name, manifest):
    if sha(run / name) != manifest['sha256'][name]:
        raise ValueError('Evidence hash mismatch: ' + name)
    return json.loads((run / name).read_text())


def audit_demo(run):
    from palm_handover_audit import assess_handover
    from wrist_audit import assess_wrist
    run = Path(run).resolve()
    manifest = json.loads((run / 'artifact_manifest.json').read_text())
    result = verified_json(run, 'result.json', manifest)
    if not (result.get('full_episode') is True and result.get('dynamic_grasp_passed') is True
            and result.get('execution_interrupted') is False and result.get('failure') is None
            and result.get('neural_policy_trained') is False
            and result.get('cube_pose_writes_during_control') == 0):
        raise ValueError('A complete accepted teacher demonstration is required')
    cfg = verified_json(run, 'config.json', manifest)
    contract = verified_json(run, 'asset_contract.json', manifest)
    rows = verified_json(run, 'trajectory.json', manifest)
    if len(rows) != result['steps'] or [r['step'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete or noncontiguous trajectory')
    if not rows or rows[-1]['phase'] != 'tilt_hold' or any(r['phase'] not in PICKUP_PHASES for r in rows):
        raise ValueError('Expected a complete pickup-to-wrist episode without a later task')
    names = cfg['joint_names']
    indices = [i for i, n in enumerate(names) if n.endswith('_L') or n.startswith('left_')]
    if len(names) != 56 or len(set(names)) != 56 or len(indices) != 28:
        raise ValueError('Unexpected 56/28 joint layout')
    for row in rows:
        if (row['self_contact_impulse'] > 1e-6 or row['robot_table_impulse'] > 1e-6 or row['drop']
                or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit'])):
            raise ValueError('Unsafe demonstration transition')
    if len({r['pose_reset_count'] for r in rows}) != 1:
        raise ValueError('Cube pose changed through a reset')
    measured = assess_handover(rows, names, regrasp=False, full_episode=True)
    wrist = assess_wrist([r if r['layer_measurement']['assembled_for_grasp'] else r | {'phase': 'cube_not_assembled'} for r in rows], names)
    if not measured['accepted'] or not wrist['accepted']:
        raise ValueError('Recorded success does not reproduce original physical gates')
    candidate = verified_json(run, 'contact_hold_candidate.json', manifest)
    for name, digest in candidate['source_sha256'].items():
        if Path(name).name != name or sha(run / 'contact_hold_source' / name) != digest:
            raise ValueError('Candidate controller snapshot changed')
    # The original harness archives its own controller/audit dependencies.
    experiment = verified_json(run, 'experiment.json', manifest)
    evidence = ['result.json', 'config.json', 'asset_contract.json', 'trajectory.json',
                'experiment.json', 'contact_hold_candidate.json']
    iterations = experiment.get('position_iterations')
    if iterations != candidate.get('position_iterations'):
        raise ValueError('Teacher and experiment solver settings differ')
    if iterations is not None:
        solver = verified_json(run, 'solver_verification.json', manifest)
        scene = solver['scene']
        if not (solver['verified'] is True and solver['physics_hz'] == 240 and solver['control_hz'] == 30
                and scene['solverType'] == 'TGS' and scene['minPositionIterationCount'] == iterations
                and scene['minVelocityIterationCount'] == 4):
            raise ValueError('Solver readback differs from the teacher experiment')
        evidence.append('solver_verification.json')
    for name, digest in experiment['source_sha256'].items():
        archived = experiment['source_archive_paths'][name]
        if Path(archived).name != archived or sha(run / archived) != digest:
            raise ValueError('Original experiment snapshot changed: ' + name)
    return rows, names, indices, contract, {
        'path': str(run), 'seed': result['seed'], 'rows': len(rows),
        'manifest_sha256': sha(run / 'artifact_manifest.json'),
        'evidence_sha256': {name: manifest['sha256'][name] for name in evidence},
        'candidate': candidate, 'recomputed_wrist_hold_s': wrist['longest_jointly_qualified_hold_s'],
        'scope': 'complete successful teacher pickup-to-wrist development episode',
    }


def training_pairs(rows, indices):
    xs, ys, validation = [], [], []
    phase_start = 0
    for i, current in enumerate(rows):
        if i == 0 or current['phase'] != rows[i-1]['phase']:
            phase_start = i
        if i == 0:
            continue  # Initial reset observation was not in historical post-step logs.
        previous = rows[i-1]
        elapsed = min((i - phase_start) / 300., 1.)
        x = features(previous, current['phase'], elapsed)
        before = np.asarray(previous['applied_target'], dtype=np.float32)
        target = np.asarray(current['applied_target'], dtype=np.float32)
        if before.shape != (56,) or target.shape != (56,) or not np.isfinite(target).all():
            raise ValueError('Invalid applied target')
        if np.max(np.abs(target-before)) > .04001:
            raise ValueError('Applied motion exceeds original maximum step')
        xs.append(x)
        ys.append(target[indices])
        validation.append(((i-phase_start)//30) % 5 == 4)
    return np.asarray(xs, dtype=np.float32).reshape(-1, FEATURES), np.asarray(ys, dtype=np.float32), np.asarray(validation)
