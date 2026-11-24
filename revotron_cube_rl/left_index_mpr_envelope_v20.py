"""V20 scalar MPR range, checked with every unchanged original distance predicate.

The full sampled Cartesian index/ring envelope is covered by factoring pairwise
predicates: each right path sample is checked against each index value and each
ring value; all index/ring pairs are also checked statically. Other left joints
and the torso are invariant. Rotating layer theta and all original path samples
remain in both dynamic finger sweeps. This is sampled geometric preflight, not a
continuous-motion proof or physical validation.
"""
import hashlib
import time

from left_index_continuation_v20 import SETTINGS, SCHEMA, validate_state, index_envelope_values
from left_ring_support import JOINT, envelope_values, parameters
from left_support_envelope_v3 import _check_shard

INDEX_JOINT = SETTINGS['joint']
# Frozen implementations that define the imported predicates / scene semantics.
PREDICATE_SHA256 = {
    'left_support_envelope_v3.py': '4fd2602d5b87abbe5f081c6b52c9dfdb6f7681096b30f921bfdc091f406ff3c7',
    'opposed_distance_cache_v1.py': '6c9e269437cc0a7f8b55552115fc903df6037b2ff804807c63b430697960e55e',
}
CONTRACT_SHA256 = '9cca4bd8fa175aa984d510fa2e6b43c1f11f4580c81a65a5c42e5f28b915d971'


def verify_predicate_sources():
    import json
    from pathlib import Path
    import left_support_envelope_v3
    import opposed_distance_cache_v1
    modules = (left_support_envelope_v3, opposed_distance_cache_v1)
    result = {}
    for module in modules:
        path = Path(module.__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PREDICATE_SHA256[path.name]:
            raise ValueError('Frozen geometric predicate changed: ' + path.name)
        result[path.name] = digest
    contract_path = Path(__file__).with_name('left_index_mpr_preflight_contract_v20.json')
    raw = contract_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError('Geometry dependency contract changed')
    contract = json.loads(raw)
    if contract.get('schema') != 'frozen_v19_pairwise_geometry_dependencies_for_v20':
        raise ValueError('Unknown geometry contract')
    project = Path(left_support_envelope_v3.__file__).resolve().parent.parent
    for category in ('source_sha256', 'asset_sha256'):
        for relative, expected in contract[category].items():
            path = (project / relative).resolve()
            if not path.is_relative_to(project) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen geometry dependency changed: ' + relative)
    return dict(direct_predicate_sha256=result, contract_sha256=CONTRACT_SHA256,
                source_sha256=contract['source_sha256'], asset_sha256=contract['asset_sha256'])


def build_jobs(request, plan):
    """Same <=.01 rad path subdivision and theta reconstruction as frozen V19."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from kinematics import RobotKinematics
    if not plan.get('passed'):
        raise ValueError('Cannot validate a rejected plan')
    names, initial = request['joint_names'], request['initial']
    k = RobotKinematics()
    C = Rotation.from_quat(np.asarray(request['cube_quaternion'])[[1, 2, 3, 0]]).as_matrix()
    close = next(path[-1] for phase, path in plan['phases'] if phase == 'right_close')
    reference = k.fk_link(dict(zip(names, close)), 'right_palm_link')[:3, :3]
    previous = np.asarray([initial[n] for n in names])
    jobs = []
    for phase, path in plan['phases']:
        for step, point in enumerate(path):
            point = np.asarray(point)
            if point.shape != (len(names),) or not np.isfinite(point).all():
                raise ValueError('Invalid path target')
            for fraction in np.linspace(0, 1, max(1, int(np.ceil(np.max(np.abs(point - previous)) / .01))) + 1)[1:]:
                state = dict(zip(names, (1 - fraction) * previous + fraction * point))
                if any(abs(state[n] - initial[n]) > 1e-10 for n in names if n.startswith('left_') or n.endswith('_L')):
                    raise ValueError('Factoring requires the original left-hand path to be fixed')
                theta = 0.
                if phase in ('opposed_face_turn', 'opposed_face_hold'):
                    p = k.fk_link(state, 'right_palm_link')[:3, :3]
                    theta = float(-Rotation.from_matrix(C.T @ p @ reference.T @ C).as_rotvec()[2] * 180 / np.pi)
                    if phase == 'opposed_face_turn' and fraction == 1.:
                        angles = plan.get('turn_path_angles_deg', [])
                        if len(angles) != len(path) or abs(theta - angles[step]) > 1e-6:
                            raise ValueError('Turn reference changed')
                jobs.append((phase, step, state, theta))
            previous = point
    if not jobs:
        raise ValueError('Empty path')
    return jobs


def factored_values(index_values, ring_values, index_baseline, ring_baseline):
    if index_baseline not in index_values or ring_baseline not in ring_values:
        raise ValueError('Applied baseline must be explicitly sampled')
    # No final predicate is replaced by a bound or proxy. Only repeated checks
    # of predicates independent of one of these two coordinates are omitted.
    dynamic = sorted(set([(i, ring_baseline) for i in index_values] +
                         [(index_baseline, r) for r in ring_values]))
    static = [(i, r) for i in index_values for r in ring_values]
    return dynamic, static


def measured_worker(payload):
    import resource
    start = time.monotonic()
    result = _check_shard(payload)
    result['worker_elapsed_s'] = time.monotonic() - start
    result['worker_peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return result


def validate_plan(request, plan, baseline_target, index_state, workers=12):
    start = time.monotonic()
    hashes = verify_predicate_sources()
    validate_state(index_state)
    names, initial = request['joint_names'], request['initial']
    baseline = dict(zip(names, baseline_target)) if not isinstance(baseline_target, dict) else dict(baseline_target)
    if baseline[INDEX_JOINT] != index_state['last_applied_target_rad']:
        raise ValueError('Handover belongs to another applied target')
    ring_values = envelope_values(initial[JOINT], baseline[JOINT])
    index_values = sorted(set(index_envelope_values(initial[INDEX_JOINT], index_state) + [baseline[INDEX_JOINT]]))
    dynamic, static = factored_values(index_values, ring_values, baseline[INDEX_JOINT], baseline[JOINT])
    jobs = build_jobs(request, plan)
    # Includes every index x ring self-collision / fixed-cube combination.
    # Other finger-vs-right predicates are separately checked at every job.
    static_result = measured_worker((request, jobs[:1], static, baseline))
    workers = max(1, min(int(workers), 12, len(jobs)))
    results = []
    if static_result['passed']:
        payloads = [(request, jobs[i::workers], dynamic, baseline) for i in range(workers)]
        if workers == 1:
            results = [measured_worker(payloads[0])]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(measured_worker, payloads))
    failed = next((r for r in [static_result] + results if not r['passed']), None)
    if verify_predicate_sources() != hashes:
        raise ValueError('Geometric sources changed during preflight')
    return dict(schema='factorized_pairwise_full_path_index_mpr_v20', passed=failed is None,
        parameters=parameters(), index_parameters=dict(SETTINGS), index_continuation=index_state,
        absolute_index_samples_rad=index_values, absolute_ring_samples_rad=ring_values,
        path_samples=len(jobs), full_cartesian_equivalent_checks=len(jobs) * len(static),
        dynamic_coordinate_pairs_per_job=len(dynamic), static_joint_combinations=len(static),
        checks=static_result['checks'] + sum(r['checks'] for r in results),
        failure=failed, workers=workers, elapsed_s=time.monotonic() - start,
        maximum_worker_peak_RSS_MiB=max(r['worker_peak_RSS_MiB'] for r in [static_result] + results),
        predicate_sha256=hashes, static_joint_result=static_result, shard_results=results,
        coverage_contract={'all_original_right_path_samples_and_theta': True,
            'maximum_support_sample_step_rad': .001, 'maximum_right_path_joint_step_rad': .01,
            'right_vs_each_index_and_each_ring': 'every original sampled path and layer theta',
            'index_vs_ring': 'all sampled Cartesian combinations; both fixed relative to robot root',
            'changed_finger_vs_other_left_torso_table_limits_stationary_cube': 'each sampled coordinate through original predicates',
            'changed_finger_vs_rotating_target_layer': 'each sampled coordinate at every path theta',
            'original_acceptance_and_collision_thresholds_unchanged': True,
            'continuous_interval_collision_proof': False}, physics_validated=False)
