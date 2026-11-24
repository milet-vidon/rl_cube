"""CUDA float64 adaptive basis fitting; all-frame compression, NO holdout claim."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from phase_basis_policy_v4 import PHASES, SCHEMA, PhaseBasisPolicy, accepted_prefix, phase_progress, features, sha, basis


def fit_head(t, y, tolerance, maximum_iterations):
    # Six polynomial features represent the smooth schedule; learned ReLU
    # knots are introduced at training residuals to represent feedback jumps.
    indices = set()
    history = []
    best = None
    for iteration in range(maximum_iterations):
        knots = t[sorted(indices)] if indices else t[:0]
        design = basis(t, knots)
        # Explicit tiny ridge provides full column rank to CUDA's gels solver.
        ridge = 1e-20
        augmented = torch.cat((design, torch.eye(design.shape[1], dtype=t.dtype, device=t.device) * ridge**.5))
        rhs = torch.cat((y, torch.zeros((design.shape[1], y.shape[1]), dtype=y.dtype, device=y.device)))
        coefficients = torch.linalg.lstsq(augmented, rhs).solution
        error = (design @ coefficients - y).abs()
        maximum = float(error.max())
        history.append({'iteration': iteration, 'knots': len(indices), 'all_frame_max_error_rad': maximum,
                        'all_frame_rmse_rad': float(error.square().mean().sqrt())})
        if best is None or maximum < best[0]:
            best = (maximum, knots.clone(), coefficients.clone())
        if maximum <= tolerance:
            break
        # Add residual locations and their immediate neighbors. All are training
        # rows; this adaptive choice is never described as held-out validation.
        ranked = torch.argsort(error.max(dim=1).values, descending=True).tolist()
        new = set()
        for index in ranked:
            for candidate in (index-1, index, index+1):
                if 0 < candidate < len(t)-1 and candidate not in indices:
                    new.add(candidate)
            if len(new) >= 8:
                break
        if not new:
            break
        indices.update(new)
    return best[1], best[2], history, best[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=2e-7)
    p.add_argument('--max-iterations', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.source.name != 'center_j56_dev_v3_20261103':
        raise ValueError('Only the accepted baseline source is authorized for v4')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this offline experiment')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    rows, names, left, provenance = accepted_prefix(a.source)
    progress = phase_progress(rows)
    # Frame 0 is included explicitly. Its state fields are ignored by the model.
    x = np.asarray([features(rows[max(0, i-1)], row['phase'], progress[i]) for i, row in enumerate(rows)])
    y = np.asarray([row['applied_target'] for row in rows], dtype=np.float32)[:, left]
    phase = np.asarray([PHASES.index(row['phase']) for row in rows])
    from kinematics import RobotKinematics
    kinematics = RobotKinematics()
    bounds = np.asarray([kinematics.limits[names[i]] for i in left], dtype=np.float64)
    contract = json.loads((a.source/'asset_contract.json').read_text())['joint_limit_safety']['soft_limits_rad']
    for j, i in enumerate(left):
        if names[i] in contract:
            bounds[j, 0] = max(bounds[j, 0], contract[names[i]][0])
            bounds[j, 1] = min(bounds[j, 1], contract[names[i]][1])
    if np.any(y < bounds[:, 0]-1e-7) or np.any(y > bounds[:, 1]+1e-7):
        raise ValueError('Source target outside preserved limits')
    files = ('phase_contact_policy.py', 'phase_basis_policy_v4.py', 'train_phase_basis_v4.py',
             'evaluate_phase_basis_v4.py', 'test_phase_basis_v4.py', 'centered_transport_audit.py',
             'tilt_audit.py', 'wrist_audit.py', 'elbow_audit.py', 'kinematics.py')
    source_hashes = {name: sha(Path(__file__).parent/name) for name in files}
    meta = {'schema': SCHEMA, 'scope': 'single_source_all_discrete_frame_neural_trajectory_compression',
            'validation_kind': 'none_all_frames_used_for_fit_and_adaptive_knot_selection',
            'purpose': 'action_precision_diagnostic_only_no_robustness_or_generalization_claim',
            'sources': [provenance], 'unique_source_seeds': [provenance['seed']], 'joint_names': names,
            'output_indices': left, 'training_samples': len(rows), 'validation_samples': 0,
            'includes_frame_zero': True, 'right_turn_trained': False, 'closed_loop_validated': False,
            'stage_gate_unlocked': False, 'source_sha256': source_hashes,
            'max_target_increment_rad': .04, 'measured_feedback_bound_rad': 0.,
            'device': 'cuda', 'fit_dtype': 'float64', 'inference_progress_dtype': 'float32_then_float64',
            'tolerance_rad': a.tolerance, 'max_iterations_per_phase': a.max_iterations,
            'ridge': 1e-20, 'memory_fraction_cap': .25,
            'runtime_teacher_target_lookup': False, 'runtime_reference_left_state_used': False}
    heads, histories = [], []
    elapsed = 0.
    if a.resume:
        saved = torch.load(a.output/'last.pt', map_location='cuda', weights_only=True)
        if saved['metadata'] != meta:
            raise ValueError('Resume provenance/config mismatch')
        heads = saved['fitted_heads']
        histories = saved['fit_histories']
        elapsed = saved['elapsed_s']
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        snapshot = a.output/'source_snapshot'
        snapshot.mkdir()
        for name in files:
            shutil.copy2(Path(__file__).parent/name, snapshot/name)
        (a.output/'experiment.json').write_text(json.dumps(meta, indent=2))
    started = time.monotonic()
    targets = torch.as_tensor(y, dtype=torch.float64, device='cuda')
    inputs = torch.as_tensor(x, device='cuda')

    def save(name, extra):
        payload = {'metadata': meta, 'fitted_heads': [(k.cpu(), w.cpu()) for k, w in heads],
                   'fit_histories': histories, 'elapsed_s': elapsed+time.monotonic()-started, **extra}
        tmp = a.output/(name+'.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, a.output/name)

    with (a.output/'metrics.jsonl').open('a') as log:
        for j in range(len(heads), len(PHASES)):
            selected = phase == j
            knots, weights, history, error = fit_head(inputs[selected, -1].double(), targets[selected],
                                                      a.tolerance, a.max_iterations)
            heads.append((knots, weights))
            histories.append(history)
            record = {'phase': PHASES[j], 'samples': int(selected.sum()), 'knots': len(knots),
                      'all_frame_max_error_rad_before_bounds': error, 'iterations': len(history),
                      'elapsed_s': elapsed+time.monotonic()-started}
            log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
            save('last.pt', {'completed_phases': len(heads)})
    model = PhaseBasisPolicy(bounds[:, 0], bounds[:, 1], heads).cuda().eval()
    with torch.no_grad():
        predictions = model(inputs)
        # Evaluate the actual downstream float32 command representation too.
        error = (predictions.float().double()-targets).abs()
        reports = {}
        for j, name in enumerate(PHASES):
            selected = phase == j
            e = error[selected]
            worst = int(e.argmax())
            local_row, joint = divmod(worst, 28)
            reports[name] = {'samples': int(selected.sum()), 'max_error_rad': float(e.max()),
                             'rmse_rad': float(e.square().mean().sqrt()),
                             'worst_step': int(np.flatnonzero(selected)[local_row]),
                             'worst_joint': names[left[joint]], 'knots': len(heads[j][0])}
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    save('last.pt', {'model': state, 'completed_phases': len(PHASES), 'all_frame_report': reports})
    save('best.pt', {'model': state, 'completed_phases': len(PHASES), 'all_frame_report': reports})
    result = {**meta, 'completed': True, 'completed_phases': len(PHASES), 'all_frame_report': reports,
              'all_frame_max_error_rad_after_float32_command': float(error.max()),
              'all_frame_rmse_rad_after_float32_command': float(error.square().mean().sqrt()),
              'trainable_coefficients': sum(w.numel() for _, w in heads),
              'relu_knots': sum(len(k) for k, _ in heads), 'source_target_scalar_count': int(y.size),
              'elapsed_s': elapsed+time.monotonic()-started,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'best_checkpoint_sha256': sha(a.output/'best.pt'), 'last_checkpoint_sha256': sha(a.output/'last.pt'),
              'task_success_claim': False}
    (a.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
