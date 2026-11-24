"""Inspect completed v3 supervision; train only with an explicit --train flag.

One trajectory is development fitting, never an independent generalization
result. The original environment remains the final action projection boundary.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch

import phase_basis_policy_v4
import phase_contact_policy
import phase_residual_policy_v5
import dagger_residual_dataset_v5
import dagger_collection_core_v1
import dagger_projected_collection_v3
from phase_residual_policy_v5 import PhaseResidualPolicy, project_left_targets, FEATURES, SCHEMA, FEATURE_SCHEMA
from dagger_residual_dataset_v5 import load_run, verify_saved_datasets, sha


def load_baseline(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for module in (phase_basis_policy_v4, phase_contact_policy):
        name = Path(module.__file__).name
        if saved['metadata']['source_sha256'][name] != sha(module.__file__):
            raise ValueError('Frozen baseline inference source changed: ' + name)
    phase_basis_policy_v4.restored(saved)
    if any(not torch.isfinite(value).all() for value in saved['model'].values()):
        raise ValueError('Nonfinite frozen baseline state')
    return saved, sha(path)


def tensor_episode(episode, device):
    return tuple(torch.as_tensor(episode[key], device=device).unsqueeze(0) for key in
                 ('x', 'previous_left', 'labels', 'mask'))


def build_model(baseline, episodes, bound, hidden_size):
    eligible = np.concatenate([episode['x'][episode['mask']] for episode in episodes])
    return PhaseResidualPolicy(baseline, eligible.mean(0), np.maximum(eligible.std(0), .001), bound, hidden_size)


def inspect(episodes, model):
    """Report label range and reachable projected intervals without clipping."""
    differences, misses, distances = [], [], []
    total, student = 0, 0
    with torch.no_grad():
        for episode in episodes:
            x, previous, labels, mask = tensor_episode(episode, 'cpu')
            base = model.baseline_targets(x, previous)
            low_proposal = torch.clamp(base - model.residual_bound, model.baseline.lower.float(), model.baseline.upper.float())
            high_proposal = torch.clamp(base + model.residual_bound, model.baseline.lower.float(), model.baseline.upper.float())
            low = project_left_targets(low_proposal, previous, episode['parameters'], episode['left_indices'], episode['projector'])
            high = project_left_targets(high_proposal, previous, episode['parameters'], episode['left_indices'], episode['projector'])
            lower, upper = torch.minimum(low, high), torch.maximum(low, high)
            distance = torch.maximum((lower - labels).clamp_min(0), (labels - upper).clamp_min(0))
            differences.append((labels - base)[mask].numpy())
            misses.append((distance[mask] > 1e-6).numpy())
            distances.append(distance[mask].numpy())
            total += int(mask.sum())
            student += sum(bool(keep) and source == 'student' for keep, source in zip(episode['mask'], episode['executed_sources']))
    delta, unreachable, distance = np.concatenate(differences), np.concatenate(misses), np.concatenate(distances)
    return {'eligible_steps': total, 'eligible_student_executed_steps': student,
            'residual_bound_rad': model.residual_bound.tolist(),
            'joint_names': [episodes[0]['joint_names'][i] for i in episodes[0]['left_indices']],
            'expert_projected_minus_frozen_base_min_rad': delta.min(0).tolist(),
            'expert_projected_minus_frozen_base_max_rad': delta.max(0).tolist(),
            'absolute_difference_quantiles': {str(q): np.quantile(np.abs(delta), q, axis=0).tolist() for q in (.5, .9, .95, .99, 1.)},
            'projected_unreachable_count_by_joint': unreachable.sum(0).tolist(),
            'projected_max_unreachable_distance_rad_by_joint': distance.max(0).tolist(),
            'unreachable_label_elements': int(unreachable.sum()),
            'reachability_float32_tolerance_rad': 1e-6,
            'labels_clipped_or_bound_misses_filtered': False}


def metrics(episodes, model, device='cpu'):
    model.eval()
    errors, base_errors, student_errors = [], [], []
    with torch.no_grad():
        for episode in episodes:
            x, previous, labels, mask = tensor_episode(episode, device)
            proposed, _, _ = model(x, previous, None)  # reset for every episode
            applied = project_left_targets(proposed, previous, episode['parameters'], episode['left_indices'], episode['projector'])
            base = project_left_targets(model.baseline_targets(x, previous), previous, episode['parameters'], episode['left_indices'], episode['projector'])
            error = (applied - labels)[0].cpu().numpy()
            errors.append(error[episode['mask']])
            base_errors.append((base - labels)[mask].cpu().numpy())
            select = episode['mask'] & (np.asarray(episode['executed_sources']) == 'student')
            if select.any(): student_errors.append(error[select])
    def summarize(parts):
        if not parts: return None
        value = np.concatenate(parts)
        return {'rmse_rad': float(np.sqrt(np.mean(value ** 2))), 'max_abs_rad': float(np.abs(value).max()),
                'rmse_by_joint_rad': np.sqrt(np.mean(value ** 2, axis=0)).tolist()}
    return {'expert_projected_fit': summarize(errors), 'frozen_baseline_projected_fit': summarize(base_errors),
            'student_visited_state_fit': summarize(student_errors),
            'claim': 'offline_development_fit_only_no_new_closed_loop_success'}


def source_hashes():
    return {Path(module.__file__).name: sha(module.__file__) for module in
            (phase_residual_policy_v5, dagger_residual_dataset_v5, phase_basis_policy_v4, phase_contact_policy,
             dagger_collection_core_v1, dagger_projected_collection_v3)} | {Path(__file__).name: sha(__file__)}


def qualification_manifest(datasets, baseline_hash, model):
    """New transition qualification; never reinterpret old episode flags."""
    return {'schema': 'v5_reviewable_development_bc_transition_qualification_v1',
            'eligible_for_bc_transition_supervision': True, 'eligible_for_rl_episode_training': False,
            'pure_student_episode_success_claim': False, 'independent_generalization_claim': False,
            'requires_root_review_before_training': True,
            'qualification_basis': 'completed records with verified actual pre/post state, original gates and reproduced expert projected labels only',
            'original_result_files_modified': False,
            'physics_family_sha256': datasets[0]['physics_family_sha256'],
            'frozen_baseline_checkpoint_sha256': baseline_hash, 'residual_bound_rad': model.residual_bound.tolist(),
            'source_sha256': source_hashes(),
            'datasets': [{'evidence': data['evidence'], 'collection_source_sha256': data['collection_source_sha256'],
                          'original_result_flags': {key:data['result'].get(key) for key in
                              ('eligible_training_episode','runtime_bridge_validated','pure_student_episode_claim',
                               'original_wrist_gate_qualified','failure','collection_mode')},
                          'eligible_transition_steps': np.flatnonzero(data['mask']).tolist(),
                          'excluded_records': data['excluded_labels'],
                          'eligible_teacher_steps': sum(bool(ok) and source=='teacher' for ok,source in zip(data['mask'],data['executed_sources'])),
                          'eligible_student_steps': sum(bool(ok) and source=='student' for ok,source in zip(data['mask'],data['executed_sources']))}
                         for data in datasets]}


def require_reviewed_qualification(path, expected_sha256, recomputed):
    if not path or not expected_sha256:
        raise ValueError('Training requires an explicit reviewed qualification file and pinned SHA256')
    if sha(path) != expected_sha256: raise ValueError('Qualification SHA256 mismatch')
    if json.loads(Path(path).read_text()) != recomputed:
        raise ValueError('Reviewed transition qualification differs from current data/code/bounds')


def load_checkpoint(path, baseline_path, dataset_directories, *, expected_physics_family):
    """Strict deployment/evaluation boundary; all external evidence rechecked."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    if metadata['schema'] != SCHEMA or metadata['feature_schema'] != FEATURE_SCHEMA:
        raise ValueError('V5 checkpoint feature/model schema mismatch')
    if metadata['physics_family_sha256'] != expected_physics_family:
        raise ValueError('Do not load baseline state policy into a different physics family')
    if metadata['source_sha256'] != source_hashes(): raise ValueError('V5 inference/training consumer source changed')
    baseline, baseline_hash = load_baseline(baseline_path)
    if baseline_hash != metadata['frozen_baseline_checkpoint_sha256']:
        raise ValueError('Frozen v4 checkpoint hash mismatch')
    verify_saved_datasets(metadata, dataset_directories)
    state = saved['model']
    for name, value in baseline['model'].items():
        if not torch.equal(value, state['baseline.' + name]): raise ValueError('Frozen baseline weights were modified')
    model = PhaseResidualPolicy(baseline, state['mean'], state['std'], metadata['residual_bound_rad'], metadata['hidden_size'])
    if not torch.equal(model.residual_bound, state['residual_bound']): raise ValueError('Residual bound metadata differs from checkpoint')
    model.load_state_dict(state)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite V5 model weights')
    return model.eval(), metadata


def train(model, episodes, validation, args, metadata):
    device = args.device
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-5)
    rng = random.Random(args.seed)
    best = float('inf')
    started = time.monotonic()
    with (args.output / 'training.jsonl').open('w') as journal:
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(episodes))); rng.shuffle(order)
            losses = []
            for index in order:
                episode = episodes[index]
                x, previous, labels, mask = tensor_episode(episode, device)
                optimizer.zero_grad(set_to_none=True)
                proposed, _, _ = model(x, previous, None)  # full history from reset, no shuffled windows
                applied = project_left_targets(proposed, previous, episode['parameters'], episode['left_indices'], episode['projector'])
                loss = (((applied - labels)[mask] / .001) ** 2).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            assessed = metrics(validation or episodes, model, device)
            score = assessed['expert_projected_fit']['rmse_rad']
            entry = {'epoch': epoch + 1, 'loss_mrad_squared': float(np.mean(losses)),
                     'selection_scope': 'held_out_development_episode' if validation else 'training_trajectory_fit',
                     'metrics': assessed, 'elapsed_s': time.monotonic() - started}
            journal.write(json.dumps(entry, allow_nan=False) + '\n'); journal.flush()
            if score < best:
                best = score
                torch.save({'model': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'metadata': metadata | {'best_epoch': epoch + 1, 'selection_scope': entry['selection_scope']}}, args.output / 'model.pt')
            if (epoch + 1) % 10 == 0 or epoch == 0: print('V5_DEVELOPMENT_FIT', json.dumps(entry), flush=True)
    best_saved = torch.load(args.output / 'model.pt', map_location='cpu', weights_only=True)
    model.cpu().load_state_dict(best_saved['model'])
    return {'training_fit_cpu_original_projection': metrics(episodes, model),
            'validation_fit_cpu_original_projection': metrics(validation, model) if validation else None,
            'checkpoint_sha256': sha(args.output / 'model.pt'), 'best_epoch': best_saved['metadata']['best_epoch'],
            'independent_generalization_demonstrated': False, 'closed_loop_success_demonstrated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--validation-run', type=Path, action='append', default=[])
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--residual-bound', default='.01', help='candidate scalar or 28 comma-separated radian bounds; inspect before choosing')
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--train', action='store_true')
    p.add_argument('--qualification', type=Path, help='reviewed inspect qualification.json; required for training')
    p.add_argument('--qualification-sha256', help='pin the reviewed qualification bytes')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0: p.error('Positive training hyperparameters required')
    paths = args.run + args.validation_run
    if len({str(path.resolve()) for path in paths}) != len(paths): p.error('Split only by distinct complete episodes')
    baseline, baseline_hash = load_baseline(args.baseline)
    datasets, family = [], None
    for path in paths:
        dataset = load_run(path, baseline, baseline_hash, family)
        family = dataset['physics_family_sha256']; datasets.append(dataset)
    if len({data['evidence']['files_sha256']['transitions.jsonl'] for data in datasets}) != len(datasets):
        raise ValueError('Duplicate trajectory content cannot be train/validation episodes')
    episodes, validation = datasets[:len(args.run)], datasets[len(args.run):]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    bound = [float(x) for x in args.residual_bound.split(',')]
    model = build_model(baseline, episodes, bound, args.hidden_size)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'development_fit_only', 'physics_family_sha256': family,
              'train_runs': [d['evidence'] for d in episodes], 'validation_runs': [d['evidence'] for d in validation],
              'training_bound_inspection': inspect(episodes, model),
              'validation_bound_inspection': inspect(validation, model) if validation else None,
              'excluded_labels': {d['episode_id']: d['excluded_labels'] for d in datasets},
              'baseline_fit': metrics(episodes, model), 'training_started': False,
              'actor_scope': 'privileged_simulation_state_and_public_phase_history_not_vision',
              'previous_executed_source_is_actor_feature': False,
              'source_sha256': source_hashes(), 'frozen_baseline_checkpoint_sha256': baseline_hash}
    (args.output / 'inspection.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    qualification = qualification_manifest(datasets, baseline_hash, model)
    (args.output / 'qualification.json').write_text(json.dumps(qualification, indent=2, allow_nan=False))
    if not args.train:
        print('V5_INSPECTION_ONLY', args.output / 'inspection.json'); return
    for key in ('training_bound_inspection', 'validation_bound_inspection'):
        if report[key] and report[key]['unreachable_label_elements']:
            raise ValueError('Residual bound cannot reach original labels; inspect per-joint evidence and explicitly choose bounds before training')
    if report['training_bound_inspection']['eligible_student_executed_steps'] == 0:
        raise ValueError('No eligible student-visited state supervision; preserve replay is not new DAgger data')
    require_reviewed_qualification(args.qualification, args.qualification_sha256, qualification)
    metadata = {'schema': SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'hidden_size': args.hidden_size,
                'residual_bound_rad': model.residual_bound.tolist(), 'joint_names': episodes[0]['joint_names'],
                'output_indices': episodes[0]['left_indices'], 'physics_family_sha256': family,
                'frozen_baseline_checkpoint_sha256': baseline_hash, 'source_sha256': source_hashes(),
                'dataset_evidence': [d['evidence'] for d in datasets], 'seed': args.seed,
                'reviewed_qualification_sha256': args.qualification_sha256,
                'eligible_for_rl_episode_training': False,
                'teacher_hidden_state_or_labels_in_actor': False, 'previous_executed_source_in_actor': False,
                'phase_source': 'original_stage_controller_same_action_phase_progress',
                'reward_learning': False, 'development_only': True, 'independent_generalization_demonstrated': False}
    result = train(model, episodes, validation, args, metadata)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('V5_DEVELOPMENT_TRAINING_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__': main()
