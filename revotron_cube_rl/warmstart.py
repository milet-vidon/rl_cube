"""Supervise the RGB actor from executed expert actions, without launching Isaac.

Required NPZ arrays:
    policy_obs: [N, proprio_dim + height*width*3], float, RGB already in [0, 1]
    critic_obs: [N, critic_dim], float; used only to construct the critic shape
    actions: [N, 56], float supervision labels in [-1, 1]; applied successful
        demonstrations by default, expert-query labels for mixed DAgger data
Optional arrays:
    episode_id: [N], integer or string; split complete episodes, never random frames
    config_sha256: scalar string; must match the selected config when supplied
    supervision_type: scalar string; defaults to successful_executed_demonstrations
        or declares mixed_demonstrations_and_dagger_expert_queries
    executed_actions: [N, 56], float actual behavior actions in [-1, 1]; may differ
        from expert-query labels and is validated/reported, never used as labels

The checkpoint has a supervised actor, an unchanged critic, and a fresh RSL-RL
PPO optimizer. An optional initial checkpoint loads policy weights only. All
augmentation defaults are zero; nonzero denoising noise affects training only,
never the RGB pixels or the original-data MSE evaluation. Synthetic perturbations
have no physical rollout and are not additional successful demonstrations.
MSE is imitation accuracy, not evidence of successful grasping.
checkpoint.pt keeps the original best-MSE selection; checkpoint_last.pt preserves
the final training epoch separately, without claiming closed-loop validation.
Optional feedback normalization persists inside the actor checkpoint and preserves
the loaded function by compensating first-layer weights; raw data is unchanged.
Optional action-history disabling zeros that actor input block without weight
compensation: it intentionally removes the shortcut and changes the function.
Example:
    python warmstart.py demonstrations.npz --output runs/visual_warmstart --epochs 50
    python train.py --mode train --checkpoint runs/visual_warmstart/checkpoint.pt ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from isaac_compat import runtime_versions
import time

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from policy import VisionActorCritic
from rsl_compat import PPO

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_array(name: str, array: np.ndarray, n: int | None = None) -> None:
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f'{name} must be a floating-point 2D array, got {array.shape}/{array.dtype}')
    if n is not None and len(array) != n:
        raise ValueError(f'{name} has {len(array)} rows; expected {n}')
    for offset in range(0, len(array), 256):
        if not np.isfinite(array[offset:offset + 256]).all():
            raise ValueError(f'{name} contains non-finite values')


def load_demonstrations(path: Path, config: dict, config_hash: str):
    with np.load(path, allow_pickle=False) as data:
        missing = {'policy_obs', 'critic_obs', 'actions'} - set(data.files)
        if missing:
            raise ValueError(f'Missing demonstration arrays: {sorted(missing)}')
        policy_obs, critic_obs, actions = (data[key] for key in ('policy_obs', 'critic_obs', 'actions'))
        validate_array('policy_obs', policy_obs)
        n = len(policy_obs)
        if n < 2:
            raise ValueError('At least two demonstration transitions are required')
        validate_array('critic_obs', critic_obs, n)
        validate_array('actions', actions, n)
        if actions.shape[1] != 56:
            raise ValueError('Expected 56 action components')
        if float(np.abs(actions).max()) > 1.000001:
            raise ValueError('Supervision actions must be normalized/clipped labels in [-1,1]')
        supervision_type='successful_executed_demonstrations'
        if 'supervision_type' in data.files:
            declared=data['supervision_type']
            if declared.size!=1 or declared.dtype.kind not in 'SU':
                raise ValueError('supervision_type must contain one scalar string')
            item=declared.reshape(-1)[0]
            supervision_type=item.decode('utf-8') if isinstance(item,bytes) else str(item)
        if supervision_type not in ('successful_executed_demonstrations',
                                     'mixed_demonstrations_and_dagger_expert_queries'):
            raise ValueError(f'Unsupported supervision_type: {supervision_type}')
        executed_difference=None
        if 'executed_actions' in data.files:
            executed_actions=data['executed_actions']
            validate_array('executed_actions',executed_actions,n)
            if executed_actions.shape[1]!=56:
                raise ValueError('Expected 56 executed action components')
            if float(np.abs(executed_actions).max())>1.000001:
                raise ValueError('Executed actions must be normalized/clipped commands in [-1,1]')
            difference=np.abs(actions-executed_actions)
            executed_difference={'mean_absolute_difference':float(difference.mean()),
                                 'max_absolute_difference':float(difference.max()),
                                 'fraction_rows_different':float(np.any(difference>1e-6,axis=1).mean())}
        resolution = int(config['camera_resolution'])
        rgb_dim = resolution * resolution * 3
        proprio_dim = policy_obs.shape[1] - rgb_dim
        if proprio_dim <= 0 or proprio_dim % 56:
            raise ValueError(f'Invalid proprioception dimension {proprio_dim}; check image resolution/schema')
        if 'proprio_dim' in config and int(config['proprio_dim']) != proprio_dim:
            raise ValueError('Dataset proprioception width disagrees with config proprio_dim')
        for offset in range(0, n, 256):
            rgb = policy_obs[offset:offset + 256, proprio_dim:]
            if rgb.min() < -1e-6 or rgb.max() > 1.000001:
                raise ValueError('RGB policy inputs must already be normalized to [0,1]')
        ids_provided = 'episode_id' in data.files
        episode_id = data['episode_id'] if ids_provided else np.zeros(n, dtype=np.int64)
        if episode_id.shape != (n,) or episode_id.dtype.kind not in 'iuUS':
            raise ValueError('episode_id must be a 1D integer/string array with N entries')
        if episode_id.dtype.kind == 'S':
            episode_id = episode_id.astype('U')
        recorded_hash = None
        if 'config_sha256' in data.files:
            value = data['config_sha256']
            if value.size != 1:
                raise ValueError('config_sha256 must contain one scalar string')
            item = value.reshape(-1)[0]
            recorded_hash = item.decode('ascii') if isinstance(item, bytes) else str(item)
            if recorded_hash != config_hash:
                raise ValueError('Demonstration/config SHA256 mismatch; do not silently migrate observations')
    return policy_obs, critic_obs, actions, episode_id, {
        'transitions': n, 'policy_dim': policy_obs.shape[1], 'critic_dim': critic_obs.shape[1],
        'proprio_dim': proprio_dim, 'rgb_dim': rgb_dim, 'episode_ids_supplied': ids_provided,
        'config_hash_verified': recorded_hash is not None,
        'supervision_type':supervision_type,
        'action_label_semantics':'Applied successful demonstration actions.'
            if supervision_type=='successful_executed_demonstrations' else
            'Successful demonstration actions plus queried expert labels at DAgger states; labels can differ from behavior actions.',
        'executed_actions_semantics':'Optional actual behavior/mixed-policy commands; audited only, not supervision labels.',
        'executed_actions_supplied':executed_difference is not None,
        'labels_vs_executed':executed_difference,
    }


def split_episodes(episode_id: np.ndarray, seed: int, holdout_fraction: float):
    episodes = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    if len(episodes) < 2:
        return np.arange(len(episode_id)), np.empty(0, dtype=np.int64), episodes, episodes[:0]
    shuffled = rng.permutation(episodes)
    count = min(len(episodes) - 1, max(1, math.ceil(len(episodes) * holdout_fraction)))
    holdout_ids = shuffled[:count]
    is_holdout = np.isin(episode_id, holdout_ids)
    return np.flatnonzero(~is_holdout), np.flatnonzero(is_holdout), shuffled[count:], holdout_ids


def tensor_batch(array: np.ndarray, indices: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array[indices], dtype=torch.float32, device=device)


def augment_training_batch(observations: torch.Tensor, actions: torch.Tensor,
                           scales: torch.Tensor, target_noise: float = 0.,
                           joint_state_noise: float = 0., action_history_noise: float = 0.):
    """Perturb v2/v3 proprioception and correct toward the next expert target.

    q'=q+dq, target'=target+dt imply target'-q'=(target-q)+dt-dq.
    The corrective action is a-dt/scale, before the required [-1,1] clipping.
    Neither RGB nor velocity is changed. Default zero noise consumes no RNG and
    returns the original tensors, preserving the previous BC training semantics.
    """
    if not (target_noise or joint_state_noise or action_history_noise):
        return observations, actions
    dq=torch.randn_like(actions)*joint_state_noise if joint_state_noise else torch.zeros_like(actions)
    dt=torch.randn_like(actions)*target_noise if target_noise else torch.zeros_like(actions)
    da=torch.randn_like(actions)*action_history_noise if action_history_noise else torch.zeros_like(actions)
    augmented=observations.clone()
    augmented[:,:56]+=dq
    augmented[:,112:168]=(augmented[:,112:168]+da).clamp(-1,1)
    augmented[:,168:224]+=dt-dq
    corrected=(actions-dt/scales).clamp(-1,1)
    return augmented,corrected


def augment_rgb_batch(observations: torch.Tensor, resolution: int,
                      translation_px: float = 2.0, noise_std: float = .01,
                      brightness: float = .08, contrast: float = .08) -> torch.Tensor:
    """Apply small camera-domain perturbations while preserving the expert label.

    The simulator camera is fixed, so this is deliberately conservative: a few
    pixels of translation, mild exposure/contrast variation, and sensor noise.
    It regularizes local image changes without pretending to add new physics
    rollouts or changing the supervision labels.
    """
    if observations.shape[1] < 224 + resolution * resolution * 3:
        raise ValueError('RGB augmentation requires the v2/v3 224-dimensional proprioception layout')
    batch = observations.shape[0]
    rgb = observations[:, 224:].reshape(batch, resolution, resolution, 3).permute(0, 3, 1, 2)
    if translation_px:
        tx = (torch.rand(batch, device=rgb.device) * 2.0 - 1.0) * translation_px * 2.0 / resolution
        ty = (torch.rand(batch, device=rgb.device) * 2.0 - 1.0) * translation_px * 2.0 / resolution
        theta = torch.zeros(batch, 2, 3, device=rgb.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, rgb.shape, align_corners=False)
        rgb = F.grid_sample(rgb, grid, mode='bilinear', padding_mode='border', align_corners=False)
    if brightness:
        rgb = rgb + (torch.rand(batch, 1, 1, 1, device=rgb.device) * 2.0 - 1.0) * brightness
    if contrast:
        factor = 1.0 + (torch.rand(batch, 1, 1, 1, device=rgb.device) * 2.0 - 1.0) * contrast
        rgb = (rgb - .5) * factor + .5
    if noise_std:
        rgb = rgb + torch.randn_like(rgb) * noise_std
    augmented = observations.clone()
    augmented[:, 224:] = rgb.clamp(0., 1.).permute(0, 2, 3, 1).reshape(batch, -1)
    return augmented


@torch.no_grad()
def evaluate_mse(actor: nn.Module, observations: np.ndarray, actions: np.ndarray,
                 indices: np.ndarray, batch_size: int, device: str) -> dict | None:
    if not len(indices):
        return None
    actor.eval()
    total_error = 0.0
    active_error = 0.0
    active_count = 0
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        target = tensor_batch(actions, ix, device)
        predicted = actor(tensor_batch(observations, ix, device))
        error = (predicted - target).square()
        if not torch.isfinite(error).all():
            raise ValueError('Non-finite actor prediction/loss')
        active = target.abs() > .02
        total_error += float(error.sum())
        active_error += float(error[active].sum())
        active_count += int(active.sum())
    return {'mse': total_error / (len(indices) * actions.shape[1]),
            'active_action_mse': active_error / active_count if active_count else None,
            'active_action_components': active_count, 'samples': len(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('demonstrations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--initial-checkpoint', type=Path,
                        help='Load policy weights with the same config SHA256; never restore its optimizer')
    parser.add_argument('--normalize-feedback', action='store_true',
                        help='Scale target-q by inverse action scales, preserving the initial actor function')
    parser.add_argument('--disable-action-history', action='store_true',
                        help='Zero actor input gains for previous actions without compensating weights')
    parser.add_argument('--target-noise', type=float, default=0.,
                        help='Training-only Gaussian joint-target noise standard deviation in radians')
    parser.add_argument('--joint-state-noise', type=float, default=0.,
                        help='Training-only Gaussian measured-joint noise standard deviation in radians')
    parser.add_argument('--action-history-noise', type=float, default=0.,
                        help='Training-only Gaussian previous-action noise standard deviation in normalized units')
    parser.add_argument('--rgb-augmentation', action='store_true',
                        help='Training-only small RGB translation/exposure/noise augmentation')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--holdout-fraction', type=float, default=.2)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error('epochs, batch-size, threads and learning-rate must be positive')
    if not 0 < args.holdout_fraction < 1:
        parser.error('holdout-fraction must be between zero and one')
    noise_values=(args.target_noise,args.joint_state_noise,args.action_history_noise)
    if any(not math.isfinite(value) or value<0 for value in noise_values):
        parser.error('Noise standard deviations must be finite and nonnegative')
    output = args.output.resolve()
    if (output / 'checkpoint.pt').exists() and not args.overwrite:
        parser.error('Checkpoint already exists; select a new output directory or pass --overwrite')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    config_hash = sha256(config_path)
    observations, critic_obs, actions, episode_id, dataset_info = load_demonstrations(
        args.demonstrations, config, config_hash)
    augmentation_enabled=any(value>0 for value in noise_values) or args.rgb_augmentation
    if augmentation_enabled and dataset_info['proprio_dim']!=224:
        parser.error('Denoising augmentation requires the 224-dimensional v2/v3 proprioception layout')
    if len(config['joint_names'])!=56:
        parser.error('Config must specify the ordered 56 joint names')
    scales=torch.tensor([[.025 if name.startswith('Joint') else .06 for name in config['joint_names']]],
                        dtype=torch.float32,device=args.device)
    train_ix, holdout_ix, train_ids, holdout_ids = split_episodes(
        episode_id, args.seed, args.holdout_fraction)
    first = np.array([train_ix[0]])
    obs = TensorDict({'policy': tensor_batch(observations, first, args.device),
                      'critic': tensor_batch(critic_obs, first, args.device)}, batch_size=[1])
    policy = VisionActorCritic(obs, resolution=config['camera_resolution']).to(args.device)
    checkpoint_source=None
    if args.initial_checkpoint is not None:
        initial_path=args.initial_checkpoint.resolve()
        saved=torch.load(initial_path,map_location=args.device,weights_only=False)
        if saved.get('config_sha256')!=config_hash:
            raise ValueError('Initial checkpoint/config SHA256 mismatch')
        policy.load_state_dict(saved['policy'])
        checkpoint_source={'path':str(initial_path),'sha256':sha256(initial_path),
                           'optimizer_restored':False}
    if args.normalize_feedback:
        policy.actor.set_target_error_scaling(scales)
    if args.disable_action_history:
        policy.actor.disable_action_history()
    # Critic values are never supplied to actor training; only actor parameters are optimized.
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
    initial_actor = {key: value.detach().cpu().clone() for key, value in policy.actor.state_dict().items()}
    initial_critic = {key: value.detach().cpu().clone() for key, value in policy.critic.state_dict().items()}
    report = {'method': 'supervised_RGB_actor_warmstart', 'task_success_claim': False,
              'critic_trained': False, 'ppo_optimizer': 'fresh, initialized after supervised training',
              'config_sha256': config_hash, 'dataset_sha256': sha256(args.demonstrations),
              'dataset': dataset_info, 'seed': args.seed, 'device': args.device,
              'initial_checkpoint': checkpoint_source,
              'action_history': {'disable_requested':args.disable_action_history,
                  'disabled':bool((policy.actor.proprio_scale[112:168]==0).all())
                      if policy.actor.proprio_dim==224 else False,
                  'method':'Zero actor input gains without compensating weights; critic and raw data unchanged.',
                  'stored_in_checkpoint':True},
              'feedback_scaling': {'requested':args.normalize_feedback,
                  'nonidentity':bool((policy.actor.proprio_scale!=1).any()),
                  'target_error_gains':policy.actor.proprio_scale[168:224].detach().cpu().tolist()
                      if policy.actor.proprio_dim==224 else None,
                  'stored_in_checkpoint':True,'raw_observations_modified':False,
                  'initial_function_preserved_by_weight_compensation':bool(args.normalize_feedback)},
              'augmentation': {'enabled':augmentation_enabled,
                  'method':'independent_zero_mean_Gaussian_proprioception_denoising',
                  'target_noise_std_rad':args.target_noise,'joint_state_noise_std_rad':args.joint_state_noise,
                  'action_history_noise_std_normalized':args.action_history_noise,
                  'joint_block':'q-home += dq','target_error_block':'target-q += dt-dq',
                  'action_history_block':'clamp(previous_action+da,-1,1)',
                  'label':'clamp(original_action-dt/action_scale,-1,1)',
                  'arm_action_scale_rad':.025,'finger_action_scale_rad':.06,
                  'training_only':True,'rgb_modified':False,'velocity_modified':False,
                  'mse_protocol':'Original observations/actions for all reported train/holdout MSE and model selection.',
                  'rgb_enabled':args.rgb_augmentation,
                  'rgb_method':'random ±2 px translation, ±8% brightness/contrast, Gaussian sensor noise std .01' if args.rgb_augmentation else 'disabled',
                  'limitations':'Synthetic perturbations have no physics rollout or camera rerender; they are not new successful demonstrations. Action clipping can prevent exact one-step target recovery.'},
              'train_episode_ids': train_ids.tolist(), 'holdout_episode_ids': holdout_ids.tolist(),
              'independent_episode_holdout': bool(len(holdout_ix)),
              'validation_limit': 'Held-out imitation MSE does not establish closed-loop grasp success.'
                  if len(holdout_ix) else 'Single/missing episode IDs: NO independent validation; training fit only.',
              'initial_train': evaluate_mse(policy.actor, observations, actions, train_ix, args.batch_size, args.device),
              'initial_holdout': evaluate_mse(policy.actor, observations, actions, holdout_ix, args.batch_size, args.device),
              'epochs': []}
    if not dataset_info['config_hash_verified']:
        report['provenance_warning'] = 'Dataset has no config_sha256; dimensions checked, collection config not verified.'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_bytes(config_path.read_bytes())
    start = time.time()
    rng = np.random.default_rng(args.seed)
    best_score = float('inf')
    best_actor = None
    best_epoch = None
    for epoch in range(args.epochs):
        policy.actor.train()
        shuffled = rng.permutation(train_ix)
        for offset in range(0, len(shuffled), args.batch_size):
            ix = shuffled[offset:offset + args.batch_size]
            batch_obs = tensor_batch(observations, ix, args.device)
            target = tensor_batch(actions, ix, args.device)
            batch_obs,target=augment_training_batch(batch_obs,target,scales,*noise_values)
            if args.rgb_augmentation:
                batch_obs = augment_rgb_batch(batch_obs, config['camera_resolution'])
            prediction = policy.actor(batch_obs)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite supervised loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        train_metrics = evaluate_mse(policy.actor, observations, actions, train_ix, args.batch_size, args.device)
        holdout_metrics = evaluate_mse(policy.actor, observations, actions, holdout_ix, args.batch_size, args.device)
        record = {'epoch': epoch, 'train': train_metrics, 'holdout': holdout_metrics,
                  'elapsed_s': time.time() - start}
        report['epochs'].append(record)
        score = (holdout_metrics or train_metrics)['mse']
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_actor = {key: value.detach().cpu().clone() for key, value in policy.actor.state_dict().items()}
        print('BC_EPOCH', json.dumps(record), flush=True)
        (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    assert best_actor is not None
    last_actor = {key: value.detach().cpu().clone() for key, value in policy.actor.state_dict().items()}
    last_record = report['epochs'][-1]
    policy.actor.load_state_dict(best_actor)
    # Fresh optimizer groups exactly match the current train.py PPO construction.
    ppo = PPO(policy, num_learning_epochs=3, num_mini_batches=4, learning_rate=3e-4,
              entropy_coef=.002, device=args.device)
    with torch.no_grad():
        sample = obs.clone()
        reference = policy.act_inference(sample)
        sample['critic'] = sample['critic'] + 1.
        leak = float((reference - policy.act_inference(sample)).abs().max())
    assert leak == 0.0, 'Critic data changed actor output'
    assert all(torch.equal(initial_critic[key], value.detach().cpu()) for key, value in policy.critic.state_dict().items())
    report.update({'status': 'completed', 'selected_epoch': best_epoch,
                   'selection_metric': 'holdout_mse' if len(holdout_ix) else 'train_mse_no_validation',
                   'selected_train': evaluate_mse(policy.actor, observations, actions, train_ix, args.batch_size, args.device),
                   'selected_holdout': evaluate_mse(policy.actor, observations, actions, holdout_ix, args.batch_size, args.device),
                   'last_epoch': last_record['epoch'], 'last_train': last_record['train'],
                   'last_holdout': last_record['holdout'],
                   'last_checkpoint': 'checkpoint_last.pt',
                   'privileged_actor_leak_max': leak,
                   'actor_parameter_max_delta': max(float((value - initial_actor[key]).abs().max()) for key, value in best_actor.items()),
                   'elapsed_s': time.time() - start})
    contract = {'runtime_versions': runtime_versions(), 'action_names': config['joint_names'], 'revolute_dofs': 56,
                'camera_resolution': config['camera_resolution'], 'proprio_dim': dataset_info['proprio_dim'],
                'observation_policy': ['RGB', 'proprioception supplied by demonstration producer'],
                'critic_input_used_for_actor': False}
    torch.save({'policy': policy.state_dict(), 'optimizer': ppo.optimizer.state_dict(),
                'iteration': -1, 'config_sha256': config_hash, 'asset_contract': contract,
                'seed': args.seed, 'status': 'supervised_warmstart_not_validated_skill',
                'warmstart': {'selected_epoch': best_epoch, 'dataset_sha256': report['dataset_sha256']}},
               output / 'checkpoint.pt')
    policy.actor.load_state_dict(last_actor)
    torch.save({'policy': policy.state_dict(), 'optimizer': ppo.optimizer.state_dict(),
                'iteration': -1, 'config_sha256': config_hash, 'asset_contract': contract,
                'seed': args.seed, 'status': 'final_epoch_not_closed_loop_validated',
                'warmstart': {'selected_epoch': last_record['epoch'], 'dataset_sha256': report['dataset_sha256']}},
               output / 'checkpoint_last.pt')
    (output / 'asset_contract.json').write_text(json.dumps(contract, indent=2, allow_nan=False))
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('BC_COMPLETE', json.dumps({key: report[key] for key in ['status', 'selected_epoch', 'selected_train',
          'selected_holdout', 'last_epoch', 'last_train', 'last_holdout',
          'independent_episode_holdout', 'task_success_claim', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
