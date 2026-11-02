"""Collect real-image DAgger expert queries with an explicitly mixed controller.

This is TRAINING DATA COLLECTION, never a pure-policy evaluation. The teacher
uses the initial simulated cube pose. ``actions`` contains clipped expert query
LABELS, while ``executed_actions`` contains the actual clipped mixed commands.
Failed collection episodes are retained; query labels are not successful demos.

An optional --include-base must be the original successful, executed expert
demonstrations NPZ. Its executed_actions are copied from its original actions.
The original caller's successful-demonstration provenance is retained as an
assumption; NPZ tensor validation alone cannot establish physical task success.

Importing this module only loads NumPy/standard-library helpers, not Isaac.
"""
from __future__ import annotations
from isaac_compat import close_application

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SUPERVISION = 'mixed_demonstrations_and_dagger_expert_queries'
MAX_STEPS = 570


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError('Expected one scalar string')
    item = array.reshape(-1)[0]
    return item.decode('utf-8') if isinstance(item, bytes) else str(item)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_dataset(data: dict, config: dict, config_hash: str, *, base: bool = False) -> dict:
    """Validate/rematerialize arrays without claiming execution or task success."""
    required = {'policy_obs', 'critic_obs', 'actions', 'episode_id', 'config_sha256'}
    if not required.issubset(data):
        raise ValueError(f'Missing dataset fields: {sorted(required - set(data))}')
    if scalar_text(data['config_sha256']) != config_hash:
        raise ValueError('Dataset/config SHA256 mismatch')
    if base and 'supervision_type' in data and 'dagger' in scalar_text(data['supervision_type']).lower():
        raise ValueError('--include-base requires original executed demonstrations, not an earlier DAgger query dataset')
    arrays = {}
    expected = {'policy_obs': int(config['proprio_dim']) + int(config['camera_resolution']) ** 2 * 3,
                'critic_obs': int(config['critic_dim']), 'actions': 56}
    for name, width in expected.items():
        value = np.asarray(data[name])
        if value.ndim != 2 or value.shape[1] != width or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f'Invalid {name} shape/dtype: {value.shape}/{value.dtype}')
        for start in range(0, len(value), 256):
            if not np.isfinite(value[start:start + 256]).all():
                raise ValueError(f'Non-finite {name}')
        arrays[name] = value.astype(np.float32, copy=False)
    n = len(arrays['actions'])
    if not n or any(len(value) != n for value in arrays.values()):
        raise ValueError('Dataset must contain the same nonzero number of samples in every array')
    ids = np.asarray(data['episode_id'])
    if ids.shape != (n,) or ids.dtype.kind not in 'iuUS':
        raise ValueError('episode_id must contain one integer/string ID per sample')
    arrays['episode_id'] = ids.astype('U')
    if np.abs(arrays['actions']).max() > 1.000001:
        raise ValueError('Expert query labels must be clipped to [-1,1]')
    proprio_dim = int(config['proprio_dim'])
    for start in range(0, n, 256):
        rgb = arrays['policy_obs'][start:start + 256, proprio_dim:]
        if rgb.min() < -1e-6 or rgb.max() > 1.000001:
            raise ValueError('RGB must be normalized to [0,1]')
    if base:
        if 'executed_actions' in data and not np.array_equal(np.asarray(data['executed_actions']), arrays['actions']):
            raise ValueError('Base labels differ from execution; use original executed demonstrations')
        arrays['executed_actions'] = arrays['actions'].copy()
    else:
        executed = np.asarray(data.get('executed_actions'))
        if executed.shape != (n, 56) or not np.issubdtype(executed.dtype, np.floating):
            raise ValueError('DAgger collection must provide actual executed_actions [N,56]')
        if not np.isfinite(executed).all() or np.abs(executed).max() > 1.000001:
            raise ValueError('Invalid executed mixed actions')
        arrays['executed_actions'] = executed.astype(np.float32, copy=False)
    return arrays


def merge_datasets(base: dict | None, queries: dict, config: dict, config_hash: str) -> dict:
    """Prefix episode IDs so base/query episodes can never collide, including failures."""
    parts = []
    if base is not None:
        original = validate_dataset(base, config, config_hash, base=True)
        original['episode_id'] = np.asarray(['base:' + value for value in original['episode_id']])
        original['sample_source'] = np.full(len(original['actions']), 'base_executed_demonstration')
        parts.append(original)
    queried = validate_dataset(queries, config, config_hash)
    queried['episode_id'] = np.asarray(['dagger:' + value for value in queried['episode_id']])
    queried['sample_source'] = np.full(len(queried['actions']), 'dagger_expert_query')
    parts.append(queried)
    merged = {name: np.concatenate([part[name] for part in parts], axis=0)
              for name in ('policy_obs', 'critic_obs', 'actions', 'executed_actions', 'episode_id', 'sample_source')}
    merged.update(config_sha256=np.asarray(config_hash), supervision_type=np.asarray(SUPERVISION),
                  action_semantics=np.asarray('actions=clipped expert labels; executed_actions=actual commands; query labels need not have been executed'))
    return merged


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seeds', default='211,223,227')
    parser.add_argument('--beta', type=float, default=.8)
    parser.add_argument('--include-base', type=Path)
    parser.add_argument('--teacher-mode', choices=['staged_parallel','staged_oblique','legacy_downward'],
                        default='staged_parallel',
                        help='Requires a current staged teacher safety certificate.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.beta) or not 0. <= args.beta <= 1.:
        parser.error('beta must be finite and within [0,1]')
    try:
        args.seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    except ValueError:
        parser.error('seeds must be comma-separated integers')
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    name = Path(args.name)
    if name.is_absolute() or len(name.parts) != 1 or ':' in args.name or name.name in ('.', '..'):
        parser.error('name must be one new directory name under runs/')
    if (ROOT / 'runs' / args.name).exists():
        parser.error('Output already exists; use a new name to preserve prior data')
    for key in ('checkpoint', 'include_base'):
        value = getattr(args, key)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f'{key} does not exist: {value}')
            setattr(args, key, value)
    return args


def main(argv=None):
    args = parse_args(argv)
    from stage_gate import require_visual_training_gate
    require_visual_training_gate()
    if args.teacher_mode!='staged_parallel':
        raise ValueError('This collection path requires the certified staged_parallel teacher')
    config_path = ROOT / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config_hash = sha256(config_path)
    base = None
    if args.include_base:
        with np.load(args.include_base, allow_pickle=False) as data:
            base = {name: data[name] for name in data.files}
        validate_dataset(base, config, config_hash, base=True)
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(config_path.read_bytes())
    report = {
        'status': 'starting', 'mode': 'dagger_collection', 'real_rgb': True, 'action_dim': 56,
        'started_at': datetime.now(timezone.utc).isoformat(), 'config_sha256': config_hash,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'beta': args.beta, 'mixing_rule': 'clamp(beta * clipped_expert_query + (1-beta) * policy_output, -1, 1)',
        'teacher_mode': args.teacher_mode,
        'max_steps_per_episode': MAX_STEPS, 'preview_camera': False,
        'supervision_type': SUPERVISION, 'actions_semantics': 'clipped expert query labels, not necessarily executed',
        'executed_actions_semantics': 'actual commands read back from env.last_actions after each step',
        'failed_query_episodes_retained': True, 'task_success_claim': False,
        'teacher_ground_truth': 'Initial simulator cube pose is used only by the training teacher/mixture. It is not a pure-policy evaluation input.',
        'collection_success_limit': 'Success belongs to the teacher-policy mixture, not to the policy alone.',
        'include_base': str(args.include_base) if args.include_base else None,
        'base_sha256': sha256(args.include_base) if args.include_base else None,
        'base_provenance_limit': 'Caller supplies original successful executed demonstrations; tensor validation does not independently prove success.',
        'episodes': [],
    }
    write_json(out / 'result.json', report)
    app = None
    env = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        from runtime_platform import platform_kit_args
        app = AppLauncher(headless=True, enable_cameras=True,
            kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
        import torch
        from environment import CubePickupEnv
        from policy import VisionActorCritic
        from position_policy import PositionActorCritic, rgb_to_position_features
        from absolute_policy import AbsolutePositionActorCritic
        from traditional_controller import DownwardController, StagedDownwardController
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved['config_sha256'] != config_hash:
            raise ValueError('Checkpoint/config SHA256 mismatch')
        env = CubePickupEnv(vision=True, preview=False, auto_reset=False)
        if sha256(config_path) != config_hash:
            raise ValueError('Config changed during collector initialization')
        position_only = bool(saved.get('position_features', False))
        absolute_mode = saved.get('output_mode') == 'absolute_joint_target'
        position_feature_dim = int(saved.get('position_feature_dim', 6))
        report['actor_input'] = '224 proprioception + 6 explicit cube-position features' if position_only else '224 proprioception + RGB'
        report['actor_output'] = 'absolute joint targets converted to PD increments' if absolute_mode else 'incremental joint targets'
        report['raw_rgb_used_only_for_position_features'] = position_only
        def model_obs(raw_obs):
            if not position_only:
                return raw_obs
            converted = raw_obs.clone()
            raw = raw_obs['policy']
            features = rgb_to_position_features(raw[:, env.proprio_dim:], config['camera_resolution'])
            if position_feature_dim == 7:
                progress = env.episode_length_buf.float().reshape(-1, 1) / max(1, env.max_episode_length)
                features = torch.cat((features, progress.to(features.device, dtype=features.dtype)), dim=-1)
            if features.shape[-1] != position_feature_dim:
                raise ValueError(f'Expected {position_feature_dim} position features, got {features.shape[-1]}')
            converted['policy'] = torch.cat((raw[:, :env.proprio_dim], features), dim=-1)
            return converted
        policy = (AbsolutePositionActorCritic(model_obs(env.get_observations()), saved['joint_center'], saved['joint_half_range']) if absolute_mode
                  else PositionActorCritic(model_obs(env.get_observations())) if position_only
                  else VisionActorCritic(env.get_observations(), resolution=config['camera_resolution']))
        policy.load_state_dict(saved['policy'])
        policy.eval()
        del saved
        write_json(out / 'asset_contract.json', env.contract)
        chunks = []
        report['status'] = 'collecting'
        for episode, seed in enumerate(args.seeds):
            episode_started = time.monotonic()
            env.rng = np.random.default_rng(seed)
            env.reset()
            env.pair_impulses = {}
            if args.teacher_mode == 'legacy_downward':
                teacher = DownwardController(env, width=.096)
            elif args.teacher_mode == 'staged_oblique':
                teacher = StagedDownwardController(env, width=.096, grasp_mode='oblique', tilt_deg=1.0)
            else:
                teacher = StagedDownwardController(env, width=.096, grasp_mode='parallel')
            observations, critics, labels, executed, rows = [], [], [], [], []
            obs = env.get_observations()
            if float(obs['policy'][:, env.proprio_dim:].std()) <= .015:
                raise ValueError('Black/constant chest camera image')
            for step in range(MAX_STEPS):
                obs = env.get_observations()
                with torch.no_grad():
                    policy_action = policy.act_inference(model_obs(obs))
                    if absolute_mode:
                        target = policy.actor.decode_targets(policy_action)
                        policy_action = (target - env.targets.to(target.device)) / env.scales.to(target.device)
                        policy_action = policy_action.clamp(-1, 1)
                    expert_action = teacher.action().clamp(-1, 1)
                    action = (args.beta * expert_action + (1. - args.beta) * policy_action).clamp(-1, 1)
                    if action.shape != (1, 56) or not torch.isfinite(action).all():
                        raise ValueError('Invalid mixed action')
                observations.append(obs['policy'][0].cpu().numpy().copy())
                critics.append(obs['critic'][0].cpu().numpy().copy())
                labels.append(expert_action[0].cpu().numpy().copy())
                _, _, done, _ = env.step(action)
                executed.append(env.last_actions[0].cpu().numpy().copy())
                rows.append(dict(step=step, phase=teacher.name, **env.metrics))
                if step % 100 == 0:
                    progress = dict(status='collecting', episode=episode, seed=seed, beta=args.beta,
                                    step=step + 1, max_steps=MAX_STEPS, elapsed_s=time.monotonic() - started,
                                    height_m=env.metrics['height_m'], hold_steps=env.metrics['hold_steps'])
                    write_json(out / 'progress.json', progress)
                    print('DAGGER_PROGRESS', json.dumps(progress), flush=True)
                if bool(done[0]):
                    break
            chunk = dict(policy_obs=np.asarray(observations, dtype=np.float32),
                         critic_obs=np.asarray(critics, dtype=np.float32), actions=np.asarray(labels, dtype=np.float32),
                         executed_actions=np.asarray(executed, dtype=np.float32),
                         episode_id=np.full(len(rows), episode, dtype=np.int64),
                         config_sha256=np.asarray(config_hash), supervision_type=np.asarray(SUPERVISION))
            np.savez_compressed(out / f'queries_{episode:03d}_{seed}.npz', **chunk)
            chunks.append(chunk)
            result = dict(episode=episode, seed=seed, beta=args.beta, steps=len(rows),
                          success=bool(env.metrics['success']), success_is_mixed_collection=True,
                          max_height_m=max(row['height_m'] for row in rows),
                          max_bilateral_height_m=max((row['height_m'] for row in rows if row['bilateral_contact']
                              and not row['drop'] and row['table_impulse'] < 1e-5), default=0.),
                          max_hold_steps=max(row['hold_steps'] for row in rows),
                          robot_table_impulse_sum=sum(row['robot_table_impulse'] for row in rows),
                          dropped=any(row['drop'] for row in rows),
                          elapsed_s=time.monotonic() - episode_started, contact_pairs=env.pair_impulses)
            report['episodes'].append(result)
            report['elapsed_s'] = time.monotonic() - started
            write_json(out / f'trajectory_{episode:03d}_{seed}.json', rows)
            write_json(out / 'result.json', report)
            env.save_rgb(out / f'chest_{episode:03d}_{seed}.png')
            print('DAGGER_EPISODE', json.dumps(result), flush=True)
        queries = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
                   for name in ('policy_obs', 'critic_obs', 'actions', 'executed_actions', 'episode_id')}
        queries['config_sha256'] = np.asarray(config_hash)
        del chunks, observations, critics, labels, executed, chunk
        report['status'] = 'packing_dataset'
        write_json(out / 'progress.json', dict(status='packing_dataset', elapsed_s=time.monotonic() - started))
        merged = merge_datasets(base, queries, config, config_hash)
        merged['collection_beta'] = np.asarray(args.beta)
        dataset = out / 'demonstrations.npz'
        np.savez_compressed(dataset, **merged)
        report.update(status='completed', dataset=str(dataset), dataset_sha256=sha256(dataset),
                      query_transitions=len(queries['actions']), combined_transitions=len(merged['actions']),
                      query_successes=sum(int(episode['success']) for episode in report['episodes']),
                      elapsed_s=time.monotonic() - started,
                      base_episode_ids=np.unique(merged['episode_id'][merged['sample_source'] == 'base_executed_demonstration']).tolist(),
                      query_episode_ids=np.unique(merged['episode_id'][merged['sample_source'] == 'dagger_expert_query']).tolist())
        write_json(out / 'result.json', report)
        write_json(out / 'progress.json', dict(status='completed', elapsed_s=report['elapsed_s'], dataset=str(dataset)))
        print('DAGGER_COMPLETE', json.dumps({key: report[key] for key in
            ('status', 'query_transitions', 'combined_transitions', 'query_successes', 'task_success_claim', 'elapsed_s')}), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_s=time.monotonic() - started)
        write_json(out / 'result.json', report)
        raise
    finally:
        code = 1 if sys.exc_info()[0] else 0
        if app is not None:
            def cleanup_deadline():
                time.sleep(15)
                print('Kit cleanup timeout; terminating this collector child process', flush=True)
                os._exit(code)
            threading.Thread(target=cleanup_deadline, daemon=True).start()
            try:
                if env is not None:
                    env.close()
                close_application(app)
            except BaseException:
                code = 1
                raise


if __name__ == '__main__':
    main()
