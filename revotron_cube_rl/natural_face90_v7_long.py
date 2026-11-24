"""PPO with a centered index-pad approach, paired diagnostics and separately held-out evaluation."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import natural_face90_recipe_long as guided
import natural_face90_v6_long as previous
from natural_face90_recipe_long import (
    HERE, ROOT, PARENT, ZERO, available_gpu, digest, exclusive, final_seeds, read,
    request, save, stats, worker_failure, worker_software,
)

SCHEMA = 'natural_face90_recipe_contact_turn_ppo_v4'
FILES = (*previous.FILES, 'natural_face90_v7_long.py', 'natural_face90_v7_worker.py',
         'natural_face90_turn_guide_v4.py', 'run_natural_face90_v7_long.sh')


def fingerprint():
    return {name: digest(HERE/name) for name in FILES}


def transfer_origin(path, pins):
    path = path.resolve()
    match = re.fullmatch(r'iteration_(\d{4})', path.name)
    source = path.parent if match else path
    cfg, state = read(source/'config.json'), read(source/'state.json')
    k = int(match[1]) if match else state['iteration']
    if (type(k) is not int or not 1 <= k <= state['iteration']
            or cfg['schema'] != previous.SCHEMA or cfg['parent_sha'] != PARENT
            or cfg['source_sha256'] != {name: pins[name] for name in previous.FILES}):
        raise ValueError('Expected a committed v3 contact-turn iteration and matching source hashes')
    folder = source/f'iteration_{k:04d}'
    slot = folder/'train'
    result, req = read(folder/'result.json'), read(slot/'request.json')
    previous.validate_request(req)
    if (result['iteration'] != k or req['iteration'] != k or req['role'] != 'train'
            or req['software'] != worker_software(cfg['source_sha256'])):
        raise ValueError('Invalid source iteration or software')
    _, learning = previous.accepted(slot, req)
    if (result['learning'] != learning or
            (k == state['iteration'] and (state['checkpoint'], state['checkpoint_sha'])
             != (learning['output_checkpoint'], learning['output_checkpoint_sha']))):
        raise ValueError('Accepted source checkpoint disagrees with campaign state')
    return dict(folder=str(folder), iteration=k, checkpoint=learning['output_checkpoint'],
                sha=learning['output_checkpoint_sha'], schema=cfg['schema'], software=req['software'])


@contextmanager
def idle_source(source):
    parent = source.parent if re.fullmatch(r'iteration_\d{4}', source.name) else source
    with (parent/'campaign.lock').open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('源训练仍在运行；请先在原训练终端按 Ctrl+C，待工作进程退出后重试。') from exc
        yield


def validate_request(req):
    if (req.get('recipe_schema') != SCHEMA or req.get('role') not in ('train', 'validate')
            or req.get('parent_sha') != PARENT or not req.get('checkpoint')
            or not re.fullmatch(r'[0-9a-f]{64}', req.get('checkpoint_sha') or '')):
        raise ValueError('Invalid centered-contact request')
    common = dict(req, recipe_schema=guided.GUIDED_SCHEMA)
    origin = req.get('transfer')
    evaluation_flags = [req[name] for name in ('paired_validation', 'fresh_source_validation') if name in req]
    if evaluation_flags:
        if len(evaluation_flags) != 1 or evaluation_flags[0] is not True:
            raise ValueError('Invalid frozen-source evaluation mode')
        if req['role'] != 'validate' or not origin or origin['iteration'] != req['iteration']:
            raise ValueError('Frozen source evaluation requires its original iteration and checkpoint')
        common.pop('transfer')
    guided.validate_request(common)
    if origin:
        prior = {name: req['software'][name] for name in previous.FILES if name.endswith('.py')}
        if (origin.get('schema') != previous.SCHEMA or origin.get('software') != prior
                or (req['checkpoint'], req['checkpoint_sha']) != (origin['checkpoint'], origin['sha'])):
            raise ValueError('Mismatched v3 contact checkpoint transfer')


def accepted_report(slot, req):
    receipt = read(slot/'accepted.json')
    if not re.fullmatch(r'attempt_[a-f0-9]{8,32}', receipt['attempt']):
        raise ValueError('Invalid accepted attempt')
    report = (slot/receipt['attempt']/'report.json').resolve()
    if (report.parent.parent != slot.resolve() or read(slot/'request.json') != req
            or digest(slot/'request.json') != receipt['request_sha'] or digest(report) != receipt['report_sha']):
        raise ValueError('Changed accepted wave or request')
    return report


def wave(slot, req, fd):
    validate_request(req)
    if worker_software(fingerprint()) != req['software']:
        raise ValueError('Experiment software changed since configuration')
    slot.mkdir(parents=True, exist_ok=True)
    plan = slot/'request.json'
    if plan.exists() and read(plan) != req:
        raise ValueError(f'Wave request changed: {plan}')
    if not plan.exists():
        save(plan, req)
    if (slot/'accepted.json').exists():
        return previous.verify_turn_report(accepted_report(slot, req), req)
    attempt = slot/('attempt_'+uuid.uuid4().hex[:12])
    with available_gpu() as gpu_fds, attempt.with_suffix('.log').open('x') as log:
        if digest(Path(req['checkpoint'])) != req['checkpoint_sha']:
            raise ValueError('Checkpoint changed before physical rollout')
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(gpu_fds[0]), CUBE_DASH_ISAAC_FD=str(gpu_fds[1]))
        print('Centered-contact PPO', req['role'], 'iteration', req['iteration'], 'log', log.name, flush=True)
        result = subprocess.run([sys.executable, '-B', '-u', str(HERE/'natural_face90_v7_worker.py'),
                                 '--request', str(plan), '--out', str(attempt)],
                                stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(fd, *gpu_fds))
        if result.returncode:
            raise worker_failure(result.returncode, attempt, log.name)
    report = attempt/'report.json'
    rows, learning = previous.verify_turn_report(report, req)
    save(slot/'accepted.json', dict(attempt=attempt.name, request_sha=digest(plan), report_sha=digest(report)))
    return rows, learning


def score(rows):
    return dict(stats(rows),
                distal_contact_episodes=sum(r['distal_contact_frames'] > 0 for r in rows),
                qualified_contact_episodes=sum(r['qualified_index_frames'] > 0 for r in rows),
                drive_episodes=sum(r['drive_frames'] > 0 for r in rows),
                safe_index_2deg_episodes=sum(not r['unsafe'] and r['peak_index_driven_deg'] >= 2 for r in rows),
                safe_drive_2deg_episodes=sum(not r['unsafe'] and r['peak_drive_response_deg'] >= 2 for r in rows),
                index_driven_successes=sum(r['success'] and not r['unsafe'] and r['peak_drive_response_deg'] >= 2
                                           for r in rows))


def rank(result):
    return (result['successes'], -result['unsafe'], result['safe_drive_2deg_episodes'],
            result['index_driven_successes'], result['progress'], result['natural'])


def panel(slot, cfg, k, checkpoint, seed_batches, fd, frozen_source=False):
    rows = []
    for batch in seed_batches:
        req = request(cfg, 'validate', batch, k, checkpoint)
        if frozen_source:
            req['transfer'] = cfg['origin']
            req['paired_validation' if cfg['mode'] == 'compare' else 'fresh_source_validation'] = True
        group, _ = wave(slot/f'wave_{len(rows):04d}', req, fd)
        rows.extend(group)
    return score(rows)


def batches(seeds, lanes):
    return [seeds[i:i+lanes] for i in range(0, len(seeds), lanes)]


def baseline_rows(rows, evidence):
    """Reconstruct the v4 episode counters from the verified v3 physical readbacks."""
    from contact_hold90_cuda_domain import CONTACT_IMPULSE
    by_seed = {r['seed']: dict(r, drive_frames=0, distal_contact_frames=0, qualified_index_frames=0,
                               peak_index_driven_deg=0., peak_drive_response_deg=0.) for r in rows}
    first_contact, previous_sample, drive_start = {}, {}, {}
    with evidence.open() as handle:
        for line in handle:
            sample = json.loads(line)
            seed, angle, step = sample['seed'], sample['face_deg'], sample['step']
            row = by_seed[seed]
            qualified, drive = bool(sample['qualified_index_frame']), bool(sample['drive_command'])
            row['drive_frames'] += int(drive)
            row['distal_contact_frames'] += int(sample['index_distal_impulse_Ns'] > CONTACT_IMPULSE)
            row['qualified_index_frames'] += int(qualified)
            if qualified:
                start = first_contact.setdefault(seed, angle)
                net = (angle-start+180.) % 360. - 180.
                row['peak_index_driven_deg'] = max(row['peak_index_driven_deg'],
                                                   min(net, sample['signed_index_driven_deg']))
            last = previous_sample.get(seed)
            isolated = qualified and sample['other_digit_target_impulse_Ns'] <= CONTACT_IMPULSE
            if drive and isolated and last and last[0] == step-1 and last[1]:
                start = drive_start.setdefault(seed, last[2])
                net = (angle-start+180.) % 360. - 180.
                row['peak_drive_response_deg'] = max(row['peak_drive_response_deg'], net)
            else:
                drive_start.pop(seed, None)
            previous_sample[seed] = (step, isolated, angle)
    return list(by_seed.values())


def original_validation(origin):
    """Read the actual accepted seed blocks and the matching frozen policy from the old validation."""
    folder = Path(origin['folder'])
    result = read(folder/'result.json')
    cfg = read(folder.parent/'config.json')
    if not result.get('validation_after_update'):
        raise ValueError('Selected source iteration has no committed validation')
    groups, rows = [], []
    for slot in sorted((folder/'validate').glob('wave_*')):
        req = read(slot/'request.json')
        previous.validate_request(req)
        if (slot.name != f'wave_{len(rows):04d}' or req['role'] != 'validate'
                or req['iteration'] != origin['iteration'] or req['software'] != origin['software']
                or (req['checkpoint'], req['checkpoint_sha']) != (origin['checkpoint'], origin['sha'])
                or req['policy_seed'] != cfg['rng']):
            raise ValueError('Source validation is not a paired evaluation of this checkpoint')
        verified, _ = previous.accepted(slot, req)
        attempt = accepted_report(slot, req).parent
        rows.extend(baseline_rows(verified, attempt/'turn_contact_evidence.jsonl'))
        groups.append(req['seeds'])
    if (len(rows) != cfg['validation_seeds'] or len({r['seed'] for r in rows}) != len(rows)
            or stats(rows) != result['validation_after_update']):
        raise ValueError('Source validation receipts disagree with committed result')
    return groups, score(rows), cfg['rng']


def frozen_evaluation(home, cfg, fd):
    origin = cfg['origin']
    folder = home/f"iteration_{origin['iteration']:04d}"
    result = panel(folder/'validate', cfg, origin['iteration'], origin['checkpoint'],
                   cfg['frozen_seed_batches'], fd, frozen_source=True)
    record = dict(mode=cfg['mode'], checkpoint=origin['checkpoint'], checkpoint_sha=origin['sha'],
                  iteration=origin['iteration'], seeds=sum(cfg['frozen_seed_batches'], []),
                  updated=False, independent=False, result=result)
    if cfg['mode'] == 'compare':
        record['baseline'] = cfg['baseline']
    save(home/'frozen_validation.json', record)
    save(home/'state.json', dict(status='frozen_validation_complete', iteration=origin['iteration'],
                                checkpoint=origin['checkpoint'], checkpoint_sha=origin['sha']))
    print(json.dumps(record, ensure_ascii=False), flush=True)


def campaign(home, cfg, fd):
    origin = cfg['origin']
    state_file = home/'state.json'
    if not state_file.exists():
        save(state_file, dict(iteration=origin['iteration'], checkpoint=origin['checkpoint'],
                              checkpoint_sha=origin['sha'], best_validation=None, status='running'))
    state = read(state_file)
    if state['status'] != 'running':
        print('Finished', state['status'])
        return
    for k in range(state['iteration']+1, cfg['iterations']+1):
        if fingerprint() != cfg['source_sha256']:
            raise ValueError('Sources changed during campaign')
        folder = home/f'iteration_{k:04d}'
        begin = 47_000_000 + (k-1)*32
        req = request(cfg, 'train', list(range(begin, begin+cfg['lanes'])), k, state['checkpoint'])
        if req['checkpoint_sha'] != state['checkpoint_sha']:
            raise ValueError('Committed parent checkpoint changed')
        rows, learning = wave(folder/'train', req, fd)
        ck, sha = learning['output_checkpoint'], learning['output_checkpoint_sha']
        result = dict(iteration=k, training_rollouts_before_update=score(rows), learning=learning,
                      end_reasons=dict(Counter(r['end_reason'] for r in rows)))
        if k in (origin['iteration']+1, cfg['iterations']) or k % cfg['validate_every'] == 0:
            begin = 67_000_000 + (k-1)*32
            seeds = list(range(begin, begin+cfg['validation_seeds']))
            valid = panel(folder/'validate', cfg, k, ck, batches(seeds, cfg['lanes']), fd)
            result['validation_after_update'] = valid
            if state['best_validation'] is None or rank(valid) > rank(state['best_validation']):
                state['best_validation'] = valid
                save(home/'best_candidate.json', dict(iteration=k, checkpoint=ck, sha=sha, validation=valid,
                                                      qualified=False, source_sha256=cfg['source_sha256']))
            if valid['unsafe'] == 0 and valid['index_driven_successes'] >= math.ceil(.875*len(seeds)):
                held = final_seeds(home, ROOT/'runs/natural_face90_test_registry')
                test = panel(folder/'independent_test', cfg, k, ck, batches(held, cfg['lanes']), fd)
                result['independent_test'] = test
                save(home/'final_test.json', dict(checkpoint=ck, sha=sha, seeds=held, results=test))
                if test['unsafe'] == 0 and test['index_driven_successes'] >= 28:
                    save(home/'qualified_policy.json', dict(checkpoint=ck, sha=sha, results=test,
                         vision=False, simulation_only=True, source_sha256=cfg['source_sha256']))
                    state['status'] = 'qualified_15deg_state_specialist'
                else:
                    state['status'] = 'independent_test_not_met'
        if k == cfg['iterations'] and state['status'] == 'running':
            state['status'] = 'iteration_limit'
        state.update(iteration=k, checkpoint=ck, checkpoint_sha=sha)
        save(folder/'result.json', result)
        save(state_file, state)
        print('iteration', k, state['status'], 'train', result['training_rollouts_before_update'],
              'validation', result.get('validation_after_update'), flush=True)
        if state['status'] != 'running':
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--continue-from', type=Path, help='v3 contact-turn run or explicit committed iteration')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--lanes', type=int, choices=(2, 4, 8, 16, 32))
    parser.add_argument('--validate-every', type=int)
    parser.add_argument('--validation-seeds', type=int, choices=(8, 16, 32))
    parser.add_argument('--rng', type=int)
    parser.add_argument('--resume', action='store_true', help='Reuse the saved mode, options and pinned source')
    parser.add_argument('--check', action='store_true', help='Read-only verification, no simulation or run creation')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--compare-validation', dest='mode', action='store_const', const='compare',
                       help='Freeze the source checkpoint and replay the exact accepted validation seeds')
    modes.add_argument('--fresh-validation', dest='mode', action='store_const', const='fresh',
                       help='Evaluate the frozen source checkpoint on separate development seeds')
    args = parser.parse_args()
    home = args.run_dir.resolve()
    if home.parent != ROOT/'runs' or not re.fullmatch(r'r27_recipe[A-Za-z0-9_-]*', home.name):
        parser.error('Use a separate runs/r27_recipe... experiment directory')
    if args.resume and not (home/'config.json').is_file():
        parser.error('No configured run to resume')
    if home.exists() and not args.resume and not args.check:
        parser.error('Destination exists. Use --resume for an existing run or choose a new directory.')
    prior = read(home/'config.json') if args.resume else {}
    if prior and prior['schema'] != SCHEMA:
        parser.error('Expected a v4 centered-contact run')
    source = args.continue_from.resolve() if args.continue_from else None
    if prior:
        pinned = Path(prior['origin']['folder'])
        if source and source not in (pinned, pinned.parent):
            parser.error('Cannot change origin when resuming')
        source = pinned
    if not source:
        parser.error('Provide --continue-from with a v3 contact-turn run or committed iteration')
    if home in (source, source.parent):
        parser.error('Source and destination must be different')
    defaults = dict(iterations=200, lanes=32, validate_every=10, validation_seeds=16, rng=327, mode='train')
    cfg = {key: (prior.get(key, default) if getattr(args, key) is None else getattr(args, key))
           for key, default in defaults.items()}
    if not (1 <= cfg['validate_every'] <= cfg['iterations'] <= 2000 and 0 <= cfg['rng'] < 2**31):
        parser.error('Invalid iteration count, validation cadence or policy seed')
    with (nullcontext() if args.check else idle_source(source)):
        pins = fingerprint()
        origin = transfer_origin(source, pins)
        if cfg['mode'] == 'train' and cfg['iterations'] <= origin['iteration']:
            parser.error('--iterations must exceed the committed source iteration')
        cfg.update(schema=SCHEMA, parent_sha=PARENT, source_sha256=pins, origin=origin,
                   smoke=cfg['mode'] != 'train')
        if cfg['mode'] == 'compare':
            groups, baseline, source_rng = original_validation(origin)
            if (max(map(len, groups)) > cfg['lanes'] or
                    (args.validation_seeds is not None and sum(map(len, groups)) != args.validation_seeds)
                    or (args.rng is not None and args.rng != source_rng)):
                parser.error('Paired evaluation must preserve source batch widths, seed count and policy rng')
            cfg.update(frozen_seed_batches=groups, baseline=baseline, rng=source_rng,
                       validation_seeds=sum(map(len, groups)))
        if cfg['mode'] == 'fresh':
            begin = 68_000_000 + (origin['iteration']-1)*32
            seeds = list(range(begin, begin+cfg['validation_seeds']))
            cfg['frozen_seed_batches'] = batches(seeds, cfg['lanes'])
        from natural_face90_v5_worker import inputs
        from natural_face90_recipe_frames import check_source_scene
        from natural_face90_recipe_contacts import check_capacity_source
        scene, *_ = inputs(dict(role='smoke', seeds=[20_000_000, 20_000_001],
                           parameters=[ZERO, ZERO], target_deg=15, parent_sha=PARENT))
        check_source_scene(scene)
        lanes = max(map(len, cfg['frozen_seed_batches'])) if 'frozen_seed_batches' in cfg else cfg['lanes']
        cfg['raw_contact_capacity'] = check_capacity_source(lanes)
        if prior and cfg != prior:
            parser.error('Resume requires identical sources and settings')
        if args.check:
            print('CHECK OK', cfg['mode'], origin['folder'], origin['sha'], 'lanes', lanes, flush=True)
            return
        if not args.resume:
            home.mkdir(parents=True, exist_ok=False)
        with exclusive(home/'campaign.lock') as fd:
            if not args.resume:
                save(home/'config.json', cfg)
            if cfg['mode'] == 'train':
                campaign(home, cfg, fd)
            else:
                frozen_evaluation(home, cfg, fd)


if __name__ == '__main__':
    main()
