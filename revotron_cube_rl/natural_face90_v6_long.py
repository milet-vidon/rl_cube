"""Continue an accepted guided PPO checkpoint in a separate contact-turn experiment."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from collections import Counter

import natural_face90_recipe_long as guided
from natural_face90_recipe_long import (
    HERE, ROOT, PARENT, ZERO, available_gpu, digest, exclusive, final_seeds,
    read, request, save, stats, verify_result, worker_failure, worker_software,
)

SCHEMA = 'natural_face90_recipe_contact_turn_ppo_v3'
FILES = (*guided.FILES, 'natural_face90_v6_long.py', 'natural_face90_v6_worker.py',
         'natural_face90_turn_guide.py', 'natural_face90_turn_contacts.py', 'run_natural_face90_turn_long.sh')


def fingerprint():
    return {name: digest(HERE/name) for name in FILES}


def check_source_idle(source):
    """Read the live campaign lock, which spans training, validation and GPU waits."""
    source = source.resolve()
    parent = source.parent if re.fullmatch(r'iteration_\d{4}',source.name) else source
    with (parent/'campaign.lock').open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('源训练仍在运行，请在原终端退出后重试。新任务尚未启动。') from exc


def verify_turn_report(report, req):
    rows, learning = verify_result(report, req)
    evidence = report.parent/'turn_contact_evidence.jsonl'
    if not evidence.is_file() or learning.get('turn_evidence_sha256') != digest(evidence):
        raise ValueError('Missing or changed turn-contact evidence')
    return rows, learning


def accepted(slot, req):
    """Both origin transfers and local retries must reference a verified completed wave."""
    receipt = read(slot/'accepted.json')
    if not re.fullmatch(r'attempt_[a-f0-9]{8,32}', receipt['attempt']):
        raise ValueError('Invalid accepted attempt name')
    report = (slot/receipt['attempt']/'report.json').resolve()
    if (report.parent.parent != slot.resolve() or read(slot/'request.json') != req
            or receipt['request_sha'] != digest(slot/'request.json')
            or receipt['report_sha'] != digest(report)):
        raise ValueError('Changed accepted PPO wave')
    if req['recipe_schema'] == SCHEMA:
        return verify_turn_report(report, req)
    if req['recipe_schema'] == guided.GUIDED_SCHEMA:
        return verify_result(report, req)
    raise ValueError('Unsupported accepted request schema')


def transfer_origin(path, pins):
    path = path.resolve()
    # A live parent can finish a new round while we check. Snapshot the last
    # committed iteration once and pin that immutable folder in our config.
    explicit = re.fullmatch(r'iteration_(\d{4})', path.name)
    parent = path.parent if explicit else path
    cfg = read(parent/'config.json')
    state = read(parent/'state.json')
    iteration = int(explicit.group(1)) if explicit else state['iteration']
    if (type(iteration) is not int or not 1 <= iteration <= state['iteration']
            or cfg['schema'] != guided.GUIDED_SCHEMA or cfg['parent_sha'] != PARENT
            or cfg['source_sha256'] != {name:pins[name] for name in guided.FILES}):
        raise ValueError('Expected a committed original guided PPO iteration with matching source hashes')
    folder = parent/f'iteration_{iteration:04d}'
    result = read(folder/'result.json')
    slot = folder/'train'
    req = read(slot/'request.json')
    guided.validate_request(req)
    if (result['iteration'] != iteration or req['iteration'] != iteration or req['role'] != 'train'
            or req['parent_sha'] != PARENT or req['software'] != worker_software(cfg['source_sha256'])):
        raise ValueError('Guided source report is for a different iteration or policy')
    _, learning = accepted(slot, req)
    if (learning != result['learning']
            or (iteration == state['iteration'] and state['checkpoint'] != learning['output_checkpoint'])):
        raise ValueError('Guided checkpoint disagrees with committed result')
    return dict(folder=str(folder), iteration=iteration, checkpoint=learning['output_checkpoint'],
                sha=learning['output_checkpoint_sha'], schema=cfg['schema'], software=req['software'])


def validate_request(req):
    if (req.get('recipe_schema') != SCHEMA or req.get('role') not in ('train','validate')
            or req.get('parent_sha') != PARENT or not req.get('checkpoint')
            or not re.fullmatch(r'[0-9a-f]{64}', req.get('checkpoint_sha') or '')):
        raise ValueError('Invalid contact-turn request or missing checkpoint')
    # Common PPO role, parameter, and transfer checks are shared with the pinned runner.
    guided.validate_request(dict(req, recipe_schema=guided.GUIDED_SCHEMA))
    origin = req.get('transfer')
    if origin:
        prior = {name:req['software'][name] for name in guided.FILES if name.endswith('.py')}
        if origin.get('schema') != guided.GUIDED_SCHEMA or origin.get('software') != prior:
            raise ValueError('Invalid guided checkpoint transfer contract')


def wave(slot, req, fd):
    validate_request(req)
    slot.mkdir(parents=True, exist_ok=True)
    plan = slot/'request.json'
    if plan.exists() and read(plan) != req:
        raise ValueError(f'Changed wave request: {plan}')
    if not plan.exists():
        save(plan, req)
    if (slot/'accepted.json').exists():
        return accepted(slot, req)
    attempt = slot/('attempt_'+uuid.uuid4().hex[:12])
    with available_gpu() as fds, attempt.with_suffix('.log').open('x') as log:
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(fds[0]), CUBE_DASH_ISAAC_FD=str(fds[1]))
        if digest(Path(req['checkpoint'])) != req['checkpoint_sha']:
            raise ValueError('Parent checkpoint changed while waiting for GPU')
        print('Contact-turn PPO', req['role'], 'iteration=', req['iteration'], 'log=', log.name, flush=True)
        cmd = [sys.executable, '-B', '-u', str(HERE/'natural_face90_v6_worker.py'),
               '--request', str(plan), '--out', str(attempt)]
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(fd,*fds))
        if result.returncode:
            raise worker_failure(result.returncode, attempt, log.name)
    report = attempt/'report.json'
    rows, learning = verify_turn_report(report, req)
    save(slot/'accepted.json', dict(attempt=attempt.name, request_sha=digest(plan), report_sha=digest(report)))
    return rows, learning


def panel(slot, cfg, k, ck, seeds, fd):
    rows = []
    for offset in range(0,len(seeds),cfg['lanes']):
        req = request(cfg, 'validate', seeds[offset:offset+cfg['lanes']], k, ck)
        batch, _ = wave(slot/f'wave_{offset:04d}', req, fd)
        rows.extend(batch)
    return stats(rows)


def rank(score):
    return (score['successes'], -score['unsafe'], score['progress'], score['natural'])


def campaign(home, cfg, fd):
    origin = cfg['origin']
    f = home/'state.json'
    if not f.exists():
        save(f, dict(iteration=origin['iteration'], checkpoint=origin['checkpoint'],
                     checkpoint_sha=origin['sha'], best_validation=None, status='running'))
    state = read(f)
    if state['status'] != 'running':
        print('Finished:', state['status'])
        return
    for k in range(state['iteration']+1, cfg['iterations']+1):
        if fingerprint() != cfg['source_sha256']:
            raise ValueError('Contact-turn sources changed during experiment')
        folder = home/f'iteration_{k:04d}'
        # Disjoint from the guided v2 development seeds even when restarting at an older iteration.
        begin = 46_000_000 + (k-1)*32
        req = request(cfg, 'train', list(range(begin, begin+cfg['lanes'])), k, state['checkpoint'])
        if req['checkpoint_sha'] != state['checkpoint_sha']:
            raise ValueError('Committed parent checkpoint changed')
        rows, learning = wave(folder/'train', req, fd)
        ck, ck_sha = learning['output_checkpoint'], learning['output_checkpoint_sha']
        result = dict(iteration=k, training_rollouts_before_update=stats(rows), learning=learning,
                      end_reasons=dict(Counter(r['end_reason'] for r in rows)))
        if k == origin['iteration']+1 or k % cfg['validate_every'] == 0 or k == cfg['iterations']:
            begin = 66_000_000 + (k-1)*32
            valid = panel(folder/'validate', cfg, k, ck, list(range(begin,begin+cfg['validation_seeds'])), fd)
            result['validation_after_update'] = valid
            if state['best_validation'] is None or rank(valid) > rank(state['best_validation']):
                state['best_validation'] = valid
                save(home/'best_candidate.json', dict(iteration=k, checkpoint=ck, sha=ck_sha, validation=valid,
                                                      qualified=False, source_sha256=cfg['source_sha256']))
            if valid['unsafe'] == 0 and valid['successes'] >= math.ceil(.875*cfg['validation_seeds']):
                seeds = final_seeds(home, ROOT/'runs/natural_face90_test_registry')
                test = panel(folder/'independent_test', cfg, k, ck, seeds, fd)
                result['independent_test'] = test
                save(home/'final_test.json', dict(checkpoint=ck, sha=ck_sha, seeds=seeds, results=test))
                if test['unsafe'] == 0 and test['successes'] >= 28:
                    save(home/'qualified_policy.json',dict(checkpoint=ck, sha=ck_sha, results=test,
                         vision=False, simulation_only=True, source_sha256=cfg['source_sha256']))
                    state['status'] = 'qualified_15deg_state_specialist'
                else:
                    state['status'] = 'independent_test_not_met'
        if k == cfg['iterations'] and state['status']=='running':
            state['status'] = 'iteration_limit'
        state.update(iteration=k, checkpoint=ck, checkpoint_sha=ck_sha)
        save(folder/'result.json', result)
        save(f, state)
        print('iteration', k, state['status'], 'train', result['training_rollouts_before_update'],
              'validation', result.get('validation_after_update'), flush=True)
        if state['status'] != 'running':
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--continue-from', type=Path, help='Guided run (latest committed round) or iteration folder')
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--lanes', type=int, choices=(2,4,8,16,32), default=32)
    p.add_argument('--validate-every', type=int, default=10)
    p.add_argument('--validation-seeds', type=int, choices=(8,16,32), default=16)
    p.add_argument('--rng', type=int, default=327)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true', help='Validate the source and inputs; no rollout')
    args = p.parse_args()
    if not (1 <= args.validate_every <= args.iterations <= 2000 and 0 <= args.rng < 2**31):
        p.error('Invalid iteration count, cadence or policy seed')
    home = args.run_dir.resolve()
    if home.parent != ROOT/'runs' or not re.fullmatch(r'r27_recipe[A-Za-z0-9_-]*', home.name):
        p.error('Use a separate runs/r27_recipe... directory for this experiment')
    cfg = {k:getattr(args,k) for k in ('iterations','lanes','validate_every','validation_seeds','rng')}
    pins = fingerprint()
    source = args.continue_from
    stored = read(home/'config.json') if args.resume else None
    if stored is not None:
        pinned = Path(stored['origin']['folder'])
        if source and source.resolve() not in (pinned, pinned.parent):
            p.error('Resume cannot change the pinned guided source')
        source = pinned
    if not source:
        p.error('Provide --continue-from for a new experiment')
    if not args.check:
        check_source_idle(source)
    origin = transfer_origin(source, pins)
    if origin['iteration'] >= args.iterations:
        p.error('--iterations must exceed the source iteration')
    if home == Path(origin['folder']).parent:
        p.error('Source and destination are the same experiment')
    from natural_face90_v5_worker import inputs
    from natural_face90_recipe_frames import check_source_scene
    from natural_face90_recipe_contacts import check_capacity_source
    scene, *_ = inputs(dict(role='smoke', seeds=[20_000_000,20_000_001],
                            parameters=[ZERO,ZERO], target_deg=15, parent_sha=PARENT))
    check_source_scene(scene)
    cfg.update(schema=SCHEMA, parent_sha=PARENT, source_sha256=pins, origin=origin, smoke=False,
               raw_contact_capacity=check_capacity_source(args.lanes))
    if stored is not None and stored != cfg:
        p.error('Resume requires the saved experiment configuration and exact sources')
    if args.check:
        print('CHECK OK: source', origin['folder'], 'checkpoint', origin['sha'],
              'first contact-turn iteration', origin['iteration']+1)
        return
    if not args.resume:
        home.mkdir(parents=True, exist_ok=False)
    with exclusive(home/'campaign.lock') as fd:
        if not args.resume:
            save(home/'config.json',cfg)
        campaign(home,cfg,fd)


if __name__ == '__main__':
    main()
