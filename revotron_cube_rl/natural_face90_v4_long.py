"""Search a bounded, closed-loop post-pickup policy using measured 90 mm PhysX episodes."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import uuid
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/brainco/rl_cube/github_project/revotron_cube_rl')
PARENT = '7144453d2425fce0f5d4293748b4bf71cb8afbca2ec6917624842e16b2b656a7'
FILES = ('natural_face90_v4_long.py', 'natural_face90_v4_controller.py', 'natural_face90_v4_metrics.py', 'natural_face90_v4_worker.py', 'run_natural_face90_v4_long.sh', 'natural_face90_v4_contacts.py', 'natural_face90_v4_clearance.py')
LOCK_NAMES = ('contact_hold_gpu_training.lock', 'isaac_gui_evaluation.lock')
PARAMS = 6
ZERO = [0.] * PARAMS


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint():
    return {name: digest(HERE / name) for name in FILES}


@contextmanager
def exclusive(path):
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'已有任务占用 {path.name}；该任务不受影响，空闲后用 --resume 重试') from e
        yield handle.fileno()


@contextmanager
def gpu():
    locks = ROOT.parent / 'work/teacher_optimization_20261103'
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive(locks / name)) for name in LOCK_NAMES]
        procs = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], text=True)
        for line in procs.splitlines():
            _, name, mib = [s.strip() for s in line.split(',', 2)]
            if name == '/usr/NX/bin/nxnode.bin' and Path(name).is_file() and Path(name).stat().st_uid == 0 and mib.isdigit() and int(mib) <= 768:
                continue
            raise RuntimeError('GPU 0 正被使用；请空闲后加 --resume 重试：' + line)
        yield handles


def check_report(rep, req):
    if (rep.get('schema') != 'natural_face90_rollout_v1' or rep['role'] != req['role'] or rep['parent_sha'] != PARENT
            or [r['seed'] for r in rep['rows']] != req['seeds']):
        raise ValueError('Worker report does not match request')
    if rep.get('reset_count') != 1 or rep.get('physics_stepped') is not True or rep.get('cross_lane_contacts') or rep.get('face') != 'nZ':
        raise ValueError('Missing continuous physical rollout evidence')
    if 'software' in req and rep.get('software') != req['software']:
        raise ValueError('Worker sources changed during evaluation')
    for r, params in zip(rep['rows'], req['parameters']):
        if r.get('parameters') != params:
            raise ValueError('Worker attributed a row to different controller parameters')
        if r['target_deg'] != req['target_deg'] or not isinstance(r['end_reason'], str) or not 0 <= r['prefix_steps'] <= r['steps']:
            raise ValueError('Inconsistent rollout counters')
        if r['steps'] == 0 and (r['success'] or r['natural_ready'] or r['turn_started'] or r.get('post_steps', 0) != 0 or not r['end_reason']):
            raise ValueError('Inconsistent zero-step episode')
        flags = ['unsafe', 'success', 'natural_ready', 'turn_started']
        if not all(type(r[key]) is bool for key in flags) or (r['turn_started'] and not r['natural_ready']) or (r['success'] and (r['unsafe'] or not r['turn_started'])):
            raise ValueError('Inconsistent phase or safety flags')
        for key in ('progress', 'posture_quality', 'right_contact', 'hold_fraction'):
            if not isinstance(r[key], (int, float)) or not 0 <= r[key] <= 1:
                raise ValueError('Invalid physical measurement: ' + key)
    return rep['rows']


def verified_rows(report, req):
    rep = read(report)
    archived_request = report.parent/'request.json'
    evidence = report.parent/'post_measurements.jsonl'
    if read(archived_request) != req or rep['request_sha256'] != digest(archived_request) or rep['post_evidence_sha256'] != digest(evidence) or rep['collision_evidence_sha256'] != digest(report.parent/'collision_events.jsonl'):
        raise ValueError('Measured evidence or request does not match rollout receipt')
    return check_report(rep, req)


def wave(slot, req, campaign_fd):
    slot.mkdir(parents=True, exist_ok=True)
    plan = slot / 'request.json'
    if plan.exists() and read(plan) != req:
        raise ValueError('Existing wave has a different request: ' + str(slot))
    if not plan.exists():
        save(plan, req)
    receipt = slot / 'accepted.json'
    if receipt.exists():
        saved = read(receipt)
        report = slot / saved['attempt'] / 'report.json'
        if saved['request_sha'] != digest(plan) or saved['report_sha'] != digest(report):
            raise ValueError('Changed accepted wave files')
        return verified_rows(report, req)
    dest = slot / ('attempt_' + uuid.uuid4().hex[:12])
    with gpu() as fds, (slot / (dest.name + '.log')).open('w') as log:
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(fds[0]), CUBE_DASH_ISAAC_FD=str(fds[1]), PYTHONUNBUFFERED='1')
        cmd = [sys.executable, '-B', '-u', str(HERE/'natural_face90_v4_worker.py'), '--request', str(plan), '--out', str(dest)]
        print('PhysX', req['role'], 'lanes=', len(req['seeds']), 'log=', log.name, flush=True)
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, pass_fds=(campaign_fd, *fds), env=env)
        if result.returncode:
            raise RuntimeError(f'PhysX exited {result.returncode}; inspect {log.name}')
    report = dest / 'report.json'
    rows = verified_rows(report, req)
    save(receipt, dict(attempt=dest.name, request_sha=digest(plan), report_sha=digest(report)))
    return rows


def evaluate(folder, role, candidates, panel, cfg, fd):
    if fingerprint() != cfg['source_sha256']:
        raise ValueError('Strategy source changed during this campaign; use a fresh run')
    software = {k: cfg['source_sha256'][k] for k in FILES if k.endswith('.py') and k != 'natural_face90_v4_long.py'}
    pairs = [(seed, params) for params in candidates for seed in panel]
    batch = cfg['lanes']
    rows = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i+batch]
        req = dict(role=role, parent_sha=PARENT, target_deg=15, software=software, seeds=[p[0] for p in chunk], parameters=[p[1] for p in chunk])
        rows.extend(wave(folder / f'wave_{i:04d}', req, fd))
    return [rows[i:i+len(panel)] for i in range(0, len(rows), len(panel))]


def stats(rows):
    safe = [r for r in rows if not r['unsafe']]
    n = len(rows)
    def credit(key, phase=None):
        return sum(r[key] for r in safe if phase is None or r[phase]) / n
    return dict(episodes=n, unsafe=n-len(safe), successes=sum(r['success'] for r in safe),
                natural=credit('natural_ready'), turn=credit('turn_started'), progress=credit('progress', 'turn_started'),
                contact=credit('right_contact', 'natural_ready'), posture=credit('posture_quality'), hold=credit('hold_fraction'))


def probe_summary(home, rows):
    """Development diagnostics, separate from safety-filtered candidate ranking."""
    pairs=Counter();phases=Counter()
    for receipt in sorted((home/'smoke').glob('wave_*/accepted.json')):
        attempt=receipt.parent/read(receipt)['attempt']
        for line in (attempt/'collision_events.jsonl').read_text().splitlines():
            event=json.loads(line);phases[event['phase']]+=1
            for pair in event['pairs']:
                pairs[pair['body_a']+' <-> '+pair['body_b']]+=1
    return dict(diagnostic_only=True,episodes=len(rows),
        prefix_held=sum(r['prefix_endpoint_held'] for r in rows),
        natural_observed=sum(r['natural_ready'] for r in rows),
        touched_target=sum(r['target_contact_steps']>0 for r in rows),
        unsafe=sum(r['unsafe'] for r in rows),successes=sum(r['success'] for r in rows),
        end_reasons=dict(Counter(r['end_reason'] for r in rows)),
        self_collision_phases=dict(phases),self_collision_pairs=dict(pairs.most_common()))


def rank(s):
    # An unsafe rollout earns no partial credit; at panel level safety takes precedence.
    return (-s['unsafe'], s['successes'], s['turn'], s['progress'], s['natural'], s['contact'], s['posture'], s['hold'])


def population(state, cfg, k):
    rng = random.Random(cfg['rng'] + 991*k)
    mu, sigma = state['mean'], state['std']
    return [state['incumbent'].copy(), mu.copy()] + [[min(1., max(-1., rng.gauss(x, t))) for x, t in zip(mu, sigma)]
                                      for _ in range(cfg['population']-2)]


def update_distribution(state, candidates, results):
    ordering = sorted(range(len(candidates)), key=lambda i:rank(results[i]), reverse=True)
    # Do not collapse the distribution or select an arbitrary direction on identical evidence.
    if rank(results[ordering[0]]) == rank(results[ordering[-1]]):
        return
    n = max(2, len(ordering)//4)
    for i in range(PARAMS):
        elite = [candidates[j][i] for j in ordering[:n]]
        state['mean'][i] = 0.35*state['mean'][i] + 0.65*statistics.mean(elite)
        state['std'][i] = max(.15, .4*state['std'][i] + .6*statistics.pstdev(elite))


def final_seeds(home, allocation_root):
    # Claim once per campaign and keep the panel out of all later development campaigns on this host.
    allocation_root.mkdir(parents=True, exist_ok=True)
    owner = str(home.resolve())
    with exclusive(allocation_root/'registry.lock'):
        f = allocation_root/'allocations.json'
        registry = read(f) if f.exists() else {}
        if owner not in registry:
            start = max(registry.values(), default=69_000_000-32) + 32
            if start+32 > 71_000_000:
                raise ValueError('Independent test seed pool exhausted')
            registry[owner] = start
            save(f,registry)
        start = registry[owner]
    return list(range(start,start+32))


def campaign(home, cfg, fd):
    statefile = home/'state.json'
    state = read(statefile) if statefile.exists() else dict(round=0, mean=ZERO.copy(), std=[.65]*PARAMS, incumbent=ZERO.copy(), stalled=0, flat=0, best_valid_progress=0., status='running')
    if state['status'] != 'running':
        print('Campaign finished:', state['status'], flush=True)
        return
    for k in range(state['round']+1, cfg['rounds']+1):
        folder = home/f'round_{k:03d}'
        # Use the same reset seeds for every candidate within each round.
        train = [43_000_000+(k-1)*256+j for j in range(cfg['seeds_per_candidate'])]
        valid = [63_000_000+(k-1)*256+j for j in range(cfg['validation_seeds'])]
        pop = population(state, cfg, k)
        scores = [stats(v) for v in evaluate(folder/'train', 'train', pop, train, cfg, fd)]
        best = max(range(len(pop)), key=lambda i:rank(scores[i]))
        challenger = pop[best]
        contenders = [state['incumbent']]
        if challenger != contenders[0]:
            contenders.append(challenger)
        vs = [stats(v) for v in evaluate(folder/'validate', 'validate', contenders, valid, cfg, fd)]
        winner = max(range(len(vs)), key=lambda i:rank(vs[i]))
        update_distribution(state, pop, scores)
        state.update(round=k, incumbent=contenders[winner])
        # Tiny orientation jitter after initial contact is not evidence that a long search is useful.
        state['stalled'] = state['stalled']+1 if max(s['progress'] for s in scores+vs) < 3/15 else 0
        if vs[winner]['progress'] >= state['best_valid_progress'] + 1/15:
            state.update(flat=0, best_valid_progress=vs[winner]['progress'])
        else:
            state['flat'] += 1
        result = dict(parameters=state['incumbent'], parent_sha=PARENT, target='nZ +15 deg',
                      validation=vs[winner], round=k, source_sha256=cfg['source_sha256'])
        save(folder/'selection.json', dict(candidates=pop, train=scores, contenders=contenders, validation=vs, selected=winner))
        save(home/'best_candidate.json', result)
        chosen = vs[winner]
        # Test only once; this panel is never used to adjust the policy.
        if chosen['successes'] >= math.ceil(.875*len(valid)) and chosen['unsafe'] == 0:
            panel = final_seeds(home, ROOT/'runs/natural_face90_test_registry')
            test = stats(evaluate(folder/'final_test', 'validate', [state['incumbent']], panel, cfg, fd)[0])
            save(home/'final_test.json', dict(**result, independent_test=test, seeds=panel))
            if test['unsafe']==0 and test['successes']>=28:
                save(home/'qualified_policy.json', dict(**result, independent_test=test))
                state['status'] = 'qualified_15deg'
            else:
                state['status'] = 'independent_test_not_met'
        elif state['stalled'] >= cfg['patience']:
            state['status'] = 'no_safe_layer_progress'
        elif state['flat'] >= cfg['patience']+2:
            state['status'] = 'validation_plateau'
        elif k == cfg['rounds']:
            state['status'] = 'round_limit'
        save(statefile, state)
        print('round', k, 'status', state['status'], 'validation', chosen, flush=True)
        if state['status'] != 'running':
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--population', type=int, default=8)
    p.add_argument('--seeds-per-candidate', type=int, default=4)
    p.add_argument('--validation-seeds', type=int, default=8)
    p.add_argument('--lanes', type=int, choices=(2,4,8,16,32), default=16)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--rng', type=int, default=327)
    p.add_argument('--resume', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='Eight historical R27 seeds, zero parameters; development diagnostic only.')
    mode.add_argument('--check', action='store_true', help='Verify parent, frozen inputs and sources; no training or write.')
    args = p.parse_args()
    if (not 1 <= args.rounds <= 2000 or not 4 <= args.population <= 128 or not 1 <= args.patience <= args.rounds
        or any(i not in (2,4,8,16,32) for i in (args.seeds_per_candidate,args.validation_seeds))):
        p.error('invalid search bounds; seed panels must have even power-of-two sizes')
    from natural_face90_v4_worker import inputs
    inputs(dict(role='smoke', seeds=[20_000_000,20_000_001], parameters=[ZERO,ZERO], target_deg=15, parent_sha=PARENT))
    cfg = {k:getattr(args,k) for k in ('rounds','population','seeds_per_candidate','validation_seeds','lanes','patience','rng','smoke')}
    cfg.update(parent_sha=PARENT, source_sha256=fingerprint(), schema='natural_face90_search_v4')
    if args.check:
        print('CHECK OK: frozen R27, 90 mm scene, training inputs and strategy files', flush=True)
        return
    home = args.run_dir.resolve()
    if home == ROOT/'runs' or (ROOT/'runs/contact_hold90_long_v2') in (home, *home.parents):
        p.error('use a separate, new experiment directory')
    if args.resume:
        if not home.is_dir() or not (home/'config.json').is_file():
            p.error('no configured campaign to resume')
    else:
        home.mkdir(parents=True, exist_ok=False)
    with exclusive(home/'campaign.lock') as fd:
        if args.resume:
            if read(home/'config.json') != cfg:
                raise ValueError('Resume requires identical settings and source pins')
        else:
            save(home/'config.json', cfg)
        if args.smoke:
            pair = list(range(20_000_000,20_000_008))
            rows = evaluate(home/'smoke','smoke',[ZERO],pair,cfg,fd)[0]
            save(home/'smoke.json',dict(diagnostic_only=True, summary=stats(rows), rows=rows))
            summary=probe_summary(home,rows)
            save(home/'diagnosis.json',summary)
            print('smoke diagnostics:',json.dumps(summary,ensure_ascii=False),flush=True)
        else:
            campaign(home,cfg,fd)


if __name__ == '__main__':
    main()
