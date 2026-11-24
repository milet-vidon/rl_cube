"""Manual neutral-controller comparison and gated +15° PPO for Cartesian pad increments."""
import argparse
import json
import re
from pathlib import Path

from contact_hold90_cuda_domain import _specialize
import natural_face90_pad_preshape_probe as preshape
from natural_face90_hover_d_metrics import group, read_episodes
from natural_face90_placo_client import self_check


wrist, anchor, points, v7 = preshape.wrist, preshape.anchor, preshape.points, preshape.v7
worker_software = v7.worker_software
SCHEMA = 'natural_face90_recipe_cartesian_ppo_v1'
PROBE_SCHEMA = 'natural_face90_cartesian_neutral_v1'
SUMMARY = 'cartesian_probe_summary.json'
PROBE_HOME = points.RUNS / 'r27_recipe_cartesian_probe'
TRAIN_HOME = points.RUNS / 'r27_recipe_cartesian_ppo15'
TRAIN_BASE, VALIDATION_BASE = 50_000_000, 68_200_000
PAIRED_SEEDS = points.hover.ab.SEEDS
FILES = tuple(dict.fromkeys((*points.FILES, *anchor.FILES, *wrist.FILES, *preshape.FILES,
    'natural_face90_cartesian.py', 'natural_face90_cartesian_worker.py', 'natural_face90_cartesian_policy.py',
    'natural_face90_cartesian_control.py', 'natural_face90_placo_client.py', 'natural_face90_placo.py',
    'natural_face90_v5_regression.json', 'run_natural_face90_cartesian.sh')))


def fingerprint():
    return {name: v7.digest(points.HERE / name) for name in FILES}


def validate_request(req):
    seeds, k = req['seeds'], req['iteration']
    role = req['role']
    if (req['recipe_schema'] != SCHEMA or req['parent_sha'] != v7.PARENT or req['target_deg'] != 15
            or req['contact_point_offset_deg'] != 7 or req['pad_preshape_offset_deg'] != -6
            or role not in ('train', 'validate') or req['learn'] is not (role == 'train')
            or type(k) is not int or not 0 <= k <= 2000 or not 2 <= len(seeds) <= 32
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or req['parameters'] != [v7.ZERO] * len(seeds)
            or type(req['policy_seed']) is not int or not 0 <= req['policy_seed'] < 2**31
            or 'transfer' in req or 'promote_from_15' in req or not isinstance(req.get('placo'), dict)):
        raise ValueError('Unexpected Cartesian policy, task, lane layout or iteration')
    neutral = k == 0
    if req.get('paired_validation', False) is not neutral:
        raise ValueError('Pairing belongs to the neutral controller wave')
    if neutral:
        if role != 'validate' or seeds != PAIRED_SEEDS:
            raise ValueError('Neutral evaluation requires the original paired seeds')
    elif role == 'train':
        start = TRAIN_BASE + (k - 1)*32
        if seeds != list(range(start, start + len(seeds))):
            raise ValueError('Unexpected Cartesian training seeds')
    else:
        begin = VALIDATION_BASE + (k - 1)*32
        if not (all(begin <= s < begin+32 for s in seeds) or all(69_000_000 <= s < 71_000_000 for s in seeds)):
            raise ValueError('Unexpected Cartesian evaluation seeds')
    initial = neutral or (k == 1 and role == 'train')
    if initial:
        if req['checkpoint'] is not None or req['checkpoint_sha'] is not None:
            raise ValueError('Initial Cartesian actor must start from its seeded, zero-output initialization')
    elif not req['checkpoint'] or not re.fullmatch('[a-f0-9]{64}', req['checkpoint_sha'] or ''):
        raise ValueError('Missing Cartesian policy checkpoint')
    solver = req['placo']
    if (solver.get('backend') != 'placo' or solver.get('version') != '0.9.23'
            or solver.get('joints') != [f'Joint{i}_R' for i in range(1, 8)]
            or solver.get('position_frame') != 'calibrated_right_index_pad' or solver.get('masked_joints') != 49
            or solver.get('fixed_base') is not True or abs(solver['solver_dt'] - 1/30) > 1e-9):
        raise ValueError('Unexpected Cartesian arm solver or joint contract')
    if (req['software'] != worker_software(fingerprint()) or
            wrist.geometry_hash(req['pad_orientation_geometry']) != req['geometry_sha256'] or
            not re.fullmatch('[a-f0-9]{64}', req['scene_sha256'])):
        raise ValueError('Changed Cartesian software, contact geometry or scene identity')


def request(cfg, role, seeds, k, checkpoint):
    # The v7 campaign uses origin to initialize its iteration counter. A new
    # Cartesian policy always starts at zero and has no legacy transfer weights.
    req = dict(recipe_schema=SCHEMA, role=role, learn=role=='train', parent_sha=v7.PARENT, target_deg=15,
               parameters=[v7.ZERO.copy() for _ in seeds], seeds=seeds, iteration=k, policy_seed=cfg['rng'],
               software=worker_software(cfg['source_sha256']), checkpoint=checkpoint,
               checkpoint_sha=v7.digest(Path(checkpoint)) if checkpoint else None,
               contact_point_offset_deg=7, pad_preshape_offset_deg=-6,
               geometry_sha256=cfg['geometry_sha256'], scene_sha256=cfg['scene_sha256'],
               pad_orientation_geometry=cfg['pad_orientation_geometry'], placo=cfg['placo'])
    if k == 0:
        req['paired_validation'] = True
    return req


def verify_report(report, req):
    from natural_face90_cartesian_control import verify_frame
    from natural_face90_cartesian_worker import cartesian_receipt
    rows, learning = points.verify_report(report, req)
    if (v7.read(report)['domain_receipt']['scene_sha256'] != req['scene_sha256']
            or learning.get('cartesian') != cartesian_receipt(req)
            or learning.get('placo') != req['placo']
            or learning['pad_orientation'] != learning['pad_preshape']['orientation']):
        raise ValueError('Unexpected scene, controller interface or preshape evidence')
    trace = report.parent/'turn_contact_evidence.jsonl'
    preshape.policy.verify_receipt(learning['pad_preshape'], trace, req)
    with trace.open() as frames:
        for line in frames:
            row = json.loads(line)
            verify_frame(row, req['target_deg'])
            if req['iteration'] == 0 and any(row[f'pre_cartesian_latent_{i}'] != 0. for i in range(3)):
                raise ValueError('The paired controller baseline must have zero policy output')
    return rows, learning


wave = _specialize(points.wave, [
    ("HERE / 'natural_face90_contact_point_worker.py'", "HERE / 'natural_face90_cartesian_worker.py'"),
    ("if v7.digest(Path(req['checkpoint']))", "if req['checkpoint'] and v7.digest(Path(req['checkpoint']))"),
], dict(validate_request=validate_request, verify_report=verify_report, fingerprint=fingerprint))


def panel(slot, cfg, k, checkpoint, batches, fd):
    rows = []
    for seeds in batches:
        rs, _ = wave(slot/f'wave_{len(rows):04d}', request(cfg, 'validate', seeds, k, checkpoint), fd)
        rows.extend(rs)
    return v7.score(rows)


_campaign = _specialize(v7.campaign, [
    ('47_000_000 + (k-1)*32', 'TRAIN_BASE + (k-1)*32'),
    ('67_000_000 + (k-1)*32', 'VALIDATION_BASE + (k-1)*32'),
], dict(TRAIN_BASE=TRAIN_BASE, VALIDATION_BASE=VALIDATION_BASE, request=request, wave=wave,
        panel=panel, fingerprint=fingerprint))


def campaign(home, cfg, fd):
    if (home/'state.json').is_file():
        state = v7.read(home/'state.json')
        k = state['iteration']
        if type(k) is not int or not 0 <= k <= cfg['iterations']:
            raise ValueError('Invalid Cartesian resume iteration')
        if k:
            folder = home/f'iteration_{k:04d}'
            slot = folder/'train'
            previous = v7.read(home/f'iteration_{k-1:04d}'/'result.json')['learning']['output_checkpoint'] if k > 1 else None
            first = TRAIN_BASE + (k-1)*32
            req = request(cfg, 'train', list(range(first, first+cfg['lanes'])), k, previous)
            _, learning = verify_report(v7.accepted_report(slot, req), req)
            if ((state['checkpoint'], state['checkpoint_sha']) !=
                    (learning['output_checkpoint'], learning['output_checkpoint_sha'])
                    or v7.read(folder/'result.json')['learning'] != learning):
                raise ValueError('Cartesian resume checkpoint differs from its committed training wave')
        elif state['checkpoint'] is not None or state['checkpoint_sha'] is not None or state['status'] != 'running':
            raise ValueError('Invalid Cartesian initial state')
    _campaign(home, cfg, fd)


def probe_result(slot, req, plan, original, previous):
    report = v7.accepted_report(slot, req)
    rows, learning = verify_report(report, req)
    episodes = read_episodes(report, req['seeds'], treatment=False)
    anchor.metrics._geometry(report.parent, 7, episodes, req['seeds'], learning)
    for episode, row in zip(episodes, rows, strict=True):
        episode['final_face_deg'] = row['final_face_deg'] if row['post_samples'] else None
    fixed = [e for e in episodes if e['seed'] in anchor.metrics.FIXED_SEEDS]
    candidate = dict(all_16=group(episodes), fixed_11=group(fixed), per_seed=episodes,
                     unsafe_seeds=[e['seed'] for e in episodes if e['unsafe']])
    return dict(schema=PROBE_SCHEMA, plan=plan, updated=False, independent=False,
                baseline=original, previous=previous, candidate=candidate,
                candidate_report=str(report), candidate_report_sha256=v7.digest(report),
                pad_orientation=learning['pad_orientation'], cartesian=learning['cartesian'], placo=learning['placo'],
                **preshape.paired_result(candidate, original, previous, learning['pad_preshape']))


def train_config(home, cfg, args):
    prior = v7.read(home/'config.json') if (home/'config.json').is_file() else {}
    settings = {}
    for key, default in dict(iterations=100, lanes=16, validate_every=5, validation_seeds=16).items():
        val = getattr(args, key)
        settings[key] = prior.get(key, default) if val is None else val
    if not 1 <= settings['validate_every'] <= settings['iterations'] <= 2000:
        raise ValueError('Invalid iteration count or validation cadence')
    return dict(cfg, **settings, schema=SCHEMA, origin=dict(iteration=0, checkpoint=None, sha=None),
                target_deg=15, contact_point_offset_deg=7, pad_preshape_offset_deg=-6, smoke=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check','probe','train'))
    parser.add_argument('--input-summary', type=Path, default=anchor.DEFAULT_HOME/anchor.SUMMARY)
    parser.add_argument('--probe-dir', type=Path, default=PROBE_HOME)
    parser.add_argument('--run-dir', type=Path, default=TRAIN_HOME, help='Cartesian +15° training directory')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--lanes', type=int, choices=(4,8,16,32))
    parser.add_argument('--validate-every', type=int)
    parser.add_argument('--validation-seeds', type=int, choices=(8,16,32))
    args = parser.parse_args(argv)
    archive, probe, home = args.input_summary.resolve(), args.probe_dir.resolve(), args.run_dir.resolve()
    for path in (args.probe_dir, args.run_dir):
        if (path.is_symlink() or path.resolve().parent != points.RUNS
                or not re.fullmatch(r'r27_recipe[A-Za-z0-9_-]+',path.resolve().name)):
            parser.error('Choose separate Cartesian experiment directories directly under runs/')
    if archive.parent in (home,probe) or probe == home:
        parser.error('Archive, paired probe and training directories must differ')
    source_plan, source, original, previous = preshape.prepare(archive, check_weights=False)
    orientation = source_plan['orientation_plan']
    solver_check = self_check()
    cfg = dict(rng=source['policy_seed'], source_sha256=fingerprint(), placo=solver_check['receipt'],
               geometry_sha256=orientation['geometry_sha256'], scene_sha256=orientation['scene_sha256'],
               pad_orientation_geometry=source['pad_orientation_geometry'])
    plan = dict(schema=PROBE_SCHEMA, seeds=PAIRED_SEEDS, rng=cfg['rng'], source_sha256=cfg['source_sha256'],
                input_summary=str(archive), input_summary_sha256=source_plan['input_summary_sha256'],
                geometry_sha256=cfg['geometry_sha256'], scene_sha256=cfg['scene_sha256'], placo=cfg['placo'])
    neutral = request(cfg, 'validate', PAIRED_SEEDS, 0, None)
    validate_request(neutral)
    if args.mode == 'check':
        from natural_face90_cartesian_worker import cartesian_receipt
        print('CPU CHECK OK', json.dumps(dict(cartesian=cartesian_receipt(neutral), placo=solver_check), ensure_ascii=False))
        return
    slot, summary = probe/'probe', probe/SUMMARY
    if args.mode == 'probe':
        if probe.exists() and (not (probe/'plan.json').is_file() or v7.read(probe/'plan.json') != plan):
            raise ValueError('Probe directory belongs to a different experiment')
        probe.mkdir(exist_ok=True)
        with v7.exclusive(probe/'campaign.lock') as fd:
            if (probe/'plan.json').exists() and v7.read(probe/'plan.json') != plan:
                raise ValueError('Paired probe plan changed')
            v7.save(probe/'plan.json', plan)
            wave(slot, neutral, fd)
            result = probe_result(slot, neutral, plan, original, previous)
            if summary.exists() and v7.read(summary) != result:
                raise ValueError('Previously saved neutral probe differs from its verified report')
            v7.save(summary, result)
            print(summary, 'paired_contact_gate_met', result['paired_contact_gate_met'], flush=True)
        return
    if not summary.is_file():
        parser.error('Run the Cartesian probe first to measure safe first contact and +2°')
    result = probe_result(slot, neutral, plan, original, previous)
    if v7.read(probe/'plan.json') != plan or v7.read(summary) != result or not result['paired_contact_gate_met']:
        raise ValueError('The Cartesian probe has not met the paired first-contact, +2° and full-group safety gate')
    cfg = train_config(home, dict(cfg, probe_dir=str(probe), probe_summary_sha256=v7.digest(summary)), args)
    if home.exists() and (not (home/'config.json').is_file() or v7.read(home/'config.json') != cfg):
        raise ValueError('Training directory belongs to another configuration')
    home.mkdir(exist_ok=True)
    with v7.exclusive(home/'campaign.lock') as fd:
        if (home/'config.json').exists() and v7.read(home/'config.json') != cfg:
            raise ValueError('Training inputs changed')
        v7.save(home/'config.json', cfg)
        campaign(home, cfg, fd)


if __name__ == '__main__':
    main()
