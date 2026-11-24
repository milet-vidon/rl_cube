"""Train contact-to-turn residuals from scratch and transfer their verified policy to +15 degrees."""
import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
from types import FunctionType

from contact_hold90_cuda_domain import _specialize
import natural_face90_cartesian as base

v7, points = base.v7, base.points
worker_software = base.worker_software
CONTACT = 'natural_face90_cartesian_contact_curriculum_v1'
FACE = 'natural_face90_cartesian_contact_to15_v1'
CONTACT_HOME = points.RUNS/'r27_recipe_cartesian_contact_ppo'
FACE_HOME = points.RUNS/'r27_recipe_cartesian_contact_to15'
TRAIN_BASE = {'contact2': 50_100_000, 'face15': 50_200_000}
VALID_BASE = 68_400_000
SEEDS = base.PAIRED_SEEDS
FILES = (*base.FILES, 'natural_face90_cartesian_contact.py',
         'natural_face90_cartesian_contact_worker.py', 'run_natural_face90_cartesian_contact.sh')


def fingerprint():
    return {name: v7.digest(points.HERE/name) for name in FILES}


def seed_block(first, k, lanes):
    start = first + 32*(k-1)
    return list(range(start, start+lanes))


def request(cfg, role, seeds, k, checkpoint):
    req = base.request(cfg, role, seeds, k, checkpoint)
    stage = cfg['curriculum_stage']
    req.update(recipe_schema=cfg['schema'], curriculum_stage=stage)
    if stage == 'contact2' and role == 'validate':
        req['paired_validation'] = True
    if stage == 'face15' and role == 'train' and k == cfg['origin']['iteration']+1:
        req['contact_origin'] = dict(cfg['origin'])
    return req


# The same physical robot, Placo, calibrated contact point and original safety contract.
_validate_common = FunctionType(base.validate_request.__code__,
                               dict(base.validate_request.__globals__, fingerprint=fingerprint))


def validate_request(req):
    k, seeds, stage, role = req['iteration'], req['seeds'], req['curriculum_stage'], req['role']
    if (type(k) is not int or not 1 <= k <= 2000 or stage not in TRAIN_BASE
            or req['recipe_schema'] != (CONTACT if stage == 'contact2' else FACE)
            or role not in ('train', 'validate') or 'transfer' in req or 'promote_from_15' in req):
        raise ValueError('Invalid curriculum stage, schema or iteration')
    paired = stage == 'contact2' and role == 'validate'
    if req.get('paired_validation', False) is not paired:
        raise ValueError('Only the contact curriculum development panel uses paired seeds')
    if role == 'train':
        if seeds != seed_block(TRAIN_BASE[stage], k, len(seeds)):
            raise ValueError('Unexpected curriculum training seeds')
    elif paired:
        if seeds != SEEDS:
            raise ValueError('Expected all original 16 paired lanes together')
    else:
        dev = seed_block(VALID_BASE, k, 32)
        if not (all(s in dev for s in seeds) or all(69_000_000 <= s < 71_000_000 for s in seeds)):
            raise ValueError('Unexpected +15 evaluation seeds')
    origin = req.get('contact_origin')
    if 'contact_origin' in req and (stage != 'face15' or role != 'train' or k <= 1 or origin !=
            dict(iteration=k-1, checkpoint=req['checkpoint'], sha=req['checkpoint_sha'])):
        raise ValueError('Invalid contact-to-face transfer')
    if stage == 'face15' and k == 1:
        raise ValueError('+15 curriculum must start from verified contact weights')
    # Map the already-validated seed domain to the original checker so its
    # controller, checkpoint, geometry and software checks also run here.
    original_seeds = seed_block(base.TRAIN_BASE if role == 'train' else base.VALIDATION_BASE, k, len(seeds))
    if len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Invalid or duplicate physical lane seeds')
    _validate_common(dict(req, recipe_schema=base.SCHEMA, seeds=original_seeds, paired_validation=False))


def verify_report(report, req):
    from natural_face90_cartesian_contact_worker import verify_stage_report
    rows, learning = base.verify_report(report, req)
    verify_stage_report(report, req, rows, learning)
    return rows, learning


wave = _specialize(points.wave, [
    ("HERE / 'natural_face90_contact_point_worker.py'", "HERE / 'natural_face90_cartesian_contact_worker.py'"),
    ("if v7.digest(Path(req['checkpoint']))", "if req['checkpoint'] and v7.digest(Path(req['checkpoint']))"),
], dict(validate_request=validate_request, verify_report=verify_report, fingerprint=fingerprint))


def verified_baseline(probe):
    """Reproduce the original neutral comparison even when its recorded gate is false."""
    plan = v7.read(probe/'plan.json')
    archive = Path(plan['input_summary'])
    source_plan, source, original, previous = base.preshape.prepare(archive, check_weights=False)
    orientation = source_plan['orientation_plan']
    solver = base.self_check()['receipt']
    if (plan != dict(schema=base.PROBE_SCHEMA, seeds=SEEDS, rng=source['policy_seed'],
                     source_sha256=base.fingerprint(), input_summary=str(archive),
                     input_summary_sha256=source_plan['input_summary_sha256'],
                     geometry_sha256=orientation['geometry_sha256'], scene_sha256=orientation['scene_sha256'],
                     placo=solver)):
        raise ValueError('Neutral probe source, geometry or Placo changed')
    common = dict(rng=plan['rng'], geometry_sha256=plan['geometry_sha256'], scene_sha256=plan['scene_sha256'],
                  pad_orientation_geometry=source['pad_orientation_geometry'], placo=solver)
    req = base.request(dict(common, source_sha256=plan['source_sha256']), 'validate', SEEDS, 0, None)
    base.validate_request(req)
    summary = base.probe_result(probe/'probe', req, plan, original, previous)
    if v7.read(probe/base.SUMMARY) != summary:
        raise ValueError('Neutral summary differs from the accepted physical probe')
    return dict(common, probe_dir=str(probe), probe_summary_sha256=v7.digest(probe/base.SUMMARY),
                source_sha256=fingerprint()), summary


def paired_gate(report, req, neutral, learning):
    episodes = base.read_episodes(report, req['seeds'], treatment=False)
    base.anchor.metrics._geometry(report.parent, 7, episodes, SEEDS, learning)
    fixed = [r for r in episodes if r['seed'] in base.anchor.metrics.FIXED_SEEDS]
    candidate = dict(per_seed=episodes, unsafe_seeds=[r['seed'] for r in episodes if r['unsafe']])
    gate = base.preshape.paired_result(candidate, neutral['baseline'], neutral['candidate'], learning['pad_preshape'])
    return dict(gate, fixed_11=base.group(fixed), all_16=base.group(episodes),
                fixed_peak_drive_deg=max(r['peak_drive_response_deg'] for r in fixed),
                per_seed=[{k:r[k] for k in ('seed','first_contact','unsafe','safe_qualified_drive',
                                             'safe_drive_2deg','peak_drive_response_deg')} for r in episodes])


def train_req(cfg, k, checkpoint):
    seeds = seed_block(TRAIN_BASE[cfg['curriculum_stage']], k, cfg['lanes'])
    return request(cfg, 'train', seeds, k, checkpoint)


def verified_state(home, cfg):
    state = v7.read(home/'state.json')
    origin = cfg['origin']
    k = state['iteration']
    if type(k) is not int or not origin['iteration'] <= k <= cfg['iterations']:
        raise ValueError('Invalid committed curriculum iteration')
    expected = origin['checkpoint'], origin['sha']
    if k > origin['iteration']:
        previous = origin['checkpoint'] if k == origin['iteration']+1 else v7.read(
            home/f'iteration_{k-1:04d}'/'result.json')['learning']['output_checkpoint']
        folder = home/f'iteration_{k:04d}'
        slot = folder/'train'
        req = train_req(cfg, k, previous)
        validate_request(req)
        _, learning = verify_report(v7.accepted_report(slot, req), req)
        if v7.read(folder/'result.json')['learning'] != learning:
            raise ValueError('Saved iteration differs from its accepted training wave')
        expected = learning['output_checkpoint'], learning['output_checkpoint_sha']
    if (state['checkpoint'], state['checkpoint_sha']) != expected:
        raise ValueError('Saved campaign checkpoint disagrees with its accepted wave')
    return state


def certificate(cfg, state, report, gate):
    return dict(iteration=state['iteration'], checkpoint=state['checkpoint'], sha=state['checkpoint_sha'],
                schema=CONTACT, source_sha256=cfg['source_sha256'], probe_summary_sha256=cfg['probe_summary_sha256'],
                paired_gate=gate, validation_report_sha256=v7.digest(report), independent=False)


def verified_origin(home, common, neutral):
    cfg = v7.read(home/'config.json')
    if (cfg['schema'] != CONTACT or cfg['curriculum_stage'] != 'contact2'
            or cfg['origin'] != dict(iteration=0, checkpoint=None, sha=None)
            or any(cfg.get(k) != v for k,v in common.items())):
        raise ValueError('Contact training and +15 training must use the same verified probe and software')
    state = verified_state(home, cfg)
    if state['status'] != 'contact_ready':
        raise ValueError('The trained contact policy has not passed its paired development evaluation')
    k = state['iteration']
    folder = home/f'iteration_{k:04d}'
    req = request(cfg, 'validate', SEEDS, k, state['checkpoint'])
    validate_request(req)
    report = v7.accepted_report(folder/'validate'/'wave_0000', req)
    _, learn = verify_report(report, req)
    gate = paired_gate(report, req, neutral, learn)
    if (not gate['paired_contact_gate_met'] or v7.read(folder/'result.json')['paired_gate'] != gate
            or v7.read(home/'contact_ready.json') != certificate(cfg, state, report, gate)):
        raise ValueError('Contact checkpoint failed verification against its own deterministic paired rollout')
    return dict(iteration=k, checkpoint=state['checkpoint'], sha=state['checkpoint_sha'])


def contact_campaign(home, cfg, neutral, fd):
    path = home/'state.json'
    if not path.exists():
        v7.save(path, dict(iteration=0, checkpoint=None, checkpoint_sha=None, status='running'))
    state = verified_state(home, cfg)
    if state['status'] != 'running':
        return state['status']
    for k in range(state['iteration']+1, cfg['iterations']+1):
        folder = home/f'iteration_{k:04d}'
        req = train_req(cfg, k, state['checkpoint'])
        if req['checkpoint_sha'] != state['checkpoint_sha'] or fingerprint() != cfg['source_sha256']:
            raise ValueError('Changed curriculum checkpoint or source')
        rows, learn = wave(folder/'train', req, fd)
        ck = learn['output_checkpoint']
        state.update(iteration=k, checkpoint=ck, checkpoint_sha=learn['output_checkpoint_sha'])
        result = dict(iteration=k, training_rollouts_before_update=v7.score(rows), learning=learn,
                      end_reasons=dict(Counter(r['end_reason'] for r in rows)))
        if k == 1 or k == cfg['iterations'] or k % cfg['validate_every'] == 0:
            req = request(cfg, 'validate', SEEDS, k, ck)
            slot = folder/'validate'/'wave_0000'
            rows, learn = wave(slot, req, fd)
            report = v7.accepted_report(slot, req)
            gate = paired_gate(report, req, neutral, learn)
            result.update(validation_after_update=v7.score(rows), paired_gate=gate)
            if gate['paired_contact_gate_met']:
                state['status'] = 'contact_ready'
                v7.save(home/'contact_ready.json', certificate(cfg, state, report, gate))
            print('paired validation', k, 'safe index-first', gate['fixed_11']['safe_qualified_index_first'],
                  'peak isolated drive (deg)', gate['fixed_peak_drive_deg'], 'qualified seeds',
                  gate['qualifying_fixed_seeds'], 'all unsafe', gate['all_16']['unsafe'], flush=True)
        if k == cfg['iterations'] and state['status'] == 'running':
            state['status'] = 'iteration_limit'
        v7.save(folder/'result.json', result)
        v7.save(path, state)
        print('contact PPO', k, state['status'], flush=True)
        if state['status'] != 'running':
            break
    return state['status']


_panel = FunctionType(base.panel.__code__, dict(base.panel.__globals__, wave=wave, request=request))
face_campaign = _specialize(v7.campaign, [
    ('47_000_000 + (k-1)*32', "TRAIN_BASE['face15'] + (k-1)*32"),
    ('67_000_000 + (k-1)*32', 'VALID_BASE + (k-1)*32'),
], dict(TRAIN_BASE=TRAIN_BASE, VALID_BASE=VALID_BASE, request=request, wave=wave, panel=_panel, fingerprint=fingerprint))


def run_path(path):
    resolved = path.resolve()
    if path.is_symlink() or resolved.parent != points.RUNS or not re.fullmatch(r'r27_recipe[A-Za-z0-9_-]+',resolved.name):
        raise ValueError('Use a distinct experiment directory directly under runs/')
    return resolved


def long_run(args, probe, contact, face):
    # Each subprocess appears as its actual stage in CubeLab and uses the same
    # locks, accepted-wave retries and checkpoint recovery as a manual stage.
    script = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
    options = ['--probe-dir', str(probe), '--contact-dir', str(contact)]
    for key in ('updates', 'validate_every', 'lanes'):
        val = getattr(args,key)
        if val is not None:
            options.extend(['--'+key.replace('_','-'),str(val)])
    subprocess.run([*script, 'train', '--run-dir', str(contact), *options], check=True)
    status = v7.read(contact/'state.json')['status']
    if status != 'contact_ready':
        print('Contact PPO ended with', status, 'without qualification for +15 degrees', flush=True)
        return
    subprocess.run([*script, 'train15', '--run-dir', str(face), *options], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('long','check','train','train15'))
    p.add_argument('--probe-dir',type=Path,default=base.PROBE_HOME)
    p.add_argument('--contact-dir',type=Path,default=CONTACT_HOME)
    p.add_argument('--run-dir',type=Path,help='Stage output; for long, the +15 output')
    p.add_argument('--updates',type=int,help='Maximum PPO updates per stage (default 100)')
    p.add_argument('--validate-every',type=int,help='Validation interval in updates (default 5)')
    p.add_argument('--lanes',type=int,choices=(4,8,16,32))
    args=p.parse_args(argv)
    probe, contact = run_path(args.probe_dir), run_path(args.contact_dir)
    stage = 'face15' if args.mode in ('long','train15') else 'contact2'
    home = run_path(args.run_dir or (FACE_HOME if stage=='face15' else contact))
    if probe in (contact,home) or (stage=='face15' and home==contact):
        p.error('The baseline, contact and +15 outputs must be different directories')
    if args.mode == 'long':
        return long_run(args,probe,contact,home)
    common, neutral = verified_baseline(probe)
    if args.mode == 'check':
        from natural_face90_cartesian_contact_worker import reward_contract
        cfg = dict(common, schema=CONTACT, curriculum_stage='contact2', lanes=args.lanes or 16)
        validate_request(train_req(cfg, 1, None))
        print('CPU CHECK OK; source probe', neutral['candidate']['fixed_11'], 'all unsafe',
              neutral['candidate']['all_16']['unsafe'], 'neutral +2 gate',neutral['paired_contact_gate_met'],
              'contact2 PPO can start with', reward_contract('contact2')['name'], flush=True)
        return
    origin = (verified_origin(contact,common,neutral) if stage=='face15' else dict(iteration=0,checkpoint=None,sha=None))
    prior = v7.read(home/'config.json') if (home/'config.json').is_file() else {}
    updates = args.updates if args.updates is not None else prior.get('updates',100)
    cadence = args.validate_every if args.validate_every is not None else prior.get('validate_every',5)
    if not (1 <= updates <= 2000-origin['iteration'] and 1 <= cadence <= 2000):
        p.error('Invalid update count or validation interval')
    cfg = dict(common, schema=FACE if stage=='face15' else CONTACT, curriculum_stage=stage, origin=origin,
               updates=updates, iterations=origin['iteration']+updates, lanes=args.lanes or prior.get('lanes',16),
               validate_every=cadence, validation_seeds=16, target_deg=15, contact_point_offset_deg=7,
               pad_preshape_offset_deg=-6, smoke=False)
    if stage=='face15':
        cfg['contact_dir']=str(contact)
        cfg['contact_ready_sha256']=v7.digest(contact/'contact_ready.json')
    if home.exists() and (not (home/'config.json').is_file() or cfg != prior):
        raise ValueError('Output directory belongs to a different run or training configuration')
    home.mkdir(exist_ok=True)
    with v7.exclusive(home/'campaign.lock') as fd:
        if (home/'config.json').exists() and v7.read(home/'config.json') != cfg:
            raise ValueError('Another curriculum already owns this directory')
        v7.save(home/'config.json', cfg)
        if stage=='contact2':
            return contact_campaign(home,cfg,neutral,fd)
        if (home/'state.json').exists():
            verified_state(home,cfg)
        face_campaign(home,cfg,fd)


if __name__=='__main__':
    main()
