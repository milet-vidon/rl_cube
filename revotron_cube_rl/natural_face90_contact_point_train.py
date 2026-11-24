"""Train at the selected R70 material point, then promote a qualified +15 policy to +90."""
import argparse
import math
from pathlib import Path
import re

import natural_face90_contact_point as points
import natural_face90_v7_long as v7
from contact_hold90_cuda_domain import _specialize


DEFAULT_HOME = v7.ROOT / 'runs/r27_recipe_contact_point_ppo70'
DEFAULT_QUARTER_TURN = v7.ROOT / 'runs/r27_recipe_contact_point_ppo90'
TRAIN_SEED_BASE = {15: 48_000_000, 90: 49_000_000}
VALIDATION_SEED_BASE = {15: 67_500_000, 90: 68_500_000}


def request(cfg, role, seeds, iteration, checkpoint):
    req = v7.request(cfg, role, seeds, iteration, checkpoint)
    req['contact_point_offset_deg'] = cfg['contact_point_offset_deg']
    req['target_deg'] = cfg['target_deg']
    if req['target_deg'] == 90 and 'transfer' in req:
        req['promote_from_15'] = req.pop('transfer')
    return req


def panel(slot, cfg, iteration, checkpoint, batches, fd):
    rows = []
    for seeds in batches:
        req = request(cfg, 'validate', seeds, iteration, checkpoint)
        group, _ = points.wave(slot / f'wave_{len(rows):04d}', req, fd)
        rows.extend(group)
    return score(rows)


def score(rows):
    result = v7.score(rows)
    target = rows[0]['target_deg']
    if target not in TRAIN_SEED_BASE or any(r['target_deg'] != target for r in rows):
        raise ValueError('Expected one consistent target for scoring')
    result['qualification_successes'] = result['index_driven_successes']
    if target == 90:
        # Require the entire quarter turn to be attributable to isolated index drive.
        result['qualification_successes'] = sum(
            r['success'] and not r['unsafe'] and r['peak_isolated_drive_progress_deg'] >= 90 - 1e-7
            for r in rows)
    return result


# Continue the existing campaign's optimizer/update, commit, held-out allocation
# and qualification protocol. Each stage has its own development seed blocks.
campaign = _specialize(v7.campaign, [
    ('47_000_000 + (k-1)*32', 'TRAIN_SEED_BASE[cfg["target_deg"]] + (k-1)*32'),
    ('67_000_000 + (k-1)*32', 'VALIDATION_SEED_BASE[cfg["target_deg"]] + (k-1)*32'),
    ("valid['index_driven_successes'] >=", "valid['qualification_successes'] >="),
    ("test['index_driven_successes'] >=", "test['qualification_successes'] >="),
    ("'qualified_15deg_state_specialist'", "f'qualified_{cfg[\"target_deg\"]}deg_state_specialist'"),
    ('vision=False, simulation_only=True', "target_deg=cfg['target_deg'], contact_point_offset_deg=cfg['contact_point_offset_deg'], vision=False, simulation_only=True"),
], dict(TRAIN_SEED_BASE=TRAIN_SEED_BASE, VALIDATION_SEED_BASE=VALIDATION_SEED_BASE,
        wave=points.wave, request=request, panel=panel, score=score, fingerprint=points.fingerprint))


def check_state(home, cfg):
    """Check the committed resume point, including a partially finished first iteration."""
    state_file = home / 'state.json'
    if not state_file.exists():
        return
    state = v7.read(state_file)
    k = state['iteration']
    if type(k) is not int or not cfg['origin']['iteration'] <= k <= cfg['iterations']:
        raise ValueError('Invalid committed training iteration')
    if k == cfg['origin']['iteration']:
        checkpoint, sha = cfg['origin']['checkpoint'], cfg['origin']['sha']
    else:
        slot = home / f'iteration_{k:04d}' / 'train'
        req = v7.read(slot / 'request.json')
        points.validate_request(req)
        if (req['role'] != 'train' or req['iteration'] != k or req['target_deg'] != cfg['target_deg'] or
                req['contact_point_offset_deg'] != cfg['contact_point_offset_deg'] or
                req['software'] != v7.worker_software(cfg['source_sha256'])):
            raise ValueError('Saved training request disagrees with the selected point or software')
        _, learning = points.verify_report(v7.accepted_report(slot, req), req)
        if learning != v7.read(slot.parent / 'result.json')['learning']:
            raise ValueError('Saved iteration disagrees with the accepted training wave')
        checkpoint, sha = learning['output_checkpoint'], learning['output_checkpoint_sha']
    if (state['checkpoint'], state['checkpoint_sha']) != (checkpoint, sha) or v7.digest(Path(checkpoint)) != sha:
        raise ValueError('Committed training checkpoint changed')


def accepted_panel(slot, cfg, iteration, checkpoint, seeds):
    """Replay the verifier for every previously accepted wave, without starting workers."""
    rows = []
    for batch in v7.batches(seeds, cfg['lanes']):
        wave_slot = slot / f'wave_{len(rows):04d}'
        req = request(cfg, 'validate', batch, iteration, checkpoint)
        points.validate_request(req)
        group, _ = points.verify_report(v7.accepted_report(wave_slot, req), req)
        rows.extend(group)
    return score(rows)


def qualified_origin(home, sweep, angle, source):
    if home.parent != v7.ROOT / 'runs' or home == sweep:
        raise ValueError('Choose the completed +15 training run under runs/')
    cfg = v7.read(home / 'config.json')
    if (cfg.get('schema') != points.SCHEMA or cfg.get('target_deg') != 15 or
            cfg['contact_point_offset_deg'] != angle or cfg['origin'] != source['transfer'] or
            cfg['sweep_dir'] != str(sweep) or cfg['source_sha256'] != points.fingerprint() or
            cfg['sweep_summary_sha256'] != v7.digest(sweep / 'contact_point_summary.json')):
        raise ValueError('Source +15 training must use the same verified sweep, point, software and R70')
    state = v7.read(home / 'state.json')
    if state.get('status') != 'qualified_15deg_state_specialist':
        raise ValueError('The +15 training run has not qualified on its independent panel')
    check_state(home, cfg)
    k, ck, sha = state['iteration'], state['checkpoint'], state['checkpoint_sha']
    folder = home / f'iteration_{k:04d}'
    start = VALIDATION_SEED_BASE[15] + (k - 1) * 32
    valid = accepted_panel(folder / 'validate', cfg, k, ck, list(range(start, start + cfg['validation_seeds'])))
    result = v7.read(folder / 'result.json')
    held = v7.read(home / 'final_test.json')
    registry = v7.read(v7.ROOT / 'runs/natural_face90_test_registry/allocations.json')
    first = registry[str(home)]
    seeds = list(range(first, first + 32))
    if not 69_000_000 <= first <= 71_000_000 - 32 or seeds != held['seeds']:
        raise ValueError('Independent +15 qualification seeds disagree with the project registry')
    test = accepted_panel(folder / 'independent_test', cfg, k, ck, seeds)
    policy = v7.read(home / 'qualified_policy.json')
    if (valid != result['validation_after_update'] or valid['unsafe'] or
            valid['qualification_successes'] < math.ceil(.875 * cfg['validation_seeds']) or
            test != result['independent_test'] or test != held['results'] or test != policy['results'] or
            test['unsafe'] or test['qualification_successes'] < 28 or
            (held['checkpoint'], held['sha']) != (ck, sha) or
            (policy['checkpoint'], policy['sha']) != (ck, sha) or
            policy['contact_point_offset_deg'] != angle or policy['target_deg'] != 15 or
            policy['source_sha256'] != cfg['source_sha256']):
        raise ValueError('Qualified +15 policy disagrees with its validation, test or checkpoint evidence')
    return dict(iteration=k, checkpoint=ck, sha=sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir', type=Path, help='Completed frozen paired contact-point experiment')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--from-qualified15', type=Path, help='Promote this verified +15 training run to +90')
    parser.add_argument('--iterations', type=int, help='Last iteration number, inclusive (default: source iteration + 30)')
    parser.add_argument('--lanes', type=int, choices=(2, 4, 8, 16, 32))
    parser.add_argument('--validation-seeds', type=int, choices=(8, 16, 32))
    parser.add_argument('--validate-every', type=int, help='Validation interval, plus first and last update')
    parser.add_argument('--rng', type=int, help='PPO sampling seed; defaults to the R70 source setting')
    parser.add_argument('--check', action='store_true', help='Verify selection, configuration and inputs on CPU only')
    parser.add_argument('--resume', action='store_true', help='Continue the identical configured experiment')
    args = parser.parse_args(argv)
    destination = args.run_dir or (DEFAULT_QUARTER_TURN if args.from_qualified15 else DEFAULT_HOME)
    home = destination.resolve()
    if (destination.is_symlink() or home.parent != v7.ROOT / 'runs'
            or not re.fullmatch(r'r27_recipe[A-Za-z0-9_-]+', home.name)):
        parser.error('Use a dedicated runs/r27_recipe... training directory')
    if args.resume and not (home / 'config.json').is_file():
        parser.error('No configured run to resume')
    if home.exists() and not args.resume:
        parser.error('Training directory already exists; use --resume or choose a new directory')
    prior = v7.read(home / 'config.json') if args.resume else {}
    if prior and prior.get('schema') != points.SCHEMA:
        parser.error('The destination belongs to a different training campaign')
    promotion = prior.get('from_qualified15') if args.from_qualified15 is None else str(args.from_qualified15.resolve())
    source_dir = Path(promotion) if promotion else None
    if source_dir == home:
        parser.error('Use a new +90 training directory separate from the +15 source')
    default_sweep = points.DEFAULT_SWEEP
    if source_dir:
        default_sweep = v7.read(source_dir / 'config.json')['sweep_dir']
    sweep = (args.sweep_dir or Path(prior.get('sweep_dir', default_sweep))).resolve()
    if home == sweep:
        parser.error('Choose different experiment directories for screening and training')
    try:
        angle, source = points.training_selection(sweep)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    origin = source['transfer']
    if (angle not in (-7, 7) or origin['iteration'] != 70 or
            (source['checkpoint'], source['checkpoint_sha']) != (origin['checkpoint'], origin['sha'])):
        parser.error('Training requires a selected material point and the R70 source checkpoint')
    if source_dir:
        origin = qualified_origin(source_dir, sweep, angle, source)
    target = 90 if source_dir else 15
    defaults = dict(iterations=origin['iteration'] + 30, validate_every=5, lanes=16,
                    validation_seeds=16, rng=source['policy_seed'])
    cfg = {name: prior.get(name, default) if getattr(args, name) is None else getattr(args, name)
           for name, default in defaults.items()}
    max_iteration = 4000 if target == 90 else 2000
    if not origin['iteration'] < cfg['iterations'] <= max_iteration or not 1 <= cfg['validate_every'] <= cfg['iterations']:
        parser.error(f'Use a final iteration between {origin["iteration"] + 1} and {max_iteration} and a positive validation interval')
    if not 0 <= cfg['rng'] < 2**31:
        parser.error('PPO seed must be a nonnegative 31-bit integer')
    cfg.update(schema=points.SCHEMA, parent_sha=v7.PARENT, origin=origin, target_deg=target,
               sweep_dir=str(sweep), sweep_summary_sha256=v7.digest(sweep / 'contact_point_summary.json'),
               from_qualified15=promotion, contact_point_offset_deg=angle, source_sha256=points.fingerprint())
    from natural_face90_recipe_contacts import check_capacity_source
    from natural_face90_recipe_frames import check_source_scene
    from natural_face90_contact_point_worker import inputs

    # Exercise the actual first training request and source-scene checks without
    # constructing an Isaac application or allocating a training/holdout directory.
    begin = TRAIN_SEED_BASE[target] + origin['iteration'] * 32
    first = request(cfg, 'train', list(range(begin, begin + cfg['lanes'])), origin['iteration'] + 1, origin['checkpoint'])
    points.validate_request(first)
    scene, *_ = inputs(first)
    check_source_scene(scene)
    cfg['raw_contact_capacity'] = check_capacity_source(cfg['lanes'])
    if args.resume and cfg != prior:
        parser.error('Resume requires the same screen, point, software, sampling settings and checkpoint')
    if args.check:
        check_state(home, cfg)
        print(f'CHECK OK: selected {angle:+d} degrees; training iterations {first["iteration"]}..{cfg["iterations"]}, target +{target} degrees')
        return
    if not args.resume:
        home.mkdir()
    with v7.exclusive(home / 'campaign.lock') as fd:
        if args.resume:
            check_state(home, cfg)
        else:
            v7.save(home / 'config.json', cfg)
        campaign(home, cfg, fd)


if __name__ == '__main__':
    main()
