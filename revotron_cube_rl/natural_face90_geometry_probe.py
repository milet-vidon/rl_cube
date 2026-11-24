"""Replay a completed paired validation and record native right-index contact geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import FunctionType

import natural_face90_v7_long as campaign
from natural_face90_v5_long import gpu


HERE = Path(__file__).resolve().parent
PROBE_FILES = ('natural_face90_geometry_probe.py', 'natural_face90_contact_probe_native.py',
               'run_natural_face90_geometry_probe.sh')
EVIDENCE = 'turn_contact_evidence.jsonl'
CONTACTS = 'native_contact_geometry.jsonl'


def hashes():
    return {name: campaign.digest(HERE / name) for name in PROBE_FILES}


def source_wave(home):
    cfg = campaign.read(home / 'config.json')
    rec = campaign.read(home / 'frozen_validation.json')
    if (cfg['schema'] != campaign.SCHEMA or cfg['mode'] != 'compare'
            or rec['mode'] != 'compare' or rec['updated'] is not False
            or rec['independent'] is not False or cfg['source_sha256'] != campaign.fingerprint()):
        raise ValueError('Expected completed paired validation with unchanged v4 sources')
    slot = home / f"iteration_{rec['iteration']:04d}" / 'validate' / 'wave_0000'
    req = campaign.read(slot / 'request.json')
    campaign.validate_request(req)
    if (req['learn'] is not False or req['role'] != 'validate' or req.get('paired_validation') is not True
            or cfg['frozen_seed_batches'] != [req['seeds']] or req['seeds'] != rec['seeds']
            or len(set(req['seeds'])) != len(req['seeds']) or req['transfer'] != cfg['origin']
            or req['iteration'] != rec['iteration'] or req['checkpoint'] != rec['checkpoint']
            or req['checkpoint_sha'] != rec['checkpoint_sha']
            or campaign.digest(Path(req['checkpoint'])) != req['checkpoint_sha']
            or req['software'] != campaign.worker_software(cfg['source_sha256'])):
        raise ValueError('Mismatched frozen checkpoint, seed order, batch layout or request')
    report = campaign.accepted_report(slot, req)
    rows, _ = campaign.previous.verify_turn_report(report, req)
    if rec['result'] != campaign.score(rows):
        raise ValueError('Accepted rows do not reproduce the frozen validation summary')
    return req, report


def traces(path, seeds):
    digests = {seed: hashlib.sha256() for seed in seeds}
    first = {}
    with path.open('rb') as f:
        for line in f:
            row = json.loads(line)
            seed = row['seed']
            if seed not in digests:
                continue
            digests[seed].update(line)
            if row['pre_index_contact'] and seed not in first:
                first[seed] = row
    return {seed: h.hexdigest() for seed, h in digests.items()}, first


def pick_seeds(req, report, selected):
    if selected is None:
        rows = campaign.read(report)['rows']
        hits = [r['seed'] for r in rows if r['qualified_index_frames'] > 0]
        _, first = traces(report.parent / EVIDENCE, req['seeds'])
        if not hits or not (others := [s for s in first if s != hits[0]]):
            raise ValueError('Select two seeds explicitly with --seeds')
        selected = [hits[0], max(others, key=lambda s: first[s]['pre_lateral_error_m'])]
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(req['seeds']):
        raise ValueError('Select distinct seeds present in the accepted validation')
    return list(selected)


def worker(home):
    """Separate process because closing the Isaac application also exits Python."""
    from natural_face90_v5_worker import require_locks
    require_locks()
    plan = campaign.read(home / 'geometry_plan.json')
    if plan['probe_source_sha256'] != hashes():
        raise ValueError('Probe sources changed')
    req, report = source_wave(Path(plan['source']))
    if campaign.digest(report) != plan['accepted_report_sha256'] or req['seeds'] != plan['lane_seeds']:
        raise ValueError('Source changed after the probe was planned')
    selected = pick_seeds(req, report, plan['selected_seeds'])
    out = home / 'replay'
    out.mkdir()
    shutil.copyfile(report.parent / 'request.json', out / 'request.json')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    try:
        import torch
        import natural_face90_v7_worker as original
        from natural_face90_contact_probe_native import install_probe

        class RecordingPhase(original.PostPhase):
            def __init__(self, e, request, folder):
                super().__init__(e, request, folder)
                self.probe = install_probe(self, selected, folder)

            def remember(self, step, q, dq, command, core, prefix, post):
                super().remember(step, q, dq, command, core, prefix, post)
                self.probe.begin_step(step, post & self.learning)

            def finish_learning(self):
                result = super().finish_learning()
                result['geometry_probe'] = self.probe.close()
                return result

        run = FunctionType(original.run.__code__, dict(original.run.__globals__, PostPhase=RecordingPhase))
        with torch.no_grad():
            run(req, out)
    except BaseException as exc:
        campaign.save(out / 'failed.json', dict(error=repr(exc)))
        raise
    finally:
        from isaac_compat import close_application
        close_application(app)


def first_contact_points(path, seeds, earliest_frame):
    first = {}
    with path.open() as f:
        for line in f:
            point = json.loads(line)
            seed = point['seed']
            if seed not in seeds:
                raise ValueError('Unexpected geometry seed')
            if point['step'] != earliest_frame.get(seed):
                continue
            # Contacts on a control frame can come from any of its eight physical substeps.
            # Select the maximum-impulse point in the earliest measured substep of that frame.
            previous = first.get(seed)
            if (previous is None or (point['substep'], -point['impulse_Ns'])
                    < (previous['substep'], -previous['impulse_Ns'])):
                first[seed] = point
    return first


def finish(home, req, source_report, plan):
    out = home / 'replay'
    rows, learning = campaign.previous.verify_turn_report(out / 'report.json', req)
    geometry = learning['geometry_probe']
    if (geometry['sha256'] != campaign.digest(out / CONTACTS)
            or list(geometry['selected_seeds']) != plan['selected_seeds']
            or (out / 'checkpoint.pt').exists() or hashes() != plan['probe_source_sha256']):
        raise ValueError('Replay geometry receipt or frozen execution could not be verified')
    seeds = plan['selected_seeds']
    old_sha, old_first = traces(source_report.parent / EVIDENCE, seeds)
    new_sha, new_first = traces(out / EVIDENCE, seeds)
    # The source logs the impulse first as a post-step readback on frame k,
    # then as pre_index_contact on k+1 if that lane is still active. Use the
    # actual post-step signal here, including contacts on terminal frames.
    first_post = {}
    from contact_hold90_cuda_domain import CONTACT_IMPULSE
    with (out / EVIDENCE).open() as f:
        for line in f:
            row = json.loads(line)
            if row['seed'] in seeds and row['index_distal_impulse_Ns'] > CONTACT_IMPULSE:
                first_post.setdefault(row['seed'], row['step'])
    points = first_contact_points(out / CONTACTS, seeds, first_post)
    selected_results = []
    for seed in seeds:
        if seed in first_post and seed not in points:
            raise ValueError(f'Missing native points on first measured contact frame for seed {seed}')
        row = next(r for r in rows if r['seed'] == seed)
        selected_results.append(dict(seed=seed, lane=row['lane'], end_reason=row['end_reason'],
            source_trace_sha256=old_sha[seed], replay_trace_sha256=new_sha[seed],
            exact_controller_trace=old_sha[seed] == new_sha[seed],
            source_first_pre_index_contact_step=old_first.get(seed, {}).get('step'),
            replay_first_pre_index_contact_step=new_first.get(seed, {}).get('step'),
            first_native_contact=points.get(seed)))
    record = dict(selected=selected_results, geometry=geometry, updated=False, independent=False,
                  checkpoint_sha=req['checkpoint_sha'], replay_report_sha256=campaign.digest(out/'report.json'),
                  accepted_report_sha256=plan['accepted_report_sha256'], replay_score=campaign.score(rows))
    campaign.save(home / 'geometry_summary.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compare-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True, help='New directory under runs/')
    p.add_argument('--seeds', type=int, nargs='+',
                   help='Seeds to observe; default: a qualified episode and the largest first-touch lateral error')
    p.add_argument('--check', action='store_true', help='Verify inputs and print selection on CPU; creates no output')
    args = p.parse_args()
    source, home = args.compare_run.resolve(), args.run_dir.resolve()
    if home == source or home.parent != campaign.ROOT / 'runs' or home.exists():
        p.error('Use a new run directory directly under runs/')
    req, report = source_wave(source)
    selected = pick_seeds(req, report, args.seeds)
    plan = dict(schema='right_index_native_geometry_v1', source=str(source),
                iteration=req['iteration'], checkpoint_sha=req['checkpoint_sha'],
                probe_source_sha256=hashes(), accepted_report_sha256=campaign.digest(report),
                selected_seeds=selected, selected_lanes=[req['seeds'].index(s) for s in selected],
                lane_seeds=req['seeds'])
    print('Selected seeds', selected, 'lanes', plan['selected_lanes'],
          'original batch size', len(req['seeds']), 'checkpoint', req['checkpoint_sha'], flush=True)
    if args.check:
        print('CHECK OK', flush=True)
        return
    # The project-wide leases cover child startup, the entire replay, and child exit.
    with gpu() as leases:
        home.mkdir()
        campaign.save(home / 'geometry_plan.json', plan)
        log = home / 'replay.log'
        print('Frozen contact geometry replay; log', log, flush=True)
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(leases[0]), CUBE_DASH_ISAAC_FD=str(leases[1]))
        with log.open('x') as fp:
            result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                     '--worker', str(home)], env=env, pass_fds=tuple(leases),
                                    stdout=fp, stderr=subprocess.STDOUT)
        if result.returncode:
            raise campaign.worker_failure(result.returncode, home / 'replay', str(log))
    finish(home, req, report, plan)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
