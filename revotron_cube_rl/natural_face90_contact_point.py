"""Paired, frozen R70 material-point sweep using the archived C rollout as zero."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

import natural_face90_hover_c as hover
import natural_face90_v7_long as v7
from natural_face90_contact_point_metrics import compare
from natural_face90_recipe_long import worker_software

HERE = v7.HERE
RUNS = hover.ab.RUNS
DEFAULT_SWEEP = RUNS / 'r27_recipe_contact_point_sweep70'
BASELINE = RUNS / 'r27_recipe_hover_c70'
SCHEMA = 'natural_face90_contact_point_ppo_v1'
ANGLES = {7: 'plus7', -7: 'minus7'}
SUMMARY = 'contact_point_summary.json'
FILES = tuple(dict.fromkeys((*v7.FILES, *hover.ab.FILES, *hover.FILES,
    'natural_face90_hover_ab_metrics.py', 'natural_face90_hover_d_metrics.py',
    'natural_face90_contact_probe_native.py', 'natural_face90_contact_point.py',
    'natural_face90_contact_point_train.py', 'natural_face90_contact_point_worker.py',
    'natural_face90_contact_point_metrics.py', 'run_natural_face90_contact_point.sh')))


def fingerprint():
    return {name: v7.digest(HERE / name) for name in FILES}


def validate_request(req):
    angle = req.get('contact_point_offset_deg')
    target = req.get('target_deg')
    seeds = req.get('seeds', [])
    if (req.get('recipe_schema') != SCHEMA or type(angle) is not int or angle not in ANGLES
            or type(target) is not int or target not in (15, 90)
            or len(seeds) != len(set(seeds)) or not 2 <= len(seeds) <= 256
            or any(type(s) is not int for s in seeds) or len(req['parameters']) != len(seeds)):
        raise ValueError('Invalid material-point angle, target, seeds or lane count')
    # Reuse the pinned common PPO/transfer checks, which describe the 15-degree source.
    # The versioned worker independently supplies the actual target to the physical observer.
    v7.validate_request(dict(req, recipe_schema=v7.SCHEMA, target_deg=15))
    train = req['role'] == 'train'
    low, high = (41_000_000, 51_000_000) if train else (61_000_000, 71_000_000)
    if any(not low <= seed < high for seed in seeds):
        raise ValueError('Seeds outside the assigned training/validation range')
    origin = req.get('transfer')
    paired = req.get('paired_validation') is True
    if origin and (target != 15 or origin['iteration'] != 70 or origin['sha'] != hover.ab.R70_SHA
                   or not (train or paired)):
        raise ValueError('Initial material-point transfer must use frozen R70')
    if paired and (req['seeds'] != hover.ab.SEEDS or req['iteration'] != 70 or not origin):
        raise ValueError('Frozen comparison must preserve the original 16 R70 lanes')
    if not paired and set(seeds).intersection(hover.ab.SEEDS):
        raise ValueError('Development sweep seeds cannot be reused for training or fresh validation')
    promoted = req.get('promote_from_15')
    if promoted is not None and (not train or target != 90 or origin
            or req['iteration'] != promoted['iteration'] + 1
            or (req['checkpoint'], req['checkpoint_sha']) != (promoted['checkpoint'], promoted['sha'])):
        raise ValueError('Invalid transfer of the qualified 15-degree checkpoint')


def verify_quarter_turn(trace, rows):
    """Recompute net attributed motion, spanning only consecutive isolated drive endpoints."""
    from natural_face90_hover_ab_metrics import CONTACT_IMPULSE

    totals = [0.] * len(rows)
    peaks = [0.] * len(rows)
    first = [None] * len(rows)
    previous = [None] * len(rows)
    with trace.open() as stream:
        for line in stream:
            r = json.loads(line)
            lane = r['lane']
            if not 0 <= lane < len(rows) or r['seed'] != rows[lane]['seed']:
                raise ValueError('Mismatched quarter-turn trace lane')
            angle = r['face_deg']
            if not math.isfinite(angle):
                raise ValueError('Nonfinite measured layer angle')
            isolated = (r['safe'] and r['qualified_index_frame'] and r['index_distal_impulse_Ns'] > CONTACT_IMPULSE
                        and r['other_digit_target_impulse_Ns'] <= CONTACT_IMPULSE)
            last = previous[lane]
            if last is not None:
                if r['step'] != last[0] + 1:
                    raise ValueError('Missing quarter-turn control interval')
                if first[lane] is not None:
                    delta = (angle - last[1] + 180.) % 360. - 180.
                    driven = r['drive_command'] and isolated and last[2]
                    totals[lane] += delta if driven else min(0., delta)
                    if driven:
                        from_first = (angle - first[lane] + 180.) % 360. - 180.
                        peaks[lane] = max(peaks[lane], min(totals[lane], from_first))
            if isolated and first[lane] is None:
                first[lane] = angle
            previous[lane] = r['step'], angle, isolated
    for row, peak in zip(rows, peaks, strict=True):
        if (not math.isclose(row['peak_isolated_drive_progress_deg'], peak, abs_tol=1e-7, rel_tol=0.)
                or (row['success'] and peak < 90 - 1e-7)):
            raise ValueError(f"Unverified quarter-turn attribution for seed {row['seed']}")


def verify_report(report, req):
    rows, learning = v7.previous.verify_turn_report(report, req)
    receipt = learning['contact_point']
    if receipt['offset_deg'] != req['contact_point_offset_deg']:
        raise ValueError('Material-point receipt disagrees with the requested angle')
    if not req['learn'] and (report.parent / 'checkpoint.pt').exists():
        raise ValueError('Frozen material-point wave wrote a checkpoint')
    if req['target_deg'] == 90:
        verify_quarter_turn(report.parent / 'turn_contact_evidence.jsonl', rows)
    return rows, learning


def wave(slot, req, fd):
    validate_request(req)
    if worker_software(fingerprint()) != req['software']:
        raise ValueError('Material-point sources changed after planning')
    slot.mkdir(parents=True, exist_ok=True)
    plan = slot / 'request.json'
    if plan.exists():
        if v7.read(plan) != req:
            raise ValueError(f'Wave already belongs to a different request: {slot}')
    else:
        v7.save(plan, req)
    if (slot / 'accepted.json').exists():
        return verify_report(v7.accepted_report(slot, req), req)
    attempt = slot / ('attempt_' + uuid.uuid4().hex[:12])
    with v7.available_gpu() as leases, attempt.with_suffix('.log').open('x') as log:
        if v7.digest(Path(req['checkpoint'])) != req['checkpoint_sha']:
            raise ValueError('Input checkpoint changed before rollout')
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(leases[0]), CUBE_DASH_ISAAC_FD=str(leases[1]))
        print(f"{req['role']}  target +{req['target_deg']}°  point {req['contact_point_offset_deg']:+d}°  {log.name}", flush=True)
        cmd = [sys.executable, '-B', '-u', str(HERE / 'natural_face90_contact_point_worker.py'),
               '--request', str(plan), '--out', str(attempt)]
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(fd, *leases))
        if result.returncode:
            raise v7.worker_failure(result.returncode, attempt, log.name)
    report = attempt / 'report.json'
    rows, learning = verify_report(report, req)
    v7.save(slot / 'accepted.json', dict(attempt=attempt.name, request_sha=v7.digest(plan), report_sha=v7.digest(report)))
    return rows, learning


def baseline():
    report = BASELINE / 'treatment' / 'report.json'
    request_file = report.parent / 'request.json'
    req = v7.read(request_file)
    plan = v7.read(BASELINE / 'plan.json')
    saved = v7.read(BASELINE / 'hover_bc_summary.json')
    v7.validate_request(req)
    if (plan['operator'] != hover.OPERATOR or plan['source_sha256'] != {f: v7.digest(HERE / f) for f in hover.FILES}
            or req['software'] != worker_software(v7.fingerprint())
            or req['seeds'] != hover.ab.SEEDS or req['iteration'] != 70 or not req.get('paired_validation')
            or req['checkpoint_sha'] != hover.ab.R70_SHA or v7.digest(Path(req['checkpoint'])) != hover.ab.R70_SHA
            or saved['plan'] != plan or saved['updated'] is not False or saved['independent'] is not False
            or saved['treatment_report'] != str(report) or saved['treatment_report_sha256'] != v7.digest(report)):
        raise ValueError('Frozen C baseline or R70 source changed')
    learning, counts = hover.frozen_report(report, req, plan['request_sha256'])
    hover.check_anchor(report, learning, counts, req['seeds'])
    return report, req


def sweep_plan(baseline_report, original):
    return dict(schema='r70_paired_material_point_sweep_v1', checkpoint=original['checkpoint'],
                checkpoint_sha=original['checkpoint_sha'], seeds=original['seeds'], offsets_deg=list(ANGLES),
                baseline_report=str(baseline_report), baseline_report_sha256=v7.digest(baseline_report),
                source_sha256=fingerprint())


def candidate(original, plan, angle):
    return dict(original, recipe_schema=SCHEMA, software=worker_software(plan['source_sha256']),
                contact_point_offset_deg=angle)


def check_frozen_weights(original, plan):
    """Load the real archived policy through each new worker contract on the CPU."""
    import torch
    from natural_face90_contact_point_worker import ACTION_MODE, restore_expert
    from natural_face90_recipe_guide import ArmExpert

    checkpoint = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
    parent = checkpoint['contract']
    channels = [parent['right_indices'].index(j) for j in parent['arm_indices']]
    for angle in ANGLES:
        req = candidate(original, plan, angle)
        contract = dict(parent, schema=SCHEMA, action_mode=ACTION_MODE,
                        software=req['software'], contact_point_offset_deg=angle)
        restore_expert(ArmExpert(channels), None, checkpoint, contract, req)


def summarize(home, original, plan):
    reports = {}
    for angle, name in ANGLES.items():
        req = candidate(original, plan, angle)
        report = v7.accepted_report(home / name, req)
        verify_report(report, req)
        reports[angle] = report
    result = compare(Path(plan['baseline_report']), reports[7], reports[-7], original['seeds'])
    return dict(result, plan=plan, updated=False, independent=False, reports={
        str(a): dict(path=str(p), sha256=v7.digest(p)) for a, p in reports.items()})


def training_selection(sweep_dir):
    home = Path(sweep_dir).resolve()
    if not (home / SUMMARY).is_file():
        raise ValueError(f'请先完成冻结选点 sweep；尚缺少 {home / SUMMARY}')
    report, original = baseline()
    plan = sweep_plan(report, original)
    if v7.read(home / 'plan.json') != plan:
        raise ValueError('Sweep plan, sources or R70 inputs changed')
    summary = summarize(home, original, plan)
    if v7.read(home / SUMMARY) != summary:
        raise ValueError('Saved selection disagrees with paired contact evidence')
    winner = summary['selected_offset_deg']
    if winner not in ANGLES:
        raise ValueError('两个落点均未达标；请查看 contact_point_summary.json。PPO 未启动。')
    return winner, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_SWEEP)
    parser.add_argument('--check', action='store_true', help='Verify the archived baseline and requests on CPU')
    args = parser.parse_args()
    home = args.run_dir.resolve()
    if home.parent != RUNS or args.run_dir.is_symlink():
        parser.error('Choose a dedicated directory directly under runs/')
    report, original = baseline()
    plan = sweep_plan(report, original)
    for angle in ANGLES:
        validate_request(candidate(original, plan, angle))
    if home.exists() and (not (home / 'plan.json').is_file() or v7.read(home / 'plan.json') != plan):
        parser.error('Directory belongs to a different or incomplete experiment. Choose a new --run-dir.')
    if args.check:
        check_frozen_weights(original, plan)
        print('CHECK OK — restored R70 on CPU at both offsets; verified archived C and 16 paired seeds')
        return
    home.mkdir(exist_ok=True)
    with v7.exclusive(home / 'campaign.lock') as fd:
        if (home / 'plan.json').exists():
            if v7.read(home / 'plan.json') != plan:
                raise ValueError('Sweep plan changed')
        else:
            v7.save(home / 'plan.json', plan)
        for angle, name in ANGLES.items():
            wave(home / name, candidate(original, plan, angle), fd)
        if sweep_plan(report, original) != plan:
            raise ValueError('Sweep sources or baseline changed during evaluation')
        summary = summarize(home, original, plan)
        if (home / SUMMARY).exists() and v7.read(home / SUMMARY) != summary:
            raise ValueError('Existing selection differs from measured results')
        v7.save(home / SUMMARY, summary)
        print('Selected material-point offset:', summary['selected_offset_deg'], '—', home / SUMMARY)


if __name__ == '__main__':
    main()
