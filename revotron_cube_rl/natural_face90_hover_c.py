"""Frozen R70 paired hover test: compare measured-q anchoring (C) with the archived B rollout."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

import natural_face90_hover_ab as ab
from natural_face90_hover_ab_metrics import compare

campaign = ab.campaign
BASE = ab.RUNS / 'r27_recipe_hover_ab70'
OPERATOR = 'anchor_nominal_right_arm_ik_to_measured_q_during_precontact_hover'
FILES = ('natural_face90_hover_c.py', 'natural_face90_hover_c_policy.py', 'run_natural_face90_hover_c.sh')


def frozen_report(report, req, request_sha):
    rows, learning = campaign.previous.verify_turn_report(report, req)
    record = learning['hover_ablation']
    counts = record['frames_by_lane']
    if (record['operator'] != ab.OPERATOR or len(counts) != len(rows)
            or any(type(n) is not int or n < 0 for n in counts)
            or (report.parent / 'checkpoint.pt').exists()
            or campaign.digest(report.parent / 'request.json') != request_sha):
        raise ValueError('Wrong frozen request or hover residual evidence')
    return learning, counts


def inputs():
    req, original, ab_plan = ab.baseline()
    report = BASE / 'treatment/report.json'
    summary_file = BASE / 'hover_ab_summary.json'
    saved = campaign.read(summary_file)
    if (campaign.read(BASE / 'plan.json') != ab_plan or saved['plan'] != ab_plan
            or saved['updated'] is not False or saved['independent'] is not False
            or saved['treatment_report'] != str(report)
            or saved['treatment_report_sha256'] != campaign.digest(report)):
        raise ValueError('Expected the completed R70 A/B experiment with its original B report')
    _, counts = frozen_report(report, req, ab_plan['control_request_sha256'])
    reproduced = compare(original, report, req['seeds'])
    if (counts != saved['hover_intervention_frames_by_lane']
            or counts != [p['B']['tracking']['samples'] for p in reproduced['per_seed']]
            or reproduced != saved['comparison']):
        raise ValueError('Archived hover A/B summary disagrees with its rollout evidence')
    plan = dict(schema='natural_face90_r70_hover_bc_v1', operator=OPERATOR,
                iteration=70, seeds=req['seeds'], checkpoint=req['checkpoint'], checkpoint_sha=ab.R70_SHA,
                request_sha256=ab_plan['control_request_sha256'], baseline_report=str(report),
                baseline_report_sha256=saved['treatment_report_sha256'],
                ab_summary_sha256=campaign.digest(summary_file),
                source_sha256={name: campaign.digest(ab.HERE / name) for name in FILES})
    return req, report, plan, reproduced['group']['B']


def worker(home):
    from natural_face90_v5_worker import require_locks
    require_locks()
    req, report, plan, _ = inputs()
    if campaign.read(home / 'plan.json') != plan:
        raise ValueError('Frozen B/C inputs or implementation changed before rollout')
    out = home / 'treatment'
    out.mkdir()
    shutil.copyfile(report.parent / 'request.json', out / 'request.json')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    try:
        import torch
        from natural_face90_hover_c_policy import run
        with torch.no_grad():
            run(req, out)
    except BaseException as exc:
        campaign.save(out / 'failed.json', dict(error=repr(exc)))
        raise
    finally:
        from isaac_compat import close_application
        close_application(app)


def check_anchor(report, learning, counts, seeds):
    record = learning['hover_anchor_ablation']
    anchor_counts = record['frames_by_lane']
    if (record['operator'] != OPERATOR or any(type(n) is not int for n in anchor_counts)
            or anchor_counts != counts):
        raise ValueError('C anchor must cover exactly the B residual-knockout frames')
    observed = [0] * len(seeds)
    errors = ('pre_ik_anchor_error_max_rad', 'pre_previous_reference_error_max_rad',
              'pre_previous_command_error_max_rad', 'sent_command_error_max_rad')
    with (report.parent / 'turn_contact_evidence.jsonl').open() as frames:
        for line in frames:
            frame = json.loads(line)
            anchor = frame['pre_hover_measured_anchor']
            lane = frame['lane']
            used, reference, previous_command, sent_command = (frame[key] for key in errors)
            if (not 0 <= lane < len(seeds) or frame['seed'] != seeds[lane]
                    or anchor not in (0, 1) or anchor != frame['pre_hover_residual_off']
                    or any(not math.isfinite(x) or x < 0 for x in (used, reference, previous_command, sent_command))
                    or used != (0. if anchor else reference)):
                raise ValueError(f"Invalid C anchor at lane {lane}, step {frame['step']}")
            observed[lane] += int(anchor)
    if observed != counts:
        raise ValueError('Measured C anchor counts disagree with the rollout receipt')


def bc_labels(value):
    """Name the first and second arms of the existing metrics B/C, preserving recorded outcomes."""
    if isinstance(value, dict):
        renamed = {}
        for key, item in value.items():
            name = '_'.join({'A': 'B', 'B': 'C'}.get(part, part) for part in key.split('_'))
            if key == 'status' and item == 'no_early_right_digit_anchor_in_A':
                item = 'no_early_right_digit_anchor_in_B'
            renamed[name] = bc_labels(item)
        return renamed
    if isinstance(value, list):
        return [bc_labels(v) for v in value]
    return value


def finish(home, req, baseline, plan):
    report = home / 'treatment/report.json'
    learning, counts = frozen_report(report, req, plan['request_sha256'])
    metrics = bc_labels(compare(baseline, report, req['seeds']))
    if counts != [p['C']['tracking']['samples'] for p in metrics['per_seed']]:
        raise ValueError('C residual intervention disagrees with the measured hover frames')
    check_anchor(report, learning, counts, req['seeds'])
    result = dict(plan=plan, updated=False, independent=False,
                  treatment_report=str(report), treatment_report_sha256=campaign.digest(report),
                  anchor_frames_by_lane=counts, comparison=metrics)
    dest = home / 'hover_bc_summary.json'
    if dest.exists():
        if campaign.read(dest) != result:
            raise ValueError('Existing B/C summary disagrees with the physical reports')
    else:
        with NamedTemporaryFile('w', dir=home, prefix='.summary-', delete=False) as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write('\n')
        os.replace(f.name, dest)
    print('Frozen B/C complete:', dest, flush=True)
    print(json.dumps(metrics['group'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ab.RUNS / 'r27_recipe_hover_c70')
    parser.add_argument('--check', action='store_true', help='Verify the B receipt, R70 and paired seeds on CPU')
    args = parser.parse_args()
    home = args.run_dir.resolve()
    if home.parent != ab.RUNS or home in (BASE, ab.CONTROL) or args.run_dir.is_symlink():
        parser.error('Use a separate directory directly under runs/')
    req, report, plan, baseline_group = inputs()
    if home.exists() and (not (home / 'plan.json').is_file() or campaign.read(home / 'plan.json') != plan):
        parser.error('Existing directory has a different plan. Use a new --run-dir.')
    print('Frozen R70', plan['checkpoint_sha'], 'paired lanes', req['seeds'], flush=True)
    if args.check:
        print('Verified B:', json.dumps(baseline_group, ensure_ascii=False), flush=True)
        print('CHECK OK', flush=True)
        return
    if home.exists():
        if not (home / 'treatment/report.json').is_file():
            parser.error('C 组目录中尚无完整报告。请查看 treatment.log。若进程已退出，用新 --run-dir 重试。')
    else:
        with ExitStack() as stack:
            try:
                leases = stack.enter_context(ab.gpu())
            except RuntimeError as exc:
                if isinstance(exc.__cause__, BlockingIOError) or str(exc).startswith('GPU 0 正被使用'):
                    parser.exit(2, 'GPU 或 Isaac 正被占用。请等资源释放后重试同一命令。\n')
                raise
            home.mkdir()
            campaign.save(home / 'plan.json', plan)
            log = home / 'treatment.log'
            print('Frozen C evaluation:', log, flush=True)
            env = dict(os.environ, CUBE_DASH_GPU_FD=str(leases[0]), CUBE_DASH_ISAAC_FD=str(leases[1]))
            with log.open('x') as handle:
                result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker',
                                         str(home)], stdout=handle, stderr=subprocess.STDOUT,
                                        env=env, pass_fds=tuple(leases))
            if result.returncode or not (home / 'treatment/report.json').is_file():
                raise campaign.worker_failure(result.returncode, home / 'treatment', str(log))
        if inputs()[2] != plan:
            raise ValueError('R70, B or the C implementation changed during this run')
    finish(home, req, report, plan)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
