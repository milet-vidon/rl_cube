"""Pair an archived R70 C rollout with one frozen D approach-recentering evaluation."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

import natural_face90_hover_c as c
from natural_face90_hover_d_metrics import compare, read_episodes

ab = c.ab
campaign = c.campaign
BASE = ab.RUNS / 'r27_recipe_hover_c70'
FILES = ('natural_face90_hover_d.py', 'natural_face90_hover_d_policy.py',
         'natural_face90_hover_d_metrics.py', 'run_natural_face90_hover_d.sh')
OPERATOR = 'hold_depth_for_unloaded_approach_recenter'
INNER_M = .004
OUTER_M = .007


def inputs():
    req, b_report, c_plan, _ = c.inputs()
    report = BASE / 'treatment/report.json'
    summary = BASE / 'hover_bc_summary.json'
    saved = campaign.read(summary)
    if (campaign.read(BASE / 'plan.json') != c_plan or saved['plan'] != c_plan
            or saved['updated'] is not False or saved['independent'] is not False
            or saved['treatment_report'] != str(report)
            or saved['treatment_report_sha256'] != campaign.digest(report)):
        raise ValueError('Expected the completed, frozen C rollout with its original report')

    learning, counts = c.frozen_report(report, req, c_plan['request_sha256'])
    c.check_anchor(report, learning, counts, req['seeds'])
    reproduced = c.bc_labels(c.compare(b_report, report, req['seeds']))
    if (reproduced != saved['comparison'] or counts != saved['anchor_frames_by_lane']
            or counts != [p['C']['tracking']['samples'] for p in reproduced['per_seed']]):
        raise ValueError('Archived C comparison disagrees with the physical evidence')
    plan = dict(schema='natural_face90_r70_hover_cd_v1', operator=OPERATOR,
                inner_m=INNER_M, outer_m=OUTER_M, iteration=req['iteration'],
                checkpoint=req['checkpoint'], checkpoint_sha=ab.R70_SHA, seeds=req['seeds'],
                request_sha256=c_plan['request_sha256'], baseline_report=str(report),
                baseline_report_sha256=saved['treatment_report_sha256'],
                bc_summary_sha256=campaign.digest(summary),
                source_sha256={name: campaign.digest(ab.HERE / name) for name in FILES})
    return req, report, plan, reproduced['group']['C']


def worker(home):
    from natural_face90_v5_worker import require_locks
    require_locks()
    req, baseline, plan, _ = inputs()
    if campaign.read(home / 'plan.json') != plan:
        raise ValueError('Frozen C/D inputs or implementation changed before rollout')
    out = home / 'treatment'
    out.mkdir()
    shutil.copyfile(baseline.parent / 'request.json', out / 'request.json')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    try:
        import torch
        from natural_face90_hover_d_policy import run
        with torch.no_grad():
            run(req, out)
    except BaseException as exc:
        campaign.save(out / 'failed.json', dict(error=repr(exc)))
        raise
    finally:
        from isaac_compat import close_application
        close_application(app)


def finish(home, req, baseline, plan):
    report = home / 'treatment/report.json'
    learning, anchors = c.frozen_report(report, req, plan['request_sha256'])
    c.check_anchor(report, learning, anchors, req['seeds'])
    comparison = compare(baseline, report, req['seeds'])
    receipt = learning['approach_recenter_ablation']
    holds = [p['D']['hold_frames'] for p in comparison['per_seed']]
    if (receipt['operator'] != OPERATOR or receipt['inner_m'] != INNER_M
            or receipt['outer_m'] != OUTER_M or receipt['frames_by_lane'] != holds
            or any(type(n) is not int for n in receipt['frames_by_lane'])
            or anchors != [p['D']['anchor_frames'] for p in comparison['per_seed']]):
        raise ValueError('D approach-recentering receipt disagrees with the control steps')
    result = dict(plan=plan, updated=False, independent=False,
                  treatment_report=str(report), treatment_report_sha256=campaign.digest(report),
                  anchor_frames_by_lane=anchors, recenter_frames_by_lane=holds, comparison=comparison)
    dest = home / 'hover_cd_summary.json'
    if not dest.exists():
        # A second reader can finish the same report concurrently. Publish only complete bytes.
        with NamedTemporaryFile('w', dir=home, prefix='.summary-', delete=False) as f:
            temp = Path(f.name)
            try:
                json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
                f.write('\n')
                f.flush()
                try:
                    os.link(temp, dest)
                except FileExistsError:
                    pass
            finally:
                temp.unlink(missing_ok=True)
    if campaign.read(dest) != result:
        raise ValueError('Existing C/D summary disagrees with the verified reports')
    print('Frozen C/D complete:', dest, flush=True)
    print(json.dumps(comparison['group'], ensure_ascii=False), flush=True)


def launch(home, plan):
    # Obtain both existing project leases before allocating the new run directory.
    with ExitStack() as stack:
        try:
            leases = stack.enter_context(ab.gpu())
        except RuntimeError as exc:
            if isinstance(exc.__cause__, BlockingIOError) or str(exc).startswith('GPU 0 正被使用'):
                raise SystemExit('GPU 或 Isaac 正被占用。请等资源释放后重试同一命令。') from None
            raise
        home.mkdir()
        campaign.save(home / 'plan.json', plan)
        log = home / 'treatment.log'
        print('Frozen D evaluation:', log, flush=True)
        env = dict(os.environ, CUBE_DASH_GPU_FD=str(leases[0]), CUBE_DASH_ISAAC_FD=str(leases[1]))
        with log.open('x') as handle:
            result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(home)],
                                    stdout=handle, stderr=subprocess.STDOUT, env=env, pass_fds=tuple(leases))
        if result.returncode or not (home / 'treatment/report.json').is_file():
            raise campaign.worker_failure(result.returncode, home / 'treatment', str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ab.RUNS / 'r27_recipe_hover_d70')
    parser.add_argument('--check', action='store_true', help='Verify the C receipt, R70 and original seeds on CPU')
    args = parser.parse_args()
    home = args.run_dir.resolve()
    if home.parent != ab.RUNS or home in (BASE, c.BASE, ab.CONTROL) or args.run_dir.is_symlink():
        parser.error('Use a dedicated directory directly under runs/')
    req, baseline, plan, baseline_group = inputs()
    if home.exists() and (not (home / 'plan.json').is_file() or campaign.read(home / 'plan.json') != plan):
        parser.error('Existing directory belongs to a different experiment. Use a new --run-dir.')
    print('Frozen R70', plan['checkpoint_sha'], 'paired lanes', req['seeds'], flush=True)
    if args.check:
        # Exercise the new C/D phase attribution on the available real C trace on CPU.
        rows = read_episodes(baseline, req['seeds'], treatment=False)
        if (sum(r['anchor_frames'] for r in rows) != baseline_group['hover_frames']
                or sum(r['unsafe'] for r in rows) != baseline_group['unsafe']):
            raise ValueError('C readback disagrees with the archived hover or safety counts')
        print('Verified C:', json.dumps(baseline_group, ensure_ascii=False), flush=True)
        print('CHECK OK', flush=True)
        return
    if home.exists():
        if not (home / 'treatment/report.json').is_file():
            parser.error('D 组目录中尚无完整报告。请查看 treatment.log。若进程已退出，用新 --run-dir 重试。')
    else:
        launch(home, plan)
        if inputs()[2] != plan:
            raise ValueError('R70, C or the D implementation changed during this run')
    finish(home, req, baseline, plan)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
