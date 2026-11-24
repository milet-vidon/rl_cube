"""One frozen R70 evaluation with the right-arm PPO residual disabled in hover."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

from natural_face90_geometry_probe import source_wave
import natural_face90_v7_long as campaign
from natural_face90_v5_long import gpu


HERE = Path(__file__).resolve().parent
RUNS = campaign.ROOT / 'runs'
CONTROL = RUNS / 'r27_recipe_center_v4_compare70'
R70_SHA = '0d59d3da0c4eb9454581a9d2fdb2bdefa73ae66efcf87aaf06aaf10600efcb5b'
SEEDS = list(range(66002208, 66002224))
FILES = ('natural_face90_hover_ab.py', 'natural_face90_hover_ab_policy.py',
         'natural_face90_hover_ab_metrics.py', 'run_natural_face90_hover_ab.sh',
         'natural_face90_geometry_probe.py')
OPERATOR = 'zero_right_arm_ppo_residual_during_precontact_hover'


def baseline():
    req, report = source_wave(CONTROL)
    if (req['iteration'] != 70 or req['checkpoint_sha'] != R70_SHA
            or req['seeds'] != SEEDS or req['role'] != 'validate' or req['learn'] is not False):
        raise ValueError('This experiment requires the frozen R70 checkpoint and original 16 validation lanes')
    plan = dict(schema='natural_face90_r70_hover_ab_v1', operator=OPERATOR,
                iteration=70, seeds=SEEDS, checkpoint=req['checkpoint'], checkpoint_sha=R70_SHA,
                control_report=str(report), control_report_sha256=campaign.digest(report),
                control_request_sha256=campaign.digest(report.parent/'request.json'),
                ab_source_sha256={name:campaign.digest(HERE/name) for name in FILES})
    return req, report, plan


def worker(home):
    from natural_face90_v5_worker import require_locks
    require_locks()
    req, report, plan = baseline()
    if campaign.read(home/'plan.json') != plan:
        raise ValueError('Inputs or implementation changed after the A/B run was planned')
    out = home/'treatment'
    out.mkdir()
    shutil.copyfile(report.parent/'request.json', out/'request.json')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    try:
        import torch
        from natural_face90_hover_ab_policy import run
        with torch.no_grad():
            run(req, out)
    except BaseException as exc:
        campaign.save(out/'failed.json', dict(error=repr(exc)))
        raise
    finally:
        from isaac_compat import close_application
        close_application(app)


def finish(home, req, control, plan):
    from natural_face90_hover_ab_metrics import compare
    treatment = home/'treatment/report.json'
    rows, learning = campaign.previous.verify_turn_report(treatment, req)
    ablation = learning['hover_ablation']
    counts = ablation['frames_by_lane']
    if (ablation['operator'] != OPERATOR or len(counts) != len(rows)
            or any(type(n) is not int or n < 0 for n in counts)
            or (treatment.parent/'checkpoint.pt').exists()
            or campaign.digest(Path(req['checkpoint'])) != R70_SHA
            or campaign.digest(treatment.parent/'request.json') != plan['control_request_sha256']):
        raise ValueError('Frozen intervention evidence does not match the plan')
    metrics = compare(control, treatment, req['seeds'])
    if counts != [pair['B']['tracking']['samples'] for pair in metrics['per_seed']]:
        raise ValueError('Per-lane hover counters disagree with the recorded control steps')
    comparison = dict(plan=plan, updated=False, independent=False,
                      treatment_report=str(treatment), treatment_report_sha256=campaign.digest(treatment),
                      hover_intervention_frames_by_lane=counts, comparison=metrics)
    dest = home/'hover_ab_summary.json'
    if dest.exists():
        if campaign.read(dest) != comparison:
            raise ValueError('Existing A/B summary disagrees with verified reports')
    else:
        # Readers may independently finish the same completed physical rollout.
        with NamedTemporaryFile('w', dir=home, prefix='.summary-', delete=False) as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(f.name,dest)
    print('Frozen A/B complete:', dest, flush=True)
    print(json.dumps(metrics['group'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUNS/'r27_recipe_hover_ab70')
    parser.add_argument('--check', action='store_true', help='Check paired inputs and saved control evidence on CPU')
    args = parser.parse_args()
    home = args.run_dir.resolve()
    if home.parent != RUNS or home == CONTROL or args.run_dir.is_symlink():
        parser.error('Use a dedicated directory immediately under runs/')
    req, control, plan = baseline()
    if home.exists() and (not (home/'plan.json').is_file() or campaign.read(home/'plan.json') != plan):
        parser.error('Existing directory belongs to a different experiment or software version; use a new --run-dir')
    print('Frozen R70', R70_SHA, 'paired lanes', req['seeds'], flush=True)
    if args.check:
        from natural_face90_hover_ab_metrics import summarize
        group = summarize(control, req['seeds'])['group']
        print('Control:', json.dumps(group, ensure_ascii=False))
        print('CHECK OK', flush=True)
        return
    if home.exists():
        if not (home/'treatment/report.json').is_file():
            parser.error('B 组已有运行目录但尚无完整报告。请查看 treatment.log；若上次已中断，用新的 --run-dir 从同一检查点重跑。')
    else:
        with ExitStack() as stack:
            try:
                leases = stack.enter_context(gpu())
            except RuntimeError as exc:
                if isinstance(exc.__cause__, BlockingIOError) or str(exc).startswith('GPU 0 正被使用'):
                    parser.exit(2, 'GPU 或 Isaac 正被其他任务占用（暂停的任务也可能持锁）。请等资源释放后重试同一命令。\n')
                raise
            home.mkdir()
            campaign.save(home/'plan.json', plan)
            log = home/'treatment.log'
            print('Frozen B evaluation:', log, flush=True)
            env = dict(os.environ, CUBE_DASH_GPU_FD=str(leases[0]), CUBE_DASH_ISAAC_FD=str(leases[1]))
            with log.open('x') as handle:
                result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(home)],
                                        stdout=handle, stderr=subprocess.STDOUT, env=env, pass_fds=tuple(leases))
            if result.returncode or not (home/'treatment/report.json').is_file():
                raise campaign.worker_failure(result.returncode, home/'treatment', str(log))
        # A long GPU wave can outlive a local edit or an evidence replacement.
        if baseline() != (req, control, plan):
            raise ValueError('A/B inputs or sources changed during evaluation')
    finish(home, req, control, plan)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
