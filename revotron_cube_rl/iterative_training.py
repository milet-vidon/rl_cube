"""Bounded, serial PPO training and checkpoint selection; importing does not start Isaac.

Example (run with the existing Isaac Python):
  python iterative_training.py --checkpoint runs/visual_warmstart/checkpoint.pt \
      --name visual_iterations --cycles 3 --iterations 50 --eval-steps 1800

Every child uses train.py and writes its complete log and artifacts under runs/NAME.
The input checkpoint is evaluated first. Each candidate starts from the current
best checkpoint. A fixed validation seed is used for model selection, so its score
is NOT an unbiased final test of generalization. A separate final test is needed.

Create runs/NAME/STOP to finish the current child process and stop before starting
another. --max-minutes is a soft budget checked between cycles; a started cycle
finishes training and evaluation. No background worker or remote service is started
by importing this module. The caller decides whether to launch this script hidden.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import tempfile


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize_evaluation(directory: Path) -> dict:
    """Only completed child results qualify for checkpoint selection."""
    result = read_json(directory / 'result.json')
    if result.get('status') != 'completed' or result.get('mode') != 'evaluate':
        raise ValueError(f'Evaluation did not complete: {directory}')
    if not result.get('real_rgb') or result.get('action_dim') != 56:
        raise ValueError('Evaluation must use real chest RGB and all 56 actions')
    episodes = int(result['episodes'])
    successes = int(result['successes'])
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError(f'No valid completed evaluation episodes: {directory}')
    trajectory = read_json(directory / 'trajectory.json')
    if not trajectory or len(trajectory) != int(result['steps_total']):
        raise ValueError(f'Missing/incomplete evaluation trajectory: {directory}')
    heights = [float(row['height_m']) for row in trajectory]
    holds = [int(row['hold_steps']) for row in trajectory]
    if not all(math.isfinite(height) for height in heights):
        raise ValueError('Non-finite evaluation height')
    bilateral_heights = [float(row['height_m']) for row in trajectory
                         if row['bilateral_contact'] and not row['drop'] and float(row['table_impulse']) < 1e-5]
    summary = {
        'completed_episodes': episodes, 'successes': successes,
        'success_rate': successes / episodes,
        'max_hold_steps': max(holds), 'max_height_m': max(heights),
        'max_bilateral_height_m': max(bilateral_heights, default=0.),
        'steps_total': int(result['steps_total']), 'elapsed_s': float(result['elapsed_s']),
        'seed': int(result['seed']), 'config_sha256': result['config_sha256'],
        'score_basis': 'observed_success_rate' if successes else 'proxy_only_no_success',
        'task_success_claim': False,
        'limitation': 'Fixed seed is reused for model selection; final independent testing is still required.',
    }
    return summary


def selection_score(summary: dict) -> tuple:
    height = summary['max_bilateral_height_m'] if summary['successes'] == 0 else summary['max_height_m']
    return (summary['success_rate'], summary['max_hold_steps'], height)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True, help='A new directory name under this project\'s runs/')
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--rollout-steps', type=int, default=64)
    parser.add_argument('--eval-steps', type=int, default=1800)
    parser.add_argument('--train-seed', type=int, default=7)
    parser.add_argument('--eval-seed', type=int, default=10001)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--noise-std', type=float, default=.03)
    parser.add_argument('--max-minutes', type=float)
    args = parser.parse_args(argv)
    if min(args.cycles, args.iterations, args.rollout_steps, args.eval_steps) < 1:
        parser.error('cycles, iterations, rollout-steps and eval-steps must be positive')
    if args.rollout_steps < 4 or args.rollout_steps % 4:
        parser.error('rollout-steps must be a positive multiple of four')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate must be finite and positive')
    if not math.isfinite(args.noise_std) or args.noise_std <= 0:
        parser.error('noise-std must be finite and positive')
    if args.max_minutes is not None and (not math.isfinite(args.max_minutes) or args.max_minutes <= 0):
        parser.error('max-minutes must be finite and positive')
    if min(args.train_seed, args.eval_seed) < 0:
        parser.error('seeds must be nonnegative')
    if args.train_seed <= args.eval_seed < args.train_seed + args.cycles:
        parser.error('eval-seed must differ from every training cycle seed')
    name = Path(args.name)
    if name.is_absolute() or len(name.parts) != 1 or name.name in ('.', '..') or ':' in args.name:
        parser.error('name must be a single new directory name, without a path')
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    config = read_json(ROOT / 'config.json')
    if args.eval_steps < int(config.get('episode_steps', 600)):
        parser.error('eval-steps must cover at least one full episode')
    if (RUNS / args.name).exists():
        parser.error('Run directory already exists; use a new --name to preserve all earlier results')
    return args


class TrainingLoop:
    def __init__(self, args):
        self.args = args
        self.directory = RUNS / args.name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.stop_file = self.directory / 'STOP'
        self.config_hash = file_hash(ROOT / 'config.json')
        self.best_checkpoint = args.checkpoint
        self.best_summary = None
        self.interrupted = False
        self.env = os.environ.copy()
        if os.name == 'nt':
            platform_defaults = {
                'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMNI_USER_HOME': r'E:\IsaacLab\userdata',
                'OV_USER_HOME': r'E:\IsaacLab\userdata', 'TEMP': r'E:\IsaacLab\tmp',
                'TMP': r'E:\IsaacLab\tmp', 'OMP_NUM_THREADS': '4',
            }
        else:
            cache_root = Path.home() / '.cache' / 'isaaclab'
            platform_defaults = {
                'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMNI_USER_HOME': str(cache_root / 'userdata'),
                'OV_USER_HOME': str(cache_root / 'userdata'), 'TEMP': tempfile.gettempdir(),
                'TMP': tempfile.gettempdir(), 'OMP_NUM_THREADS': '4',
            }
        for key, value in platform_defaults.items():
            self.env.setdefault(key, value)
        self.manifest = {
            'status': 'initialized', 'started_at': utc_now(), 'updated_at': utc_now(),
            'python': sys.executable, 'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            'config_sha256': self.config_hash,
            'source_sha256': {name: file_hash(ROOT / name) for name in
                              ('train.py', 'environment.py', 'policy.py', 'rewards.py', 'iterative_training.py')},
            'input_checkpoint': str(args.checkpoint), 'input_checkpoint_sha256': file_hash(args.checkpoint),
            'best_checkpoint': str(args.checkpoint), 'best_evaluation': None,
            'best_checkpoint_evaluated': False, 'selection_order': ['success_rate', 'max_hold_steps', 'max_bilateral_height_m_if_zero_success_else_max_height_m'],
            'validation_seed_used_for_selection': args.eval_seed, 'independent_final_test_completed': False,
            'task_success_claim': False, 'cycles': [], 'jobs': [],
            'stop_file': str(self.stop_file), 'soft_budget_checked': 'between cycles only',
        }
        (self.directory / 'config.json').write_bytes((ROOT / 'config.json').read_bytes())
        self.save()

    def save(self):
        jobs = self.manifest['jobs']
        self.manifest['updated_at'] = utc_now()
        self.manifest['wall_elapsed_s'] = time.monotonic() - self.started
        self.manifest['cost'] = {
            'completed_training_steps': sum(job.get('steps_exact', 0) for job in jobs if job['mode'] == 'train'),
            'partial_training_completed_rollout_steps_lower_bound': sum(job.get('partial_completed_rollout_steps', 0) for job in jobs if job['mode'] == 'train'),
            'completed_evaluation_steps': sum(job.get('steps_exact', 0) for job in jobs if job['mode'] == 'evaluate'),
            'training_child_wall_s': sum(job.get('wall_elapsed_s', 0.) for job in jobs if job['mode'] == 'train'),
            'evaluation_child_wall_s': sum(job.get('wall_elapsed_s', 0.) for job in jobs if job['mode'] == 'evaluate'),
            'billing_cost': 0., 'billing_note': 'Local machine only; electricity cost is not estimated.',
        }
        atomic_json(self.directory / 'manifest.json', self.manifest)
        atomic_json(self.directory / 'status.json', {key: self.manifest.get(key) for key in
            ('status', 'updated_at', 'wall_elapsed_s', 'active_job', 'active_job_progress', 'best_checkpoint', 'best_evaluation',
             'best_checkpoint_evaluated', 'task_success_claim', 'cost', 'stop_reason', 'error')})

    def stop_requested(self) -> bool:
        return self.interrupted or self.stop_file.exists()

    def stop(self, reason: str):
        self.manifest.update(status='stopped', stop_reason=reason, active_job=None, active_job_progress=None)
        self.save()

    def child(self, label: str, mode: str, checkpoint: Path, seed: int) -> Path:
        if file_hash(ROOT / 'config.json') != self.config_hash:
            raise RuntimeError('config.json changed during the loop; refusing to mix experiment contracts')
        output = self.directory / label
        output.mkdir(exist_ok=False)
        steps = self.args.rollout_steps if mode == 'train' else self.args.eval_steps
        command = [sys.executable, '-u', str(ROOT / 'train.py'), '--mode', mode,
                   '--checkpoint', str(checkpoint), '--name', f'{self.args.name}/{label}',
                   '--seed', str(seed), '--steps', str(steps)]
        if mode == 'train':
            command += ['--iterations', str(self.args.iterations), '--learning-rate', str(self.args.learning_rate),
                        '--noise-std', str(self.args.noise_std)]
        job = {'label': label, 'mode': mode, 'command': command, 'started_at': utc_now(),
               'log': str(output / 'process.log'), 'input_checkpoint': str(checkpoint), 'status': 'running'}
        self.manifest['jobs'].append(job)
        self.manifest.update(status='training' if mode == 'train' else 'evaluating', active_job=label, active_job_progress=None)
        self.save()
        started = time.monotonic()
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (output / 'process.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            job['pid'] = process.pid
            self.save()
            while True:
                try:
                    returncode = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        progress = read_json(output / 'progress.json')
                        # Preserve the last complete update during a concurrent write.
                        json.dumps(progress, allow_nan=False)
                        self.manifest['active_job_progress'] = progress
                    except (OSError, ValueError):
                        pass
                    self.save()
                except KeyboardInterrupt:
                    # Complete the current Isaac child and preserve its artifacts.
                    self.interrupted = True
                    self.manifest['status'] = 'stopping_after_current_child'
                    self.save()
        job.update(returncode=returncode, ended_at=utc_now(), wall_elapsed_s=time.monotonic() - started)
        self.manifest.update(active_job=None, active_job_progress=None)
        job['status'] = 'failed'
        result_file = output / 'result.json'
        result = read_json(result_file) if result_file.exists() else {}
        if result.get('status') == 'completed':
            job['steps_exact'] = int(result['steps_total'])
            job['sim_and_update_elapsed_s'] = float(result['elapsed_s'])
        elif mode == 'train':
            job['partial_completed_rollout_steps'] = len(result.get('iterations', [])) * self.args.rollout_steps
        job['status'] = 'completed' if returncode == 0 and result.get('status') == 'completed' else 'failed'
        self.save()
        if job['status'] != 'completed':
            raise RuntimeError(f'{label} failed (exit {returncode}); see {output / "process.log"}')
        if result.get('config_sha256') != self.config_hash:
            raise RuntimeError(f'{label} returned a different config hash')
        expected_steps = steps * self.args.iterations if mode == 'train' else steps
        if job['steps_exact'] != expected_steps:
            raise RuntimeError(f'{label} completed {job["steps_exact"]} steps, expected {expected_steps}')
        return output

    def select(self, checkpoint: Path, summary: dict):
        self.best_checkpoint = checkpoint.resolve()
        self.best_summary = summary
        self.manifest.update(best_checkpoint=str(self.best_checkpoint), best_evaluation=summary,
                             best_checkpoint_sha256=file_hash(self.best_checkpoint), best_checkpoint_evaluated=True)
        temporary = self.directory / 'best_checkpoint.txt.tmp'
        temporary.write_text(str(self.best_checkpoint) + '\n', encoding='utf-8')
        temporary.replace(self.directory / 'best_checkpoint.txt')
        self.save()

    def run(self):
        if self.stop_requested():
            self.stop('STOP file present before baseline evaluation')
            return
        baseline = self.child('baseline_evaluation', 'evaluate', self.best_checkpoint, self.args.eval_seed)
        self.select(self.best_checkpoint, summarize_evaluation(baseline))
        for cycle in range(self.args.cycles):
            if self.stop_requested():
                self.stop('STOP file or keyboard interruption; current child finished')
                return
            if self.args.max_minutes is not None and (time.monotonic() - self.started) >= self.args.max_minutes * 60:
                self.stop('Soft wall-time budget reached between cycles')
                return
            record = {'cycle': cycle + 1, 'starting_checkpoint': str(self.best_checkpoint), 'status': 'training'}
            self.manifest['cycles'].append(record)
            training = self.child(f'cycle_{cycle + 1:03d}_train', 'train', self.best_checkpoint, self.args.train_seed + cycle)
            candidate = training / 'checkpoint.pt'
            if not candidate.is_file():
                raise RuntimeError(f'Training completed without a checkpoint: {training}')
            record.update(candidate_checkpoint=str(candidate), status='awaiting_evaluation')
            self.save()
            if self.stop_requested():
                self.stop('STOP requested after training; unevaluated candidate preserved but not selected')
                return
            evaluation = self.child(f'cycle_{cycle + 1:03d}_evaluation', 'evaluate', candidate, self.args.eval_seed)
            summary = summarize_evaluation(evaluation)
            selected = selection_score(summary) > selection_score(self.best_summary)
            if selected:
                self.select(candidate, summary)
            record.update(status='completed', evaluation=summary, selected_as_best=selected,
                          best_checkpoint_after_cycle=str(self.best_checkpoint))
            self.save()
            print(json.dumps({'cycle': cycle + 1, 'selected': selected, 'evaluation': summary,
                              'best_checkpoint': str(self.best_checkpoint)}, ensure_ascii=False), flush=True)
        if self.stop_requested():
            self.stop('STOP requested; current evaluation finished')
        else:
            self.manifest.update(status='completed', active_job=None, active_job_progress=None, ended_at=utc_now())
            self.save()


def main(argv=None) -> int:
    args = parse_args(argv)
    loop = TrainingLoop(args)
    try:
        loop.run()
        return 0
    except Exception as error:
        for job in loop.manifest['jobs']:
            if job.get('status') == 'running':
                job.update(status='failed', error=f'{type(error).__name__}: {error}')
        loop.manifest.update(status='failed', active_job=None, active_job_progress=None, error=f'{type(error).__name__}: {error}',
                             ended_at=utc_now())
        (loop.directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        loop.save()
        print(f'Training loop stopped: {error}', file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        loop.stop('Keyboard interruption between child processes')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
