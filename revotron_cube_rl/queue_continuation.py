"""Queue one bounded continuation after an existing training loop completes.

The existing run is read only. Its STOP, failure, stale heartbeat, or changed
training sources prevent a new loop from starting. After launch, this queue's
STOP is forwarded only to the newly created run, so its current Isaac child can
finish normally. This script never kills a process or changes checkpoints.

Importing this module does not launch a queue, Python worker, or Isaac process.
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

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('config.json', 'train.py', 'environment.py', 'policy.py', 'rewards.py', 'iterative_training.py')
POLL_SECONDS = 15
WAIT_LIMIT_SECONDS = 150 * 60
HEARTBEAT_LIMIT_SECONDS = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_manifest(path: Path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Manifest is not an object')
        json.dumps(data, allow_nan=False)
        return data, None
    except (OSError, ValueError) as error:
        return None, f'{type(error).__name__}: {error}'


def heartbeat_age(manifest: dict) -> float:
    stamp = datetime.fromisoformat(str(manifest['updated_at']).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Manifest heartbeat must have a timezone')
    return max(0., (datetime.now(timezone.utc) - stamp).total_seconds())


def single_name(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and len(path.parts) == 1 and path.name not in ('.', '..') and ':' not in value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--after-run', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--rollout-steps', type=int, default=64)
    parser.add_argument('--eval-steps', type=int, default=1800)
    parser.add_argument('--train-seed', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=10001)
    parser.add_argument('--max-minutes', type=float, default=120.)
    args = parser.parse_args(argv)
    if not single_name(args.after_run) or not single_name(args.name) or args.after_run == args.name:
        parser.error('after-run and name must be distinct single directory names')
    if min(args.cycles, args.iterations, args.rollout_steps, args.eval_steps) < 1 or args.rollout_steps % 4:
        parser.error('Budgets must be positive; rollout-steps must be a multiple of four')
    if min(args.train_seed, args.eval_seed) < 0 or args.train_seed <= args.eval_seed < args.train_seed + args.cycles:
        parser.error('Seeds must be nonnegative; eval-seed must differ from all training seeds')
    if not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        parser.error('max-minutes must be finite and positive')
    runs = (ROOT / 'runs').resolve()
    prior = runs / args.after_run
    if not prior.is_dir() or prior.resolve() != prior:
        parser.error('after-run must be an existing direct run directory, not a redirected path')
    if (runs / args.name).exists() or (ROOT / 'continuations' / args.name).exists():
        parser.error('The target run or queue directory already exists; choose a new name')
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    if args.eval_steps < int(config.get('episode_steps', 600)):
        parser.error('eval-steps must cover at least one full episode')
    return args


class ContinuationQueue:
    def __init__(self, args):
        self.args = args
        self.root = ROOT.resolve()
        self.runs = (self.root / 'runs').resolve()
        if not self.runs.is_relative_to(self.root):
            raise ValueError('runs is redirected outside the project')
        self.previous = self.runs / args.after_run
        self.target = self.runs / args.name
        self.directory = (self.root / 'continuations' / args.name).resolve()
        if not self.directory.is_relative_to(self.root / 'continuations'):
            raise ValueError('Queue directory is redirected outside continuations')
        if self.target.exists():
            raise ValueError('Target run already exists')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stop_file = self.directory / 'STOP'
        self.started = time.monotonic()
        self.last_good_previous = None
        self.initial_hashes = {name: sha256(self.root / name) for name in SOURCE_NAMES}
        self.manifest = {
            'status': 'waiting_previous_run', 'created_at': utc_now(), 'updated_at': utc_now(),
            'python': sys.executable, 'arguments': vars(args).copy(),
            'after_run': str(self.previous), 'target_run': str(self.target),
            'queue_stop_file': str(self.stop_file), 'previous_stop_file': str(self.previous / 'STOP'),
            'source_sha256': self.initial_hashes, 'config_sha256': self.initial_hashes['config.json'],
            'poll_seconds': POLL_SECONDS, 'wait_limit_seconds': WAIT_LIMIT_SECONDS,
            'previous_heartbeat_limit_seconds': HEARTBEAT_LIMIT_SECONDS,
            'budget': {'cycles': args.cycles, 'iterations_per_cycle': args.iterations,
                       'rollout_steps': args.rollout_steps, 'evaluation_steps_per_run': args.eval_steps,
                       'max_minutes': args.max_minutes, 'soft_time_limit_checked_between_cycles': True,
                       'planned_training_steps': args.cycles * args.iterations * args.rollout_steps},
            'worker_started': False, 'child_pid': None, 'source_best_checkpoint': None,
            'started_checkpoint': None, 'best_checkpoint': None, 'task_success_claim': False,
        }
        (self.directory / 'config.json').write_bytes((self.root / 'config.json').read_bytes())
        self.save()

    def save(self):
        self.manifest.update(updated_at=utc_now(), elapsed_s=time.monotonic() - self.started)
        atomic_json(self.directory / 'manifest.json', self.manifest)
        atomic_json(self.directory / 'status.json', {key: self.manifest.get(key) for key in
            ('status', 'updated_at', 'elapsed_s', 'after_run', 'target_run', 'worker_started', 'child_pid',
             'previous_status', 'previous_heartbeat_age_s', 'previous_read_error', 'started_checkpoint',
             'best_checkpoint', 'worker_status', 'worker_active_job', 'worker_active_job_progress',
             'worker_cost', 'stop_forwarded', 'stop_reason', 'error', 'returncode', 'task_success_claim')})

    def stop(self, reason: str):
        self.manifest.update(status='stopped', stop_reason=reason, ended_at=utc_now())
        self.save()

    def stop_reason(self):
        if self.stop_file.exists():
            return 'queue_STOP_requested'
        if (self.previous / 'STOP').exists():
            return 'previous_run_STOP_present'
        return None

    def checkpoint_in_runs(self, value: str) -> Path:
        checkpoint = Path(value).resolve(strict=True)
        if not checkpoint.is_file() or not checkpoint.is_relative_to(self.runs):
            raise ValueError('Checkpoint must be an existing file inside this project runs directory')
        return checkpoint

    def inspect_previous(self):
        manifest, error = read_manifest(self.previous / 'manifest.json')
        self.manifest['previous_read_error'] = error
        if manifest is not None:
            try:
                age = heartbeat_age(manifest)
            except (KeyError, TypeError, ValueError) as invalid:
                self.manifest['previous_read_error'] = f'Invalid heartbeat: {invalid}'
            else:
                self.last_good_previous = manifest
                self.manifest.update(previous_status=manifest.get('status'), previous_heartbeat_age_s=age,
                                     previous_manifest_updated_at=manifest.get('updated_at'))
                return manifest, age
        if self.last_good_previous is not None:
            age = heartbeat_age(self.last_good_previous)
        else:
            age = time.monotonic() - self.started
        self.manifest['previous_heartbeat_age_s'] = age
        return None, age

    def wait_for_previous(self):
        while True:
            reason = self.stop_reason()
            if reason:
                self.stop(reason)
                return None
            if time.monotonic() - self.started > WAIT_LIMIT_SECONDS:
                self.stop('previous_run_wait_limit_exceeded')
                return None
            previous, age = self.inspect_previous()
            if previous is not None and previous.get('status') in ('failed', 'stopped'):
                self.stop('previous_run_' + previous['status'])
                return None
            # Completed loops no longer heartbeat; their age must not block a continuation.
            if previous is not None and previous.get('status') == 'completed':
                return previous
            if age > HEARTBEAT_LIMIT_SECONDS:
                self.stop('previous_run_heartbeat_stale_or_unreadable')
                return None
            self.manifest['status'] = 'waiting_previous_run'
            self.save()
            time.sleep(POLL_SECONDS)

    def forward_stop(self, reason: str):
        self.manifest.update(status='stopping_after_current_child', stop_reason=reason)
        if not self.target.exists():
            self.manifest['stop_forwarded'] = False
            return
        resolved = self.target.resolve(strict=True)
        if not resolved.is_dir() or resolved != self.target or not resolved.is_relative_to(self.runs):
            raise ValueError('Refusing to write STOP to a redirected or invalid target run')
        stop = resolved / 'STOP'
        try:
            with stop.open('x', encoding='utf-8') as stream:
                stream.write(f'Requested by continuation queue {self.args.name}: {reason}\n')
        except FileExistsError:
            pass
        self.manifest.update(stop_forwarded=True, forwarded_stop_path=str(stop))

    def observe_worker(self):
        worker, error = read_manifest(self.target / 'manifest.json')
        self.manifest['worker_read_error'] = error
        if worker is None:
            return None
        self.manifest.update(worker_status=worker.get('status'), worker_active_job=worker.get('active_job'),
                             worker_active_job_progress=worker.get('active_job_progress'), worker_cost=worker.get('cost'))
        if worker.get('best_checkpoint') and worker.get('best_checkpoint_evaluated'):
            try:
                candidate = self.checkpoint_in_runs(worker['best_checkpoint'])
            except (OSError, TypeError, ValueError) as invalid:
                self.manifest['worker_checkpoint_read_error'] = str(invalid)
            else:
                self.manifest['best_checkpoint'] = str(candidate)
                self.manifest['worker_checkpoint_read_error'] = None
        return worker

    def launch(self, previous: dict):
        reason = self.stop_reason()
        if reason:
            self.stop(reason)
            return
        if self.target.exists():
            self.stop('target_run_created_while_waiting')
            return
        current = {name: sha256(self.root / name) for name in SOURCE_NAMES}
        if current != self.initial_hashes:
            self.manifest['changed_sources'] = [name for name in SOURCE_NAMES if current[name] != self.initial_hashes[name]]
            self.stop('config_or_training_source_changed_since_queue_creation')
            return
        if previous.get('config_sha256') != self.initial_hashes['config.json']:
            self.stop('previous_run_config_mismatch')
            return
        checkpoint = self.checkpoint_in_runs(previous['best_checkpoint'])
        command = [sys.executable, '-u', str(self.root / 'iterative_training.py'),
                   '--checkpoint', str(checkpoint), '--name', self.args.name,
                   '--cycles', str(self.args.cycles), '--iterations', str(self.args.iterations),
                   '--rollout-steps', str(self.args.rollout_steps), '--eval-steps', str(self.args.eval_steps),
                   '--train-seed', str(self.args.train_seed), '--eval-seed', str(self.args.eval_seed),
                   '--max-minutes', str(self.args.max_minutes)]
        self.manifest.update(status='starting_worker', source_best_checkpoint=str(checkpoint),
                             started_checkpoint=str(checkpoint), started_checkpoint_sha256=sha256(checkpoint),
                             best_checkpoint=str(checkpoint), command=command, log=str(self.directory / 'worker.log'))
        self.save()
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (self.directory / 'worker.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=self.root, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            self.manifest.update(status='running', worker_started=True, child_pid=process.pid, worker_started_at=utc_now())
            while True:
                reason = self.stop_reason()
                if reason:
                    self.forward_stop(reason)
                self.observe_worker()
                self.save()
                try:
                    returncode = process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except KeyboardInterrupt:
                    # Turn a queue interruption into the same cooperative STOP request.
                    if not self.stop_file.exists():
                        self.stop_file.write_text('Keyboard interruption; finish current child before stopping.\n', encoding='utf-8')
            worker = self.observe_worker()
        self.manifest.update(returncode=returncode, ended_at=utc_now())
        if returncode != 0:
            self.manifest.update(status='failed', error='Continuation worker exited unsuccessfully; prior checkpoint remains available.')
        elif worker is not None and worker.get('status') == 'completed':
            self.manifest['status'] = 'completed'
        elif worker is not None and worker.get('status') == 'stopped':
            self.manifest.update(status='stopped', stop_reason=self.manifest.get('stop_reason') or worker.get('stop_reason'))
        else:
            self.manifest.update(status='failed', error='Worker exited without a completed/stopped manifest; prior checkpoint retained.')
        self.save()

    def run(self):
        previous = self.wait_for_previous()
        if previous is not None:
            self.launch(previous)


def main(argv=None) -> int:
    args = parse_args(argv)
    queue = ContinuationQueue(args)
    try:
        queue.run()
        return 1 if queue.manifest['status'] == 'failed' else 0
    except KeyboardInterrupt:
        queue.stop('queue_interrupted_before_worker_launch')
        return 130
    except Exception as error:
        queue.manifest.update(status='failed', error=f'{type(error).__name__}: {error}', ended_at=utc_now())
        (queue.directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        queue.save()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
