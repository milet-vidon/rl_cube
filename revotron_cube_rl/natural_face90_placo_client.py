"""One isolated Placo CPU process per rollout worker. Batch all active lanes per frame."""
import json
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PREFIX = HERE.parent / 'work' / 'placo-0.9.23-cp312'
WIDTH = 56 + 3 + 9 + 1
N_LANES = struct.Struct('!I')


def child_command(mode):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('The pinned Placo runtime requires CPython 3.12')
    environment = dict(os.environ)
    # The Isaac process and the reference solver use different native libraries.
    # Set the search path before starting the dedicated interpreter.
    environment.pop('LD_PRELOAD', None)
    environment['LD_LIBRARY_PATH'] = os.pathsep.join((str(PREFIX / 'lib'), str(Path(sys.base_prefix) / 'lib')))
    command = [sys.executable, '-I', '-B', '-u', str(Path(__file__).resolve()), mode]
    return command, environment


def self_check():
    manifest = (PREFIX / 'runtime_manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != 'f14ac6cbe5d9877644be01e9ab840fa138e1e1d3be9ed3016f6ac513a7f0bd2a':
        raise ValueError('Changed Placo runtime manifest')
    for name, expected in json.loads(manifest).items():
        with (PREFIX / name).open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
                raise ValueError(f'Changed Placo runtime file: {name}')
    command, env = child_command('check')
    result = subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, timeout=60)
    return json.loads(result.stdout)


class PlacoClient:
    def __init__(self, joint_names, dt):
        command, env = child_command('serve')
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            self.process.stdin.write(json.dumps([list(joint_names), float(dt)]).encode() + b'\n')
            self.process.stdin.flush()
            self.receipt = json.loads(self.process.stdout.readline())
        except Exception:
            self.close()
            raise

    def solve_batch(self, q_measured, position_delta_base, desired_palm_R_base, recovering):
        q = np.asarray(q_measured)
        n = len(q)
        delta, rotation, recovery = map(np.asarray, (position_delta_base, desired_palm_R_base, recovering))
        if (q.shape != (n, 56) or delta.shape != (n, 3) or rotation.shape != (n, 3, 3)
                or recovery.shape != (n,) or n > 32):
            raise ValueError('Unexpected Placo lane inputs')
        if not n:
            return np.empty((0, 7))
        frame = np.column_stack((q, delta, rotation.reshape(n, 9), recovery)).astype('<f8', copy=False)
        self.process.stdin.write(N_LANES.pack(n) + frame.tobytes())
        self.process.stdin.flush()
        data = self.process.stdout.read(n * 7 * 8)
        if len(data) != n * 7 * 8:
            raise RuntimeError('Placo worker stopped during its solve; see the rollout log')
        return np.frombuffer(data, dtype='<f8').reshape(n, 7).copy()

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def serve(mode):
    # C++ and Boost can write diagnostics to stdout. Keep the protocol on its own fd.
    pipe = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.path[:0] = [str(PREFIX / 'lib/python3.12/site-packages'), str(HERE)]
    import natural_face90_placo as backend
    if mode == 'check':
        pipe.write(json.dumps(backend.self_check(), allow_nan=False).encode() + b'\n')
        return
    if mode != 'serve':
        raise ValueError('Unknown Placo worker mode')
    stream = sys.stdin.buffer
    names, dt = json.loads(stream.readline())
    solver = backend.PlacoArm(names, dt)
    pipe.write(json.dumps(solver.receipt, allow_nan=False).encode() + b'\n')
    while header := stream.read(N_LANES.size):
        n, = N_LANES.unpack(header)
        if not 1 <= n <= 32:
            raise ValueError('Unexpected Placo batch size')
        frame = np.frombuffer(stream.read(n * WIDTH * 8), dtype='<f8').reshape(n, WIDTH)
        if not np.isfinite(frame).all() or not np.isin(frame[:, -1], [0, 1]).all():
            raise ValueError('Nonfinite Placo inputs or invalid recovery mask')
        q = solver.solve_batch(frame[:, :56], frame[:, 56:59], frame[:, 59:68].reshape(n, 3, 3),
                               frame[:, -1].astype(bool))
        result = np.asarray(q, dtype='<f8')
        if result.shape != (n, 7) or not np.isfinite(result).all():
            raise ValueError('Placo returned invalid joint targets')
        pipe.write(result.tobytes())


if __name__ == '__main__':
    serve(sys.argv[1])
