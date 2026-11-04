"""Offline native Float3 replay. Never constructs Kit/SimulationApp/PhysX worlds."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np

# Loading the vector binding only needs its kernel package and libcarb.so on
# LD_LIBRARY_PATH (set by the launcher); importing isaacsim would be unnecessary.
ISAAC_ROOT = Path(importlib.util.find_spec('isaacsim').origin).parent
sys.path.insert(0, str(ISAAC_ROOT / 'kit/kernel/py'))
import carb

from contact_event_cache import ContactMultiplexer
from test_contact_event_cache import callbacks, fixture, resolver, assert_exact


class Float3Point:
    """Synthetic contact wrapper with genuine native vector payloads."""
    def __init__(self, values, fresh):
        self.values = tuple(float(x) for x in values)
        self.fresh = fresh
        self._impulse = carb.Float3(*self.values)
        self._position = carb.Float3(.5, .02, 1.25)
        self._normal = carb.Float3(0., 1., 0.)
        self.separation = -.000001

    @property
    def impulse(self):
        return carb.Float3(*self.values) if self.fresh else self._impulse

    @property
    def position(self):
        return carb.Float3(.5, .02, 1.25) if self.fresh else self._position

    @property
    def normal(self):
        return carb.Float3(0., 1., 0.) if self.fresh else self._normal


def replay(events, points, trials, fresh):
    headers, original_data = fixture(points)
    data = [Float3Point(point._impulse, fresh) for point in original_data]
    baseline, base_snapshot, base_reset, hashes = callbacks(
        np, SimpleNamespace(intToSdfPath=resolver))
    mux = ContactMultiplexer(resolver)
    cached, snapshot, reset, cache_hashes = callbacks(mux.numpy_proxy, mux.schema_proxy)
    tokens = [mux.subscribe(callback) for callback in cached]
    assert hashes == cache_hashes

    def run(mode, count):
        start = time.perf_counter()
        if mode == 'baseline':
            for _ in range(count):
                base_reset()
                for callback in baseline:
                    callback(headers, data)
        else:
            for _ in range(count):
                reset()
                mux.dispatch(headers, data)
        return time.perf_counter() - start

    # Both modes warm up, and measurement order alternates to reduce bias.
    run('baseline', 2)
    run('cache', 2)
    result = []
    for trial in range(trials):
        order = ('baseline', 'cache') if trial % 2 == 0 else ('cache', 'baseline')
        elapsed = {mode: run(mode, events) for mode in order}
        assert_exact(unittest.TestCase(), base_snapshot(), snapshot())
        result.append({'trial': trial, 'order': list(order), **elapsed,
                       'speedup': elapsed['baseline'] / elapsed['cache']})
    base_median = statistics.median(row['baseline'] for row in result)
    cache_median = statistics.median(row['cache'] for row in result)
    return {'point_properties': 'fresh native Float3 per access' if fresh else 'stable native Float3 per property',
            'trials': result, 'baseline_median_s': base_median,
            'cache_median_s': cache_median, 'median_speedup': base_median / cache_median,
            'all_output_types_order_float_bits_exact': True,
            'cache_stats_last_event': mux.last_stats, 'callback_AST_sha256': hashes}


def microbench(count):
    vector = carb.Float3(0.001, -0.0003333333333, 0.)
    out = {}
    for name, operation in (('np.asarray', np.asarray), ('np.array', np.array)):
        start = time.perf_counter()
        for _ in range(count):
            operation(vector)
        elapsed = time.perf_counter() - start
        array = operation(vector)
        out[name] = {'calls': count, 's': elapsed, 'ns_per_call': elapsed * 1e9 / count,
                     'dtype': str(array.dtype), 'shape': list(array.shape),
                     'strides': list(array.strides), 'owns_data': array.flags.owndata}
    array = np.asarray(vector)
    start = time.perf_counter()
    for _ in range(count):
        array.copy(order='K')
    elapsed = time.perf_counter() - start
    out['independent_array_copy'] = {'calls': count, 's': elapsed, 'ns_per_call': elapsed * 1e9 / count}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--events', type=int, default=20)
    p.add_argument('--points-per-header', type=int, default=1024)
    p.add_argument('--trials', type=int, default=3)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Unique output required')
    binding = next(ISAAC_ROOT.glob('extscache/omni.physx-*/omni/physx/bindings/_physx.pyi'))
    body = binding.read_text()
    start = body.index('class ContactData():')
    end = body.index('class ContactDataVector():', start)
    binding_class = body[start:end]
    for field in ('impulse', 'position', 'normal'):
        assert f'def {field}(self) -> carb._carb.Float3:' in binding_class
    root = Path(__file__).resolve().parent
    result = {
        'scope': 'offline native carb.Float3 payload replay of unchanged callback ASTs, synthetic headers/points',
        'events_per_trial': a.events, 'points_per_header': a.points_per_header, 'headers_per_event': 5,
        'physx_binding_path': str(binding), 'physx_binding_sha256': hashlib.sha256(binding.read_bytes()).hexdigest(),
        'ContactData_property_type': {'impulse': 'carb._carb.Float3', 'position': 'carb._carb.Float3', 'normal': 'carb._carb.Float3'},
        'native_type': str(type(carb.Float3(1., 2., 3.))),
        'microbench': microbench(200000),
        'replay': [replay(a.events, a.points_per_header, a.trials, fresh) for fresh in (False, True)],
        'norm_function_unchanged': True,
        'app_modules_loaded': [name for name in sys.modules if 'simulation_app' in name or name == 'omni.kit.app'],
        'source_sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in
                          ('contact_event_cache.py', 'test_contact_event_cache.py', 'benchmark_contact_cache_float3.py')},
        'limits': ['No native ContactDataVector or actual event delivery measured',
                   'Property identity/lifetime hypotheses tested in both stable and fresh modes, not inferred as actual binding behavior',
                   'No actual PhysX path resolver measured; synthetic resolver unchanged between modes',
                   'No global NumPy patch, native factory, original modules or controller changed',
                   'Offline replay timing cannot imply whole-simulation speedup'],
    }
    assert not result['app_modules_loaded']
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'replay')}, indent=2))
    for row in result['replay']:
        print(json.dumps({k: v for k, v in row.items() if k != 'callback_AST_sha256'}, indent=2))


if __name__ == '__main__':
    main()
