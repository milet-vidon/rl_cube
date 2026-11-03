"""Build a reproducible certificate from complete, frozen teacher episodes."""
from pathlib import Path
from evidence_paths import local_path, verified_sha, verify_source
import argparse
import hashlib
import json
import numpy as np

from palm_handover_audit import assess_handover

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return verified_sha(path,ROOT)


def build_certificate(protocol_path, runs):
    protocol_path = local_path(protocol_path,ROOT)
    runs = [local_path(run,ROOT) for run in runs]
    protocol = json.loads(protocol_path.read_text())
    failures, trials, seen = [], [], set()
    sources = protocol['source_sha256']
    for name, expected in sources.items():
        if sha256(ROOT / name) != expected:
            failures.append('current_source_changed:' + name)
    for run in map(Path, runs):
        experiment = json.loads((run / 'experiment.json').read_text())
        if experiment.get('development_resume') or not experiment.get('regrasp'):
            failures.append('ineligible_episode:' + str(run))
            continue
        actual_shift = [float(v) for v in experiment['shift'].split(',')]
        if actual_shift != protocol['shift_after_tilt_m']:
            failures.append('wrong_post_tilt_shift:' + str(run))
        runtime = json.loads((run / 'runtime_config.json').read_text())
        for key, expected in protocol.get('runtime_contract', {}).items():
            if runtime.get(key) != expected:
                failures.append('runtime_contract_changed:' + str(run) + ':' + key)
        for name, expected in sources.items():
            if experiment['source_sha256'].get(name) != expected or sha256(run / name) != expected:
                failures.append('experiment_source_changed:' + str(run / name))
        config = json.loads((run / 'config.json').read_text())
        for result in json.loads((run / 'result.json').read_text())['trials']:
            seed = result['seed']
            if seed in seen:
                failures.append('duplicate_seed:' + str(seed))
            seen.add(seed)
            split = 'independent' if seed in protocol['reserved_independent_seeds'] else 'development'
            path = run / f'trajectory_{seed}.json'
            rows = json.loads(path.read_text())
            measured = assess_handover(rows, config['joint_names'], regrasp=True, full_episode=True)
            accepted = bool(result['accepted'] and measured['accepted'] and not result.get('development_resume'))
            if not accepted:
                failures.append('episode_failed:' + str(seed))
            arms = [i for i, name in enumerate(config['joint_names']) if name.startswith('Joint')]
            q = np.asarray([r['q'] for r in rows])
            limits = np.array([config['arm_soft_limits'][config['joint_names'][i]] for i in arms])
            clearance = np.minimum(q[:, arms] - limits[:, 0], limits[:, 1] - q[:, arms])
            trials.append(dict(seed=seed, split=split, accepted=accepted, run=str(run.resolve()),
                               trajectory_sha256=sha256(path), measured=measured,
                               config_sha256=sha256(run / 'config.json'),
                               runtime_config_sha256=sha256(run / 'runtime_config.json'),
                               asset_contract_sha256=sha256(run / 'asset_contract.json'),
                               minimum_actual_arm_soft_limit_clearance_deg=float(np.degrees(clearance.min())),
                               soft_limit_hit_steps=sum(bool(r['arm_soft_limit_hit']) for r in rows),
                               hard_limit_hit_steps=sum(any(r['hard_limit_hit']) for r in rows),
                               dropped=any(r['drop'] for r in rows)))
    required = set(protocol['reserved_independent_seeds']) | set(protocol['development_seeds'])
    if required != seen:
        failures.append('protocol_seed_mismatch:' + str(sorted(required ^ seen)))
    baseline = json.loads((ROOT / 'opposition_gate_certificate.json').read_text())
    for path, expected in baseline['assets_sha256'].items():
        if sha256(path) != expected:
            failures.append('asset_changed:' + path)
    independent = [t for t in trials if t['split'] == 'independent']
    return dict(passed=not failures, kind=protocol['method'], task=protocol['task'],
                protocol_sha256=sha256(protocol_path), failures=failures, trials=trials,
                accepted=sum(t['accepted'] for t in trials), total=len(trials),
                independent_accepted=sum(t['accepted'] for t in independent), independent_total=len(independent),
                source_sha256=sources, assets_sha256=baseline['assets_sha256'],
                gate_source_sha256=sha256(__file__), soft_limit_version='revo3_user_soft_limits_v2',
                neural_policy_trained=False, layer_turn_validated=False, real_robot_tested=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    certificate = build_certificate(args.protocol, args.runs)
    args.out.write_text(json.dumps(certificate, indent=2))
    print(json.dumps({k: certificate[k] for k in ('passed', 'accepted', 'total', 'independent_accepted', 'independent_total', 'failures')}))
    raise SystemExit(0 if certificate['passed'] else 1)
