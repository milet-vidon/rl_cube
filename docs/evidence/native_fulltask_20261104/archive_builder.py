"""Preserve completed failed evidence, with byte-verifiable gzip round trips."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil

ROOT = Path('/home/brainco/rl_cube/github_project').resolve()
WORK = ROOT / 'work/teacher_optimization_20261103'
RUN = ROOT / 'revotron_cube_rl/runs/teacher_native_fulltask_v18_20261104'
FINITE = WORK / 'native_initial_face90_probe_v1_20261104'
OUT = ROOT / 'docs/evidence/native_fulltask_20261104'


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def read(path):
    return json.loads(Path(path).read_text())


result = read(RUN / 'result.json')
task = result['task_episode']
assert result['steps'] == 3017 and task['completed_control_steps'] == 3017
assert task['terminated'] and task['done'] and not task['truncated'] and not task['success']
assert task['reason'] == 'left_grasp_lost_during_opposition'
assert result['face_turn_validated'] is False and result['push']['turn_started'] is False
candidate = read(FINITE / 'candidate/result.json')
failed = {key: value for key, value in candidate['checks'].items() if value['passed'] is not True}
assert set(failed) == {'finite_face_marker_permutation_correct'}
assert not (FINITE / 'candidate/trajectory.json').exists()
pose = candidate['checks']['finite_pose_representation_correct']
assert pose['passed'] and pose['detail']['max_center_marker_error_m'] > 1e-6
assert 'finite_face_marker_permutation_correct' in candidate['error']
finite_source = FINITE / 'candidate/source/finite_angle_cube_native_probe.py'
source_text = finite_source.read_text()
assert source_text.index('check("finite_face_marker_permutation_correct"') < source_text.index('sim.step(render=False)')

audit_expected = {
    'native_fulltask_v18_contact_failure_diagnosis.json': 'fc7e3edcc1892d3bc696527d8362460de68d529586b3fa0661c9a0b3f46bb312',
    'native_fulltask_v18_final_audit.json': '250225211b42f45c25faaf760038b5a3e6d417ac383fb4ad3ae19e465735e281',
    'finite_native_pose_probe_independent_review_20261104.json': '0f7435e3ee0183882b86f3c0d4ed5b18a588d78b48715c1ad32e8b8f2c42dcf1',
}
for name, expected in audit_expected.items():
    assert sha(WORK / name) == expected, name
final_audit = read(WORK / 'native_fulltask_v18_final_audit.json')
assert final_audit['full_task_events']['no_steps_after_task_terminal']
assert final_audit['full_task_events']['transitions'] == 3017

OUT.mkdir(parents=True, exist_ok=False)
entries = []


def preserve(source, destination):
    source = source.resolve()
    assert source.is_relative_to(ROOT) and source.is_file() and not source.is_symlink()
    relative = Path(destination)
    assert not relative.is_absolute() and '..' not in relative.parts
    before = sha(source)
    compressed = source.suffix in ('.json', '.jsonl') and source.stat().st_size >= 512 * 1024
    if compressed:
        relative = Path(str(relative) + '.gz')
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        with source.open('rb') as reader, target.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
        with gzip.open(target, 'rb') as reader:
            restored = digest_stream(reader)
    else:
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
        restored = sha(target)
    assert before == restored == sha(source), str(source)
    entries.append({'source_path': str(source.relative_to(ROOT)),
                    'archive_path': str(relative), 'encoding': 'gzip' if compressed else 'identity',
                    'source_bytes': source.stat().st_size, 'archive_bytes': target.stat().st_size,
                    'source_sha256': before, 'archive_sha256': sha(target),
                    'decompressed_sha256': restored, 'source_unchanged_after_copy': True})


for source_root, folder in ((RUN, 'v18_run'), (FINITE, 'finite90_v1_run')):
    files = [p for p in sorted(source_root.rglob('*')) if p.is_file()]
    assert not any('__pycache__' in p.parts for p in files)
    for path in files:
        preserve(path, Path(folder) / path.relative_to(source_root))

audits = list(audit_expected) + [
    'audit_native_fulltask_v18.py', 'diagnose_native_v18_contact_failure.py',
    'native_fulltask_v18_final_audit_console.txt',
    'native_fulltask_v18.log', 'native_fulltask_v18_queue.py', 'native_fulltask_v18_queue.json',
    'native_fulltask_v18_queue.log',
    'finite_angle_native_probe_protocol_v1_20261104.json',
    'native_finite90_after_v18_queue.py', 'native_finite90_after_v18_queue.json',
    'native_finite90_after_v18.log', 'native_finite90_queue.log',
    'test_finite_cube_pose_math.py',
]
for name in audits:
    preserve(WORK / name, Path('audits_and_launch') / name)
preserve(Path(__file__), 'archive_builder.py')

# Record original manifests without rewriting paths or hashes. Every compressed
# member maps back to the original bytes via archive_manifest.json.
source_entries = {e['source_path']: e for e in entries}
declared_checks = []
original_manifest = read(RUN / 'artifact_manifest.json')
for name, expected in original_manifest['sha256'].items():
    rel = str((RUN / name).relative_to(ROOT))
    entry = source_entries.get(rel)
    declared_checks.append({'source_path': rel, 'declared_sha256': expected,
                            'actual_sha256': None if entry is None else entry['source_sha256'],
                            'matched': entry is not None and entry['source_sha256'] == expected})
assert all(item['matched'] for item in declared_checks), 'Original v18 artifact manifest mismatch'

finite_source_checks = []
for phase in ('baseline', 'candidate'):
    report = read(FINITE / phase / 'result.json')
    for original_path, expected in report['provenance_sha256'].items():
        source = Path(original_path)
        snapshot = FINITE / phase / 'source' / source.name
        checked = snapshot if snapshot.is_file() else source
        actual = sha(checked)
        finite_source_checks.append({'phase': phase, 'original_path': original_path,
                                     'verified_path': str(checked.relative_to(ROOT)),
                                     'expected_sha256': expected, 'actual_sha256': actual,
                                     'matched': actual == expected})
assert all(item['matched'] for item in finite_source_checks)

asset_hashes = read(RUN / 'runtime_robot_asset.json')['usd_layers_sha256'].copy()
for path, expected in candidate['provenance_sha256'].items():
    if path.endswith(('.usda', '.usd', '.xml')):
        asset_hashes[path] = expected
external_assets = []
for path, expected in sorted(asset_hashes.items()):
    actual = sha(path)
    assert actual == expected, path
    external_assets.append({'path': path, 'declared_sha256': expected, 'verified_sha256': actual,
                            'bytes': Path(path).stat().st_size, 'asset_bytes_archived': False})

summary = {
    'schema': 'native_fulltask_v18_and_finite90_v1_failure_archive_v1',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Completed failed development evidence. No source or historical result edits; no simulation.',
    'v18': {'run': RUN.name, 'steps': 3017, 'task_episode': task,
            'failure': result['failure'], 'turn_started': False, 'face_turn_validated': False,
            'dynamic_grasp_passed': result['dynamic_grasp_passed'],
            'center_transport_passed': result['center_transport_passed'],
            'no_steps_after_terminal': True, 'native_verification': final_audit['native_verification']},
    'finite90_v1': {'workflow': read(FINITE / 'workflow.json'),
                    'candidate_status': candidate['status'], 'failed_checks': failed,
                    'initial_pose_errors': pose['detail'],
                    'candidate_physics_steps': 0,
                    'zero_step_evidence': 'Recorded traceback fails the marker initialization check before the first sim.step; no candidate trajectory exists. This is a control-flow inference, not a sampled dynamics counter.',
                    'baseline_physics_steps': read(FINITE / 'baseline/result.json').get('baseline_physics_steps'),
                    'pre_execution_peer_review': 'Host review PASS only; actual candidate initialization subsequently FAILED.',
                    'finite_dynamics_validated': False, 'contact_driven_turn_validated': False},
    'original_v18_artifact_manifest': {'checked_entries': len(declared_checks),
                                       'all_match': True, 'checks': declared_checks},
    'finite_snapshot_source_checks': finite_source_checks,
    'external_asset_hashes': external_assets,
    'asset_scope': 'Runtime generated USD, model/contract metadata and all recorded asset hashes are archived. Large external source robot/cube assets remain at the verified original paths.',
    'no_full_face_turn_or_six_face_solution_success_claim': True,
}
(OUT / 'failure_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
readme = '''# Native FullTask v18 and finite90 v1 failed evidence

v18 completed 3017 control steps and terminated normally with
`left_grasp_lost_during_opposition`. Pickup and centered transport passed;
opposed grasp did not pass and face turning never started. The task outcome is
failure, with no transitions after termination. This is not a trained policy.

finite90 v1's paired ordinary-body baseline completed. Its passive-articulation
candidate failed the initial center-marker permutation check: 2.5946666986 µm
versus the unchanged 1 µm bound. The 0.0055251462° orientation error passed.
The failure occurs before the first candidate physics step. Host-review PASS
does not mean initialization, finite dynamics, or contact-driven turning passed.

All completed run files, source snapshots, configuration, original manifests,
native verification, finite calibration/results, audits and launch evidence
are preserved. JSON/JSONL members at least 512 KiB use deterministic gzip.
`archive_manifest.json` records original/compressed/decompressed SHA256 and byte
counts. Original manifests remain byte-identical and reference original names;
decompress the corresponding `.gz` member to recover those exact bytes.
`failure_summary.json` records explicit checks and external asset hashes.
No old result or source was modified; no git staging/commit was performed.
'''
(OUT / 'README.md').write_text(readme)
generated = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in (OUT / 'failure_summary.json', OUT / 'README.md')}
manifest = {'schema': 'sha256_lossless_completed_evidence_archive_v1',
            'source_files': entries, 'generated_files': generated,
            'source_file_count': len(entries),
            'total_source_bytes': sum(e['source_bytes'] for e in entries),
            'total_archived_source_bytes': sum(e['archive_bytes'] for e in entries),
            'all_decompressed_hashes_equal_original': True,
            'all_source_files_unchanged_after_copy': True,
            'original_v18_manifest_all_entries_verified': True}
(OUT / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps({'archive': str(OUT), 'manifest_sha256': sha(OUT / 'archive_manifest.json'),
                  'files': len(entries), 'source_bytes': manifest['total_source_bytes'],
                  'archived_bytes': manifest['total_archived_source_bytes']}), flush=True)
