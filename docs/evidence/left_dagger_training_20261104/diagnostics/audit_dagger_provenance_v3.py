"""Independent CPU-only integrity/equivalence and serialization component audit."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
from types import SimpleNamespace

from dagger_collection_core_v1 import DaggerTransitionCollector, FrozenTeacherOracle, canonical
from dagger_projected_collection_v3 import (
    PROVENANCE_FILE, ProjectedDaggerCollector, expand_record_provenance, persist_provenance,
)
from test_dagger_collection_core_v1 import NAMES, Teacher, gate, observation

HERE = Path(__file__).resolve().parent
EXPECTED_WRAPPER = '268653ef77b2b500d5299ebb377a6311f98ed9d5199f0b345795b26a92669a33'
EXPECTED_FIXTURE = 'efad1567813765bb1e470326c1f72c05cff43f1e3b43fff942ecd3fec9ed18fb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InlineProjectedCollector(ProjectedDaggerCollector):
    """Same projection/transaction path, with the previous inline metadata only."""
    def __init__(self, *args, runtime, **kwargs):
        DaggerTransitionCollector.__init__(self, *args, **kwargs)
        self.runtime = runtime


def collector_result(kind, provenance, output, source, abort=False):
    state = observation()
    state['oracle_env_inputs'] = {}
    state['control_settings'] = {}
    teacher = Teacher(state)
    oracle = FrozenTeacherOracle(teacher, NAMES, lambda: state, lambda a: a[0],
        'host_only_decoder', lambda o, b, a: {'phase': a['name'], 'phase_progress': .5})
    runtime = SimpleNamespace(output=output,
        project_expert_target=lambda a: {'requested_target': a[0],
            'projected_target': [.003] + [0.] * 55,
            'projection_parameters_file_sha256': 'host_fixture_only'},
        verify_teacher_projected_target=lambda label: True)
    stream = io.StringIO()
    collector = kind(oracle, stream, 'independent_integrity_fixture', provenance,
        list(range(28)), runtime=runtime)
    received = []
    def actor(actor_input):
        received.append(deepcopy(actor_input))
        assert 'expert_supervision' not in actor_input
        assert 'run_provenance' not in actor_input
        return [.002] * 28
    prepared = collector.prepare(state, actor, gate(), execute_source=source,
        selection_reason='host_integrity_fixture', selection_probability=.5,
        selection_draw=.25)
    packet = collector.execution_packet()
    if abort:
        bad = deepcopy(state)
        bad['cube_position'][0] = float('nan')
        collector.abort('host_abort_integrity_fixture', bad)
    else:
        state['applied_target'] = [.001] * 56
        collector.complete(state, gate(), stage_complete=True)
    return {'prepared': prepared, 'packet': packet, 'actor_inputs': received,
            'record': json.loads(stream.getvalue()), 'teacher_query_count': teacher.tick}


def component_benchmark(operation, *, iterations=20, repeats=5):
    operation()  # one warmup, excluded
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        samples.append((time.perf_counter_ns() - started) / iterations / 1e6)
    return {'iterations_per_repeat': iterations, 'repeats': repeats,
            'milliseconds_per_operation': samples,
            'median_milliseconds': statistics.median(samples)}


def main():
    wrapper = HERE / 'dagger_projected_collection_v3.py'
    fixture_path = HERE / 'dagger_provenance_real_rows_fixture.json'
    assert sha(wrapper) == EXPECTED_WRAPPER
    assert sha(fixture_path) == EXPECTED_FIXTURE
    frozen_paths = [HERE / n for n in ('dagger_collection_core_v1.py',
        'left_dagger_baseline_runtime_v1.py', 'left_dagger_baseline_runtime_v2.py',
        'collect_left_dagger_baseline_v1.py', 'collect_left_dagger_baseline_v2.py')]
    frozen_before = {p.name: sha(p) for p in frozen_paths}
    fixture = json.loads(fixture_path.read_text())
    records = [item['record'] for item in fixture['selected']]
    provenance = records[0]['run_provenance']
    assert all(canonical(row['run_provenance']) == canonical(provenance) for row in records)
    directory = Path(tempfile.mkdtemp(prefix='dagger_v3_integrity_audit_', dir=HERE))
    assert directory.resolve().is_relative_to(HERE.resolve())
    ref = persist_provenance(directory, provenance)
    original_bytes = (directory / PROVENANCE_FILE).read_bytes()
    compact = [deepcopy(row) for row in records]
    for row in compact:
        row['run_provenance'] = deepcopy(ref)
    restored = [expand_record_provenance(row, directory) for row in compact]
    fields = ('actor_observation_pre', 'actor_observation_post', 'expert_supervision',
        'student_proposed_target', 'student_proposed_left_target',
        'selected_requested_target', 'actual_applied_target', 'executed_source',
        'teacher_intervention', 'pre_gates', 'post_gates', 'terminal', 'failure_reason')
    for old, new in zip(records, restored):
        assert canonical(old) == canonical(new), 'Whole actual record changed'
        for field in fields:
            assert canonical(old[field]) == canonical(new[field]), field

    semantic_cases = []
    for source, abort in (('teacher', False), ('student', False), ('teacher', True)):
        inline_dir = directory / ('inline_' + source + str(abort)); inline_dir.mkdir()
        compact_dir = directory / ('compact_' + source + str(abort)); compact_dir.mkdir()
        old = collector_result(InlineProjectedCollector, provenance, inline_dir, source, abort)
        new = collector_result(ProjectedDaggerCollector, provenance, compact_dir, source, abort)
        new['prepared'] = expand_record_provenance(new['prepared'], compact_dir)
        new['record'] = expand_record_provenance(new['record'], compact_dir)
        assert canonical(old) == canonical(new), 'Storage change altered collector data'
        assert old['teacher_query_count'] == new['teacher_query_count'] == 1
        semantic_cases.append({'source': source, 'abort': abort,
            'prepared_record_packet_actor_inputs_and_query_count_exact': True})

    rejected = {}
    def reject(name, function):
        try:
            function()
        except (ValueError, FileNotFoundError, FileExistsError, TypeError) as error:
            rejected[name] = {'rejected': True, 'error_type': type(error).__name__, 'reason': str(error)}
        else:
            raise AssertionError('Accepted tamper: ' + name)
    def changed_ref(key, value):
        row = deepcopy(compact[0]); row['run_provenance'][key] = value
        return row
    reject('wrong_file_SHA', lambda: expand_record_provenance(changed_ref('sha256', '0' * 64), directory))
    reject('wrong_physics_SHA', lambda: expand_record_provenance(changed_ref('physics_identity_sha256', '0' * 64), directory))
    reject('path_traversal', lambda: expand_record_provenance(changed_ref('file', '../' + PROVENANCE_FILE), directory))
    reject('unexpected_reference_field', lambda: expand_record_provenance(changed_ref('unchecked_extra', 1), directory))
    empty = directory / 'empty'; empty.mkdir()
    reject('missing_file', lambda: expand_record_provenance(compact[0], empty))
    reject('immutable_overwrite', lambda: persist_provenance(directory, provenance))
    assert (directory / PROVENANCE_FILE).read_bytes() == original_bytes
    (directory / PROVENANCE_FILE).write_bytes(original_bytes + b' ')
    reject('file_bytes_appended', lambda: expand_record_provenance(compact[0], directory))
    (directory / PROVENANCE_FILE).write_bytes(original_bytes)
    for case, mutation in (
        ('modified_physics_with_rehashed_file', lambda full: full['physics_identity'].update(environment_class='tampered')),
        ('missing_checkpoint_with_rehashed_file', lambda full: full.update(student_checkpoint_sha256='')),
        ('missing_sources_with_rehashed_file', lambda full: full.update(source_sha256={})),
    ):
        bad_full = deepcopy(provenance); mutation(bad_full)
        data = (canonical(bad_full) + '\n').encode()
        (directory / PROVENANCE_FILE).write_bytes(data)
        bad_record = changed_ref('sha256', hashlib.sha256(data).hexdigest())
        reject(case, lambda: expand_record_provenance(bad_record, directory))
        (directory / PROVENANCE_FILE).write_bytes(original_bytes)
    bad_abort = {'run_provenance': deepcopy(ref), 'record_type': 'abort',
                 'pending_transition': changed_ref('sha256', '0' * 64)}
    reject('nested_abort_reference_tamper', lambda: expand_record_provenance(bad_abort, directory))
    legacy = deepcopy(records[0])
    reject('unchecked_inline_record_rejected_at_v3_loader', lambda: expand_record_provenance(legacy, directory))

    # Deliberately excludes PhysX, teacher queries, neural inference, real disk
    # throughput, and complete collector time. It measures only the named CPU
    # copying/serialization components with two actual diagnostic row shapes.
    perf = []
    for full_row, small_row in zip(records, compact):
        full = component_benchmark(lambda: canonical(deepcopy(full_row)))
        small = component_benchmark(lambda: canonical(deepcopy(small_row)))
        loader = component_benchmark(lambda: expand_record_provenance(small_row, directory), iterations=5, repeats=3)
        perf.append({'step': full_row['step'],
            'full_row_canonical_bytes': len(canonical(full_row).encode()),
            'compact_row_canonical_bytes': len(canonical(small_row).encode()),
            'full_copy_and_canonical_serialization': full,
            'compact_copy_and_canonical_serialization': small,
            'verified_loader_read_hash_parse_and_expand': loader,
            'copy_serialization_component_ratio': full['median_milliseconds'] / small['median_milliseconds']})
    assert sha(wrapper) == EXPECTED_WRAPPER
    assert {p.name: sha(p) for p in frozen_paths} == frozen_before
    result = {'scope': 'independent offline integrity and CPU serialization audit; no Isaac run',
        'wrapper_sha256': EXPECTED_WRAPPER, 'real_fixture_sha256': EXPECTED_FIXTURE,
        'actual_source': fixture['source'], 'actual_source_steps': [r['step'] for r in records],
        'whole_actual_records_equal_after_expansion': True,
        'explicit_actor_expert_action_gate_fields_equal': list(fields),
        'collector_semantic_equivalence': semantic_cases, 'rejection_tests': rejected,
        'provenance_canonical_bytes': len(canonical(provenance).encode()),
        'immutable_file_bytes': len(original_bytes), 'reference_canonical_bytes': len(canonical(ref).encode()),
        'environment': {'platform': platform.platform(), 'python': platform.python_version()},
        'offline_component_benchmarks': perf, 'frozen_core_v1_v2_unchanged_sha256': frozen_before,
        'scratch_directory': str(directory),
        'limitations': ['No simulator wall-time speedup measured or claimed',
            'No complete training-loop speedup measured or claimed',
            'No disk throughput benchmark; serialization output remains in memory',
            'Loader intentionally validates file bytes and physics hash on each call',
            'Two real v2 rows provide realistic metadata/label shapes; v3 collector equivalence uses host mocks',
            'Content hashes provide integrity against the referenced manifest, not cryptographic authentication']}
    destination = HERE / 'dagger_provenance_v3_independent_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps({'report': str(destination), 'all_checks_passed': True,
        'tamper_and_invalid_input_cases': len(rejected),
        'byte_counts': [(x['full_row_canonical_bytes'], x['compact_row_canonical_bytes']) for x in perf],
        'offline_copy_serialization_component_ratio': [x['copy_serialization_component_ratio'] for x in perf]}, indent=2))


if __name__ == '__main__':
    main()
