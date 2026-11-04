"""Strict completed-v3-run consumer for development-only recurrent supervision."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

import dagger_projected_collection_v3 as provenance_reader
import dagger_collection_core_v1 as collection_core
from dagger_collection_core_v1 import STATE_FIELDS, canonical, physics_fingerprint, validate_gate
from phase_residual_policy_v5 import features

REQUIRED_FILES = ('transitions.jsonl', 'trajectory.jsonl', 'collection_provenance.json', 'projection_parameters.json',
                  'result.json', 'config.json', 'runtime_protocol.json')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + text)))


def bits_equal(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32))


def verify_evidence(directory):
    """Used for both training and checkpoint loading; no unchecked references."""
    run = Path(directory).resolve()
    manifest = read_json(run / 'artifact_manifest.json')
    hashes = {}
    for name in REQUIRED_FILES:
        hashes[name] = sha(run / name)
        if hashes[name] != manifest['sha256'].get(name):
            raise ValueError('Completed run artifact hash mismatch: ' + name)
    with (run / 'transitions.jsonl').open() as stream:
        first = json.loads(next(stream))
    full = provenance_reader.expand_record_provenance(first, run)['run_provenance']
    family = physics_fingerprint(full['physics_identity'])
    if full['physics_identity']['config_sha256'] != hashes['config.json']:
        raise ValueError('Physics identity config hash differs from actual dataset config')
    if full.get('capture_begins_at_episode_reset') is not True or full.get('policy_observation_scope') != 'privileged_simulator_state_not_vision_actor':
        raise ValueError('Explicit reset-based privileged-state collection required')
    topology = full['physics_identity']['cube_constraint_topology']
    if topology.get('d6_damping_augmentation') is not False or topology.get('native_articulation_facade') is not False:
        raise ValueError('V5 baseline dataset cannot mix native/D6 physics domains')
    if full['physics_identity']['environment_class'] != 'articulated_cube_env.DynamicPickupEnv':
        raise ValueError('Unexpected baseline environment class')
    for module in (provenance_reader, collection_core):
        if sha(module.__file__) != full['source_sha256'][Path(module.__file__).name]:
            raise ValueError('Provenance consumer differs from the recorded v3 source')
    return {'directory': str(run), 'artifact_manifest_sha256': sha(run / 'artifact_manifest.json'),
            'files_sha256': hashes, 'physics_family_sha256': family,
            'provenance_reference': first['run_provenance'],
            'full_provenance': full}


def verified_projector(parameters, full_provenance):
    import joint_safety
    if sha(joint_safety.__file__) != parameters['source_sha256']['joint_safety.py']:
        raise ValueError('Original project_target source hash mismatch')
    for name in ('joint_safety.py', 'environment.py'):
        if parameters['source_sha256'][name] != full_provenance['source_sha256'][name]:
            raise ValueError('Projection and collection source disagree: ' + name)
        if parameters['source_sha256'][name] != full_provenance['physics_identity']['source_sha256'][name]:
            raise ValueError('Projection and physics-family source disagree: ' + name)
    if parameters['dtype'] != 'torch.float32' or parameters['control_dt'] <= 0:
        raise ValueError('Unexpected original projection dtype/control interval')
    for key, shape in (('limits', (1, 56, 2)), ('scales', (1, 56)),
                       ('max_target_delta', (1, 56)), ('step_limit', (1, 56)),
                       ('command_velocity_limits', (1, 56))):
        value = np.asarray(parameters[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid recorded projection parameter: ' + key)
        if key != 'limits' and np.any(value <= 0):
            raise ValueError('Nonpositive original rate/scale: ' + key)
    if not bits_equal(parameters['step_limit'], np.minimum(parameters['scales'], parameters['max_target_delta'])):
        raise ValueError('Recorded step limit disagrees with original minimum')
    return joint_safety.project_target


def load_run(directory, baseline_saved, baseline_sha, expected_family=None):
    evidence = verify_evidence(directory)
    run = Path(evidence['directory'])
    family, full = evidence['physics_family_sha256'], evidence['full_provenance']
    if expected_family is not None and family != expected_family:
        raise ValueError('Different exact physics family: do not pool or evaluate together')
    if full['student_checkpoint_sha256'] != baseline_sha:
        raise ValueError('This initial V5 dataset requires the declared frozen v4 collection actor')
    metadata = baseline_saved['metadata']
    names, left = metadata['joint_names'], metadata['output_indices']
    if len(names) != 56 or len(left) != 28 or set(left) != {i for i, name in enumerate(names) if name.endswith('_L') or name.startswith('left_')}:
        raise ValueError('Invalid frozen baseline left joint schema')
    if read_json(run / 'config.json')['joint_names'] != names:
        raise ValueError('Dataset and frozen baseline joint order differ')
    protocol = read_json(run / 'runtime_protocol.json')
    sources = metadata['sources']
    if len(sources) != 1 or protocol['source_manifest']['manifest_sha256'] != sources[0]['manifest_sha256']:
        raise ValueError('Frozen baseline and collection phase-source manifest differ')
    parameters = read_json(run / 'projection_parameters.json')
    if parameters['joint_names'] != names: raise ValueError('Projection joint order differs')
    projector = verified_projector(parameters, full)
    tensors = {key: torch.tensor(parameters[key], dtype=torch.float32) for key in
               ('limits', 'scales', 'step_limit', 'command_velocity_limits')}
    xs, previous, labels, masks, sources_executed = [], [], [], [], []
    previous_post = previous_source = episode_name = None
    ended = False
    excluded = []
    with (run / 'transitions.jsonl').open() as stream, (run / 'trajectory.jsonl').open() as trajectory_stream:
        for line_number, line in enumerate(stream):
            raw = json.loads(line, parse_constant=lambda text: (_ for _ in ()).throw(ValueError('Nonfinite transition')))
            record = provenance_reader.expand_record_provenance(raw, run)
            if canonical(record['run_provenance']) != canonical(full):
                raise ValueError('Provenance changes within a collected episode')
            if ended: raise ValueError('Record appears after terminal event')
            if record['step'] != len(xs): raise ValueError('Skipped/reordered actual-state step')
            if episode_name is None: episode_name = record['episode']
            if record['episode'] != episode_name: raise ValueError('Multiple episodes in one reset trace')
            if record.get('record_type') == 'abort':
                ended = True
                excluded.append({'step': record['step'], 'reason': 'terminal_abort_not_a_transition'})
                continue
            if record.get('supervision_schema') != 'baseline_left_projected_expert_v3':
                raise ValueError('Missing expert projected target schema; actual action is not a label')
            pre, post, expert = record['actor_observation_pre'], record['actor_observation_post'], record['expert_supervision']
            raw_line = next(trajectory_stream, None)
            if raw_line is None: raise ValueError('Missing executed trajectory row')
            raw_execution = json.loads(raw_line)
            if raw_execution['step'] != record['step'] or raw_execution['phase'] != pre['phase']:
                raise ValueError('Executed trajectory and transition step/phase disagree')
            termination = raw_execution['termination']
            timeout = expert['oracle_env_inputs']['episode_length'] + 1 >= expert['oracle_env_inputs']['max_episode_length']
            if expert['oracle_env_inputs']['episode_length'] != record['step']:
                raise ValueError('Original environment length does not match reset trace')
            raw_done = bool(termination['raw_done'])
            if (termination['done_matches_original_causes'] is not True
                    or raw_done != bool(raw_execution['success'] or raw_execution['drop'] or timeout)
                    or bool(raw_execution['terms']['terminal']) != raw_done
                    or bool(raw_execution['terms']['timeout']) != timeout
                    or bool(termination['derived_timeout_truncated']) != timeout
                    or bool(termination['derived_drop_terminated']) != bool(raw_execution['drop'])):
                raise ValueError('Original raw done/drop/timeout evidence is inconsistent')
            if raw_done and record['terminal'] is not True:
                raise ValueError('Original done was ignored by collection')
            validate_gate(record['pre_gates'], stage_required=record['phase_transition'])
            if expert['failure_reason']: raise ValueError('Oracle failure cannot provide a training label')
            if previous_post is None:
                if pre['previous_executed_source'] != 'reset': raise ValueError('Trace does not start at reset')
            else:
                if pre['previous_executed_source'] != previous_source: raise ValueError('Previous-source audit mismatch')
                for key in STATE_FIELDS:
                    if canonical(pre[key]) != canonical(previous_post[key]):
                        raise ValueError('Actual-state history discontinuity: ' + key)
            schedule = expert['public_same_action_schedule']
            if pre['phase'] != schedule['phase'] or pre['phase_progress'] != schedule['phase_progress']:
                raise ValueError('Stale phase/progress relative to the action label')
            if not bits_equal(expert['projection_previous_applied_target'], pre['applied_target']):
                raise ValueError('Expert projection uses a different previous actual command')
            if expert['projection_parameters_file'] != 'projection_parameters.json' or expert['projection_parameters_file_sha256'] != evidence['files_sha256']['projection_parameters.json']:
                raise ValueError('Projection parameter reference hash/path mismatch')
            if expert['projection_source_sha256'] != parameters['source_sha256']:
                raise ValueError('Expert label projection source mismatch')
            old = torch.as_tensor(pre['applied_target'], dtype=torch.float32).reshape(1, 56)
            action = torch.as_tensor(expert['normalized_action'], dtype=torch.float32).reshape(1, 56)
            requested = old + tensors['scales'] * action
            projected, hit = projector(old, requested, tensors['limits'], tensors['step_limit'],
                                       tensors['command_velocity_limits'], parameters['control_dt'])
            if not bits_equal(expert['requested_target'], requested[0]) or not bits_equal(expert['projected_target'], projected[0]):
                raise ValueError('Expert projected label does not reproduce original float32 projection')
            if expert['projection_soft_limit_hit'] != hit[0].tolist():
                raise ValueError('Expert projection limit audit differs')
            if not bits_equal(record['actual_applied_target'], post['applied_target']):
                raise ValueError('Actual executed action disagrees with post-state')
            source = record['executed_source']
            if raw_execution['executed_source'] != source:
                raise ValueError('Executed source differs between independent journals')
            native = torch.as_tensor(raw_execution['native_action'], dtype=torch.float32).reshape(1,56)
            right = [i for i in range(56) if i not in left]
            if not bits_equal(native[0,right], action[0,right]):
                raise ValueError('Original right-hand native action was changed')
            actual_projected, _ = projector(old, old+tensors['scales']*native, tensors['limits'],
                tensors['step_limit'], tensors['command_velocity_limits'], parameters['control_dt'])
            if (not bits_equal(actual_projected[0], record['actual_applied_target'])
                    or not bits_equal(raw_execution['applied_target'], record['actual_applied_target'])):
                raise ValueError('Actual native action does not reproduce executed command')
            if source == 'teacher':
                if (not bits_equal(native, action) or record['expert_projection_matches_teacher_actual_bitwise'] is not True or not bits_equal(expert['projected_target'], record['actual_applied_target'])):
                    raise ValueError('Teacher projected label was not verified against actual command')
            elif source == 'student':
                if record['expert_projection_matches_teacher_actual_bitwise'] is not None:
                    raise ValueError('Student actual must not certify the expert label')
            else: raise ValueError('Unknown actual execution source')
            safe_label = record['failure_reason'] is None and not raw_done
            post_gate_failed = False
            try: validate_gate(record['post_gates'], stage_required=record['stage_complete'])
            except ValueError: safe_label = False; post_gate_failed = True
            if bool(record['terminal']) != bool(record['failure_reason'] or post_gate_failed or record['stage_complete']):
                raise ValueError('Collection terminal flag disagrees with original post-gates')
            if not safe_label: excluded.append({'step': len(xs), 'reason': record['failure_reason'] or ('original_environment_done' if raw_done else 'post_gate_failure')})
            xs.append(features(pre)); previous.append(np.asarray(pre['applied_target'], dtype=np.float32)[left])
            # The only target assignment. Never use record.actual_applied_target.
            labels.append(np.asarray(expert['projected_target'], dtype=np.float32)[left])
            masks.append(safe_label); sources_executed.append(source)
            previous_post, previous_source = post, source
            ended = bool(record['terminal'])
        if next(trajectory_stream, None) is not None:
            raise ValueError('Unfinished/unrecorded physics transition in trajectory journal')
    result = read_json(run / 'result.json')
    if not ended or not xs or result['steps'] != len(xs):
        raise ValueError('Run is incomplete or transition count disagrees with final result')
    if not any(masks): raise ValueError('No eligible development supervision in this run')
    return {'x': np.stack(xs), 'previous_left': np.stack(previous), 'labels': np.stack(labels),
            'mask': np.asarray(masks, dtype=bool), 'left_indices': left, 'joint_names': names,
            'parameters': parameters, 'projector': projector, 'physics_family_sha256': family,
            'evidence': {k:v for k,v in evidence.items() if k != 'full_provenance'},
            'excluded_labels': excluded, 'executed_sources': sources_executed,
            'collection_source_sha256': full['source_sha256'],
            'episode_id': str(run) + ':' + str(episode_name), 'result': result,
            'scope': 'development_actual_state_expert_supervision_not_episode_success_or_generalization'}


def verify_saved_datasets(metadata, directories):
    expected = metadata['dataset_evidence']
    if len(directories) != len(expected): raise ValueError('All source datasets required to validate checkpoint provenance')
    for directory, old in zip(directories, expected):
        current = verify_evidence(directory)
        for key in ('artifact_manifest_sha256', 'files_sha256', 'physics_family_sha256'):
            if current[key] != old[key]: raise ValueError('Checkpoint dataset provenance changed: ' + key)
        if current['physics_family_sha256'] != metadata['physics_family_sha256']:
            raise ValueError('Checkpoint physics family mismatch')
