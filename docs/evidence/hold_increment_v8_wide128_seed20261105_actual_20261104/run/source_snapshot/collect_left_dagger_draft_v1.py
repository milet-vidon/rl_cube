"""Reviewable collection entry draft; requires a separately reviewed runtime bridge.

This file does not choose a physics environment or construct an Isaac app.
`--print-contract` is host-only. Do not run a factory while another Isaac is live.
"""
from pathlib import Path
import argparse
import importlib
import json
import random

from dagger_collection_core_v1 import FrozenTeacherOracle, DaggerTransitionCollector, teacher_snapshot, canonical

RUNTIME_CONTRACT = {
    'status': 'draft_only_runtime_bridge_and_live_sampling_not_validated',
    'scope': 'privileged_simulator_state_policy_pickup_through_tilt_hold_only',
    'fields': ['teacher', 'joint_names', 'left_indices', 'physics_identity', 'source_sha256', 'decoder_id'],
    'methods': {
        'capture()': 'actual state plus oracle_env_inputs/control_settings; no reference row substitution',
        'decode_target(native_action)': 'exact environment dtype/arithmetic; supervision label only',
        'schedule_resolver(observation, teacher_before, teacher_after)': 'phase/progress for the action just queried, not next phase; no left target lookup',
        'validate_control_change(before, after, teacher_before, teacher_after)': 'only original legal gain scheduling, explicitly audited',
        'pre_gates(observation)': '(original gate report, phase_transition_requires_gate), never loosen thresholds',
        'execute_packet(packet)': 'teacher: original native action; learner: replace only left slice via original env conversion; right slice original',
        'post_gates(observation)': '(original safety report, original wrist gate accepted, failure reason)',
        'archive_changed_plans(teacher_snapshot)': 'archive full plans matching hashes without mutating controller',
        'close()': 'close owned runtime only; no second simulator creation by this core'
    },
    'original_final_gate': 'assembled cube plus original assess_wrist >=3s; no center oracle in v1',
    'warmup': 'preserve original teacher until first arrival-memory initialization; teacher intervention recorded from reset',
    'physics': 'root must select and fingerprint one environment; ordinary/D6-damped/native-articulation datasets cannot be pooled',
}


def arrival_initialized(teacher):
    snapshot = teacher_snapshot(teacher)
    def visit(node):
        if isinstance(node.get('targets'), dict) and any(k.startswith('_applied_') for k in node['targets']):
            return True
        return any(visit(value) for value in node.values() if isinstance(value, dict))
    return visit(snapshot)


def collect(runtime, actor, journal, checkpoint_sha256, *, max_steps=2000, teacher_probability=.8, seed=20261104):
    """Dependency-injected draft, suitable for host review; no hidden launch."""
    if not 0 <= teacher_probability <= 1:
        raise ValueError('Invalid teacher mixture probability')
    oracle = FrozenTeacherOracle(runtime.teacher, runtime.joint_names, runtime.capture,
        runtime.decode_target, runtime.decoder_id, runtime.schedule_resolver,
        memory_mode='preserve', control_change_validator=runtime.validate_control_change)
    provenance = {'student_checkpoint_sha256': checkpoint_sha256, 'source_sha256': runtime.source_sha256,
                  'physics_identity': runtime.physics_identity, 'collection_scope': RUNTIME_CONTRACT['scope'],
                  'capture_begins_at_episode_reset': True, 'selection_seed': seed,
                  'teacher_probability_after_warmup': teacher_probability,
                  'runtime_bridge_validated': False}
    collector = DaggerTransitionCollector(oracle, journal, 'left_dagger', provenance, runtime.left_indices)
    rng = random.Random(seed); previous_source = 'reset'; last = None
    for step in range(max_steps):
        observation = runtime.capture(); observation['previous_executed_source'] = previous_source
        initialized = arrival_initialized(runtime.teacher)
        oracle.memory_mode = 'sync_actual_applied' if initialized and teacher_probability < 1. else 'preserve'
        draw = rng.random()
        source = 'teacher' if not initialized or draw < teacher_probability else 'student'
        gates, phase_transition = runtime.pre_gates(observation)
        decision = collector.prepare(observation, actor, gates, execute_source=source,
            selection_reason='original_teacher_warmup_through_first_arrival' if not initialized else 'seeded_explicit_dagger_mixture',
            selection_probability=1. if not initialized else teacher_probability, selection_draw=draw,
            phase_transition=phase_transition)
        runtime.archive_changed_plans(decision['expert_supervision']['teacher_state_after_query'])
        try:
            runtime.execute_packet(collector.execution_packet())
            post = runtime.capture(); post['previous_executed_source'] = source
            post_gates, wrist_qualified, failure = runtime.post_gates(post)
            last = collector.complete(post, post_gates, stage_complete=wrist_qualified, failure_reason=failure)
        except Exception as exc:
            if not collector.terminal:
                collector.abort(str(exc), runtime.capture())
            raise
        previous_source = source
        if last['terminal']:
            break
    if not collector.terminal:
        collector.abort('collection_step_budget_before_original_wrist_gate', runtime.capture())
    return {'scope': RUNTIME_CONTRACT['scope'], 'steps': collector.next_step,
            'teacher_interventions': collector.interventions,
            'original_wrist_gate_qualified': bool(last and last['stage_complete'] and not last['failure_reason']),
            'pure_student_episode_claim': False, 'center_oracle_implemented': False,
            'runtime_bridge_validated': False, 'physics_identity_sha256': collector.provenance['physics_identity_sha256']}


def v4_actor(checkpoint, left_indices):
    import numpy as np
    import torch
    from phase_basis_policy_v4 import restored, features, sha
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['metadata']['output_indices'] != list(left_indices):
        raise ValueError('Behavior checkpoint left joint ordering differs from runtime')
    if saved['metadata']['source_sha256']['phase_basis_policy_v4.py'] != sha(Path(__file__).parent/'phase_basis_policy_v4.py'):
        raise ValueError('V4 source hash changed')
    model = restored(saved).eval()
    def actor(observation):
        x = torch.from_numpy(features(observation, observation['phase'], observation['phase_progress'])).unsqueeze(0)
        previous = torch.as_tensor(np.asarray(observation['applied_target'],dtype=np.float32)[left_indices]).unsqueeze(0)
        with torch.no_grad(): return model.targets(x, previous)[0].numpy().astype(np.float32).tolist()
    return actor, sha(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-contract', action='store_true')
    parser.add_argument('--runtime-factory', help='explicit reviewed module:function; no factory is provided by this draft')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--teacher-probability', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps(RUNTIME_CONTRACT,indent=2)); return
    if not args.runtime_factory or not args.checkpoint or not args.output:
        parser.error('A reviewed runtime factory, checkpoint and unique output are required; no automatic environment selection')
    args.output.mkdir(parents=True,exist_ok=False)
    module, name = args.runtime_factory.split(':',1)
    runtime = getattr(importlib.import_module(module),name)(args)
    try:
        actor, checkpoint_hash = v4_actor(args.checkpoint, runtime.left_indices)
        with (args.output/'transitions.jsonl').open('w') as journal:
            result = collect(runtime,actor,journal,checkpoint_hash,max_steps=args.max_steps,
                teacher_probability=args.teacher_probability,seed=args.seed)
        (args.output/'collection_result.json').write_text(json.dumps(result,indent=2))
    finally:
        runtime.close()


if __name__ == '__main__': main()
