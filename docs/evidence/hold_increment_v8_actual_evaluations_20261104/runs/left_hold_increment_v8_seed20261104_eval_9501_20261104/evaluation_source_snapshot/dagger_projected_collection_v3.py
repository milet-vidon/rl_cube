"""Independent v3 collection wrapper: original projection labels, frozen oracle.

No simulator construction or action reinterpretation is hidden in this module.
The environment supplies its original project_target using actual pre-state.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from dagger_collection_core_v1 import FrozenTeacherOracle, DaggerTransitionCollector, canonical, physics_fingerprint
from collect_left_dagger_draft_v1 import arrival_initialized, RUNTIME_CONTRACT

PROVENANCE_SCHEMA='left_dagger_immutable_provenance_ref_v3'
PROVENANCE_FILE='collection_provenance.json'


def persist_provenance(directory,full):
    """Write once after the frozen collector validates the full identity."""
    if physics_fingerprint(full['physics_identity'])!=full['physics_identity_sha256']:
        raise ValueError('Full physics identity hash does not match')
    data=(canonical(full)+'\n').encode('utf-8')
    with (Path(directory)/PROVENANCE_FILE).open('xb') as stream:stream.write(data)
    return {'schema':PROVENANCE_SCHEMA,'file':PROVENANCE_FILE,
            'sha256':hashlib.sha256(data).hexdigest(),
            'physics_identity_sha256':full['physics_identity_sha256']}


def expand_record_provenance(record,directory):
    """Consumer boundary: validate external bytes and physics hash, then expand.

    Both completed transitions and terminal aborts are supported. Nested abort
    pending_transition provenance is expanded too. No unchecked reference is
    considered a training record, and path traversal is never accepted.
    """
    root=Path(directory).resolve();cache={}
    def resolve(ref):
        if set(ref)!={'schema','file','sha256','physics_identity_sha256'} or ref['schema']!=PROVENANCE_SCHEMA:
            raise ValueError('Expected explicit v3 provenance reference')
        if ref['file']!=PROVENANCE_FILE:raise ValueError('Unexpected provenance path')
        path=(root/ref['file']).resolve()
        if path.parent!=root:raise ValueError('Provenance file is outside dataset directory')
        key=(ref['sha256'],ref['physics_identity_sha256'])
        if key not in cache:
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('Provenance file SHA256 mismatch')
            full=json.loads(data)
            digest=physics_fingerprint(full['physics_identity'])
            if digest!=ref['physics_identity_sha256'] or digest!=full['physics_identity_sha256']:
                raise ValueError('Physics identity SHA256 mismatch')
            if not full.get('source_sha256') or not full.get('student_checkpoint_sha256'):
                raise ValueError('Missing immutable source/checkpoint provenance')
            cache[key]=full
        return deepcopy(cache[key])
    def visit(value):
        if isinstance(value,dict):
            return {k:resolve(v) if k=='run_provenance' else visit(v) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return value
    if 'run_provenance' not in record:raise ValueError('Record has no provenance reference')
    return visit(record)


class ProjectedDaggerCollector(DaggerTransitionCollector):
    def __init__(self,*args,runtime,**kwargs):
        super().__init__(*args,**kwargs)
        self.runtime=runtime
        self.provenance=persist_provenance(runtime.output,self.provenance)

    def prepare(self,observation,*args,**kwargs):
        super().prepare(observation,*args,**kwargs)
        try:
            projection=self.runtime.project_expert_target(self.oracle.last_native_action)
            if canonical(projection['requested_target'])!=canonical(self.pending['expert_supervision']['requested_target']):
                raise RuntimeError('Projection requested target differs from native oracle decoder')
            self.pending['expert_supervision'].update(projection)
            self.pending['supervision_schema']='baseline_left_projected_expert_v3'
            return deepcopy(self.pending)
        except Exception as exc:
            self.abort(str(exc),observation)
            raise

    def complete(self,observation_post,*args,**kwargs):
        if self.pending is None:raise RuntimeError('No pending projected transition')
        verified=None
        if self.pending['executed_source']=='teacher':
            verified=self.runtime.verify_teacher_projected_target(self.pending['expert_supervision']['projected_target'])
            if not verified:
                self.abort('Expert projected label differs bitwise from original teacher actual command',observation_post)
                raise RuntimeError('Teacher projected label validation failed')
        self.pending['expert_projection_matches_teacher_actual_bitwise']=verified
        return super().complete(observation_post,*args,**kwargs)


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
    collector = ProjectedDaggerCollector(oracle, journal, 'left_dagger_v3', provenance, runtime.left_indices, runtime=runtime)
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

