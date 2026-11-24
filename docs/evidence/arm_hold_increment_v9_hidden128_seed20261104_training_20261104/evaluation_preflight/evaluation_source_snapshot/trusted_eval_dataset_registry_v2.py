"""Explicit V8 trust over frozen trajectory predicates; no training grant.

The private reader extends policy admission and separates successful-wrist
outcomes from the V4-only actor route. V8 always uses its actual 284-feature
recurrent actor. Successful V8 data must reproduce the unchanged full original
wrist/grasp auditors; failures retain the original safe-prefix exclusions.
"""
from pathlib import Path
from types import MappingProxyType,ModuleType
import hashlib
import trusted_eval_dataset_registry_v1 as prior
import train_hold_increment_v8 as trainer
import hold_increment_policy_v8 as policy

SCHEMA='trusted_completed_eval_bc_candidate_registry_v2'
V1_SHA='e19a5fd879eb9d7e814d14ec1dff906f02d773868b1ff91a1bdd9ce759ff617c'
ENGINE_SHA=prior.ENGINE_SHA
V8_RUNTIME_SHA='8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c'
V8_ENTRY_SHA='54a9f1ae958cacd55434397e72b04c10743c8a16c1be24d74f3518010ed497c3'
V8_TRAINER_SHA='055f8f0d6ed724c2932a2effa59d402c2e22472491519f9a1b09fb553e5a42cb'
V8_POLICY_SHA='b708e69d928bf073f8f7ee5f50edd9e1fe0f96213bfe3a49f51cfeff4c0561ae'
V8_QUALIFICATION_SHA='810b793afb4b827bfcea72778c66f55883552f143b0fda4bd2245eb5aaa2d1b9'
V8_TRUSTED=MappingProxyType({
 '6280460b32b1d87a7bf2512820ad1b23a83437261bf5881a6c64b441de3d4d5b':'55b3bf394fe39e98eb7f277aa96805879186129ead473ad210d9682af9221205',
 '56eba2c26ceff4d85513da3970bfa11ec83d87448f47e9c7b3abafc878f74456':'d6d5961f23215140a21658c51c6105e0ad24eb7db1a2df0c65f688708335e6ef'})
REGISTRY=MappingProxyType(dict(prior.REGISTRY)|{'v8':'reviewed_v8_actual_outcome_original_wrist_auditors'})
read_json=prior.read_json;sha=prior.sha;require=prior.require


def _sources():
    require(sha(prior.__file__)==V1_SHA,'Frozen registry v1 changed')
    require(sha(prior.frozen.__file__)==ENGINE_SHA,'Frozen predicate engine changed')
    require(sha(trainer.__file__)==V8_TRAINER_SHA,'Frozen V8 checkpoint loader changed')
    require(sha(policy.__file__)==V8_POLICY_SHA,'Frozen V8 model changed')


def _v8_paths_and_trust(run,identity):
    _sources();run=Path(run)
    require(identity['policy']=='v8','Only V8 uses this explicit trust route')
    checkpoint=identity['evaluated_policy_checkpoint_sha256']
    require(checkpoint in V8_TRUSTED,'V8 checkpoint is outside the explicitly reviewed registry')
    require(identity['qualification_sha256']==V8_QUALIFICATION_SHA,'Unreviewed V8 qualification')
    require(identity['training_authorization_sha256']==V8_TRUSTED[checkpoint],'Wrong reviewed V8 authorization')
    qualification=run/'reviewed_transition_qualification.json';authorization=run/'reviewed_training_authorization.json'
    require(sha(qualification)==V8_QUALIFICATION_SHA and sha(authorization)==V8_TRUSTED[checkpoint],'Reviewed external V8 evidence changed')
    q=read_json(qualification);directories=[Path(d['evidence']['directory'])for d in q['datasets']]
    require(tuple(p.name for p in directories)==trainer.RUN_NAMES,'V8 training histories/order changed')
    return checkpoint,qualification,authorization,q,directories


def restore_recorded_model(run,identity,saved):
    if identity['policy']!='v8':return prior.restore_recorded_model(run,identity,saved)
    run=Path(run);checkpoint,qpath,authpath,q,directories=_v8_paths_and_trust(run,identity)
    model,metadata=trainer.load_checkpoint(run/'checkpoints/evaluated_policy.pt',run/'checkpoints/frozen_v4_baseline.pt',directories,
        expected_checkpoint_sha256=checkpoint,expected_physics_family=identity['actual_physics_family_sha256'],
        qualification_path=qpath,expected_qualification_sha256=V8_QUALIFICATION_SHA,
        authorization_path=authpath,expected_authorization_sha256=V8_TRUSTED[checkpoint])
    require(metadata==saved['metadata'],'Supplied V8 metadata differs from trusted checkpoint bytes')
    require(set(saved['model'])==set(model.state_dict()),'Supplied V8 tensor keys differ')
    for name,value in model.state_dict().items():
        require(trainer.bits_equal(value,saved['model'][name]),'Supplied V8 tensor differs: '+name)
    return model


def normalize_v8_runtime(raw):
    require(hashlib.sha256(raw).hexdigest()==V8_RUNTIME_SHA,'Unreviewed V8 runtime bytes')
    source=raw.decode('utf-8').replace('from hold_increment_policy_v8 import EpisodeActor, features as v8_features',
        'from phase_residual_policy_v5 import EpisodeActor, features as v5_features')
    source=source.replace("== 'v8'","== 'v5'").replace("=='v8'","=='v5'").replace('v8_features(', 'v5_features(')
    require(hashlib.sha256(source.encode()).hexdigest()==prior.LEGACY_RUNTIME_SHA,'V8 original control/gate/terminal semantics changed')
    return source


def verify_evaluation_semantics(run,identity,protocol):
    if identity['policy']!='v8':return prior.verify_evaluation_semantics(run,identity,protocol)
    run=Path(run);checkpoint,qpath,authpath,q,directories=_v8_paths_and_trust(run,identity)
    folder=run/'evaluation_source_snapshot';runtime_path=folder/'state_hold_increment_eval_runtime_v8.py'
    require(sha(folder/'evaluate_state_hold_increment_v8.py')==V8_ENTRY_SHA,'Unreviewed V8 entry')
    normalize_v8_runtime(runtime_path.read_bytes())
    saved=prior.frozen.torch.load(run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    restore_recorded_model(run,identity,saved);metadata=saved['metadata']
    for key,metadata_key,expected in (
        ('public_residual_gate_contract','gate_contract',policy.PUBLIC_GATE_CONTRACT),
        ('action_contract','action_contract',policy.ACTION_CONTRACT),
        ('training_classification_contract','classification_contract',policy.CLASSIFICATION_CONTRACT)):
        require(identity[key]==metadata[metadata_key]==expected,'V8 public/action/classification contract differs: '+key)
    names=[metadata['joint_names'][i]for i in metadata['output_indices']]
    require(identity['update_bound_joint_names']==metadata['left_joint_names']==names,'V8 actual ordered joint names differ')
    require(identity['update_bound_rad']==metadata['update_bound_rad']==q['update_bound_rad'],'V8 increment bounds differ')
    require(identity['update_anchor_is_previous']==metadata['update_anchor_is_previous']==q['update_anchor_is_previous'],'V8 named anchor mask differs')
    require(metadata['feature_schema']==identity['actor_feature_schema']==policy.FEATURE_SCHEMA and identity['actor_feature_count']==284,'V8 features differ')
    require(metadata['physics_family_sha256']==identity['actual_physics_family_sha256']==trainer.FAMILY_SHA and
        metadata['frozen_baseline_checkpoint_sha256']==identity['frozen_v4_baseline_checkpoint_sha256']==trainer.BASELINE_SHA,'V8 physical domain/baseline differs')
    require(identity['expert_hidden_or_labels_in_actor']is False and identity['executor_identity_in_actor_features']is False,'V8 actor isolation changed')
    dependencies=metadata['source_sha256'];resolutions={}
    require(dependencies==q['source_sha256'] and len(dependencies)==18,'V8 complete frozen dependency graph differs')
    require(identity['all_checkpoint_source_dependencies_snapshotted_exactly']is True and identity['checkpoint_source_dependency_count']==18,'V8 direct source snapshot contract absent')
    for name,digest in dependencies.items():
        path=prior.frozen.inside(run,'evaluation_source_snapshot/'+name)
        require(identity['source_sha256'].get(name)==digest and path.exists()and sha(path)==digest,'V8 direct dependency snapshot missing or changed: '+name)
        resolutions[name]={'scope':'actual_evaluation_direct_snapshot','path':str(path),'sha256':digest}
    return {'policy_version':'v8','runtime_sha256':V8_RUNTIME_SHA,'entry_sha256':V8_ENTRY_SHA,
        'runtime_after_only_policy_routing_rename_sha256':prior.LEGACY_RUNTIME_SHA,
        'same_original_physics_step_projection_right_action_query_order_gates_termination_and_audit':True,
        'reviewed_checkpoint_sha256':checkpoint,'reviewed_qualification_sha256':V8_QUALIFICATION_SHA,
        'reviewed_training_authorization_sha256':V8_TRUSTED[checkpoint],
        'checkpoint_frozen_baseline_and_control_buffers_verified':True,'source_snapshot_resolutions':resolutions,
        'admitted_outcome_scope':'original_failed_safe_prefix_or_recomputed_original_wrist_success; original False flags unchanged',
        'success_is_not_whole_robot_neural_or_RL_or_generalization_qualification':True}


def admitted_successful_wrist(identity,result):
    # Outcome routing is independent of actor architecture. All actual success
    # predicates remain in the original engine and its unchanged final auditor.
    return identity['policy']=='v4' or (identity['policy']=='v8' and result['original_wrist_gate_qualified']is True)


SOURCE_HOOKS=(
 ("identity['policy']in ('v4','v5','v6')","identity['policy']in REGISTRY"),
 ("successful_v4=identity['policy']=='v4'","successful_v4=identity['policy']=='v4'\n    successful_wrist=admitted_successful_wrist(identity,result)"),
 ("(result['failure']is None and result['original_wrist_gate_qualified']is True)if successful_v4 else",
  "(result['failure']is None and result['original_wrist_gate_qualified']is True)if successful_wrist else"),
 ("require(successful_v4,'Failed-prefix consumer cannot accept a successful wrist boundary')",
  "require(successful_wrist,'Failed-prefix consumer cannot accept a successful wrist boundary')"),
 ("final['original_wrist_gate_qualified']==successful_v4 and (success_audit is not None)==successful_v4",
  "final['original_wrist_gate_qualified']==successful_wrist and (success_audit is not None)==successful_wrist"),
)


def engine_source():
    _sources();raw=Path(prior.frozen.__file__).read_bytes();source=raw.decode('utf-8')
    changed=source
    for before,after in SOURCE_HOOKS:
        require(changed.count(before)==1,'Unique explicit admission/outcome hook changed')
        changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(SOURCE_HOOKS):
        require(restored.count(after)==1,'Private engine inverse hook is ambiguous')
        restored=restored.replace(after,before)
    require(restored.encode()==raw,'Private engine inverse does not restore exact frozen bytes')
    return changed


def _engine():
    module=ModuleType('_trusted_eval_registry_v2_private_engine')
    module.__dict__.update(__file__=prior.frozen.__file__,REGISTRY=REGISTRY,admitted_successful_wrist=admitted_successful_wrist)
    exec(compile(engine_source(),str(Path(__file__).resolve())+'::frozen_engine','exec'),module.__dict__)
    module.SCHEMA=SCHEMA;module.verify_evaluation_semantics=verify_evaluation_semantics;module.restore_recorded_model=restore_recorded_model
    return module


def verify_evidence(directory):return _engine().verify_evidence(directory)


def load_run(directory,baseline_saved,baseline_sha,expected_family=None):
    data=_engine().load_run(directory,baseline_saved,baseline_sha,expected_family)
    data['registry_reader_sha256']=sha(__file__);data['frozen_predicate_engine_sha256']=ENGINE_SHA
    data['frozen_prior_registry_sha256']=V1_SHA;data['training_qualification_granted']=False
    return data


def verify_saved_datasets(metadata,directories):return _engine().verify_saved_datasets(metadata,directories)
