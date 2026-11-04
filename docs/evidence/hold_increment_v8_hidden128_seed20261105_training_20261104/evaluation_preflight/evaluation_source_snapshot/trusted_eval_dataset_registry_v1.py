"""Explicit trusted-model dispatch over the frozen, fully audited reader loop.

Only one policy-admission expression is extended in a private module instance.
The original module and source bytes are untouched. All trajectory predicates,
projection, memory synchronization, actor replay and terminal checks are reused.
V7 admission currently covers reviewed failed prefixes only, not neural success.
"""
from pathlib import Path
from types import MappingProxyType,ModuleType
import hashlib
import eval_prefix_dataset_v7 as frozen
import train_zero_gated_residual_v7 as trainer
import zero_gated_residual_policy_v7 as policy

SCHEMA='trusted_completed_eval_bc_candidate_registry_v1'
ENGINE_SHA='61c6b43cf0b2af18144723f6f3b2587dea04299feb75611e1a09c7c073b67883'
LEGACY_RUNTIME_SHA='90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64'
V7_RUNTIME_SHA='caadc319e816bdb3c00b5a744b71fbaf7c5f60ffd2d8f1893593fe441767b66b'
V7_ENTRY_SHA='6cda789c883a171d565b3a47d5ca5901ad9b4945221b281024eae7f66405b7f0'
V7_QUALIFICATION_SHA='9dc17535d63a0a7667b1b3aca1853421e6d0a5b8dbf44560462611130582474f'
V7_TRUSTED=MappingProxyType({
 '0920f7e4952b5bf534021ce8240a8467a303bffda4b7ec1750eb830b3aea4054':'bee59f0f281d62029b8fbb9681b0a1a6affeb058f0b441d3e760951d300fd797',
 '10afd5a60ec587eb040068a520dad171ce412d681d2ff46251ae74b8ea84e7bb':'9e0e019bc579812dd1265c598e7be4cc63f06f202d558f8ab2298a112b06e324',
 'a7efd54313d40c8991e8c468c4312442501ba71b9a4d427170318bcfd29a99ee':'9cdd48132d73447ad6c85ec6bebdf0b399b381e69e0d1a447d3688d3178ed9c9'})
REGISTRY=MappingProxyType({'v4':'legacy_wrist_success','v5':'legacy_failed_prefix','v6':'legacy_failed_prefix','v7':'trusted_v7_failed_prefix'})
read_json=frozen.read_json
sha=frozen.sha
require=frozen.require


def _v7_paths_and_trust(run,identity):
    checkpoint=identity['evaluated_policy_checkpoint_sha256']
    require(checkpoint in V7_TRUSTED,'V7 checkpoint is outside the explicitly reviewed registry')
    require(identity['qualification_sha256']==V7_QUALIFICATION_SHA,'Unreviewed V7 qualification')
    require(identity['training_authorization_sha256']==V7_TRUSTED[checkpoint],'Wrong reviewed training authorization')
    require(sha(trainer.__file__)=='9878b9d2c6f769624ba3889ea43797eacd02b203f65d8dcb3d765d90b1f9636e','Frozen V7 checkpoint loader changed')
    require(sha(policy.__file__)=='714c7d6b6ca151afa052ef426cc27360b61ca62c1914cd7b7bfe84a63233256e','Frozen V7 model changed')
    qualification=run/'reviewed_transition_qualification.json';authorization=run/'reviewed_training_authorization.json'
    require(sha(qualification)==V7_QUALIFICATION_SHA and sha(authorization)==V7_TRUSTED[checkpoint],'Reviewed external evidence changed')
    q=read_json(qualification)
    directories=[Path(d['evidence']['directory'])for d in q['datasets']]
    return checkpoint,qualification,authorization,q,directories


def restore_recorded_model(run,identity,saved):
    if identity['policy']!='v7':return frozen.restore_recorded_model(run,identity,saved)
    checkpoint,qpath,authpath,q,directories=_v7_paths_and_trust(run,identity)
    model,metadata=trainer.load_checkpoint(run/'checkpoints/evaluated_policy.pt',run/'checkpoints/frozen_v4_baseline.pt',directories,
        expected_checkpoint_sha256=checkpoint,expected_physics_family=identity['actual_physics_family_sha256'],
        qualification_path=qpath,expected_qualification_sha256=V7_QUALIFICATION_SHA,
        authorization_path=authpath,expected_authorization_sha256=V7_TRUSTED[checkpoint])
    # The old engine's caller supplies the bytes just read. Do not silently load
    # a different object if a caller has altered the in-memory saved payload.
    require(metadata==saved['metadata'],'Supplied behavior metadata differs from trusted checkpoint bytes')
    require(set(saved['model'])==set(model.state_dict()),'Supplied behavior tensor keys differ')
    for name,value in model.state_dict().items():
        require(trainer.bits_equal(value,saved['model'][name]),'Supplied behavior tensor differs: '+name)
    return model


def verify_evaluation_semantics(run,identity,protocol):
    if identity['policy']!='v7':return frozen.verify_evaluation_semantics(run,identity,protocol)
    checkpoint,qpath,authpath,q,directories=_v7_paths_and_trust(run,identity)
    folder=run/'evaluation_source_snapshot'
    runtime_path=folder/'state_zero_gated_eval_runtime_v7.py';entry=folder/'evaluate_state_zero_gated_v7.py'
    require(sha(runtime_path)==V7_RUNTIME_SHA and sha(entry)==V7_ENTRY_SHA,'Unreviewed V7 runtime/entry')
    normalized=runtime_path.read_text().replace('from zero_gated_residual_policy_v7 import EpisodeActor, features as v7_features',
        'from phase_residual_policy_v5 import EpisodeActor, features as v5_features')
    normalized=normalized.replace("== 'v7'","== 'v5'").replace("=='v7'","=='v5'").replace('v7_features(', 'v5_features(')
    require(hashlib.sha256(normalized.encode()).hexdigest()==LEGACY_RUNTIME_SHA,'V7 original control/gate/terminal semantics changed')
    saved=frozen.torch.load(run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    model=restore_recorded_model(run,identity,saved);metadata=saved['metadata']
    require(identity['public_residual_gate_contract']==metadata['gate_contract']==policy.PUBLIC_GATE_CONTRACT,'Public residual gate differs')
    require(identity['learned_zero_gate_contract']==metadata['zero_gate_contract']==policy.ZERO_GATE_CONTRACT,'Learned zero gate differs')
    require(identity['training_classification_contract']==metadata['classification_contract']==policy.CLASSIFICATION_CONTRACT,'Classification contract differs')
    names=[metadata['joint_names'][i]for i in metadata['output_indices']]
    require(identity['residual_bound_joint_names']==names and identity['residual_bound_rad']==metadata['residual_bound_rad'],'Recorded ordered bounds differ')
    require(metadata['feature_schema']==identity['actor_feature_schema']==policy.FEATURE_SCHEMA,'V7 feature schema differs')
    require(metadata['physics_family_sha256']==identity['actual_physics_family_sha256'] and metadata['frozen_baseline_checkpoint_sha256']==identity['frozen_v4_baseline_checkpoint_sha256'],'V7 model domain/baseline differs')
    require(identity['expert_hidden_or_labels_in_actor']is False and identity['executor_identity_in_actor_features']is False,'Actor data isolation changed')
    resolutions={}
    for name,digest in metadata['source_sha256'].items():
        matches=[]
        for subfolder in ('evaluation_source_snapshot','source_snapshot'):
            path=run/subfolder/name
            if path.exists()and sha(path)==digest:matches.append({'scope':'actual_evaluation','path':str(path),'sha256':digest})
        if not matches:
            # Frozen v6 reader is a training-time transitive dependency. Its
            # bytes are preserved in the hash-bound V6 training history, even
            # though not directly imported by the V7 inference adapter.
            for dataset in q['datasets']:
                evidence=dataset['evidence'];root=Path(evidence['directory'])
                for relative,expected in evidence.get('source_snapshot_sha256',{}).items():
                    if Path(relative).name==name and expected==digest:
                        path=frozen.inside(root,relative)
                        if path.exists()and sha(path)==digest:
                            matches.append({'scope':'qualified_training_history_transitive_source','path':str(path),'sha256':digest,
                                            'training_artifact_manifest_sha256':evidence['artifact_manifest_sha256']})
        require(matches,'Checkpoint dependency has no explicit archived source: '+name)
        resolutions[name]=matches
    return {'policy_version':'v7','runtime_sha256':V7_RUNTIME_SHA,'entry_sha256':V7_ENTRY_SHA,
        'runtime_after_only_policy_routing_rename_sha256':LEGACY_RUNTIME_SHA,
        'same_original_physics_step_projection_right_action_query_order_gates_termination_and_audit':True,
        'reviewed_checkpoint_sha256':checkpoint,'reviewed_qualification_sha256':V7_QUALIFICATION_SHA,
        'reviewed_training_authorization_sha256':V7_TRUSTED[checkpoint],
        'checkpoint_frozen_baseline_and_control_buffers_verified':True,'source_snapshot_resolutions':resolutions,
        'admitted_outcome_scope':'completed_failed_eval_safe_prefix_only_original_false_flags_preserved'}


def engine_source():
    raw=Path(frozen.__file__).read_bytes();require(hashlib.sha256(raw).hexdigest()==ENGINE_SHA,'Frozen predicate engine source changed')
    source=raw.decode();old="identity['policy']in ('v4','v5','v6')";new="identity['policy']in REGISTRY"
    require(source.count(old)==1,'Expected one explicit admission hook')
    return source.replace(old,new)


def _engine():
    # A private instance prevents any mutation of the frozen imported module or
    # cross-call behavior. The trust hooks are explicit, not metadata rewrites.
    module=ModuleType('_trusted_eval_registry_private_engine')
    module.__dict__.update(__file__=frozen.__file__,REGISTRY=REGISTRY)
    exec(compile(engine_source(),str(Path(__file__).resolve())+'::frozen_engine','exec'),module.__dict__)
    module.SCHEMA=SCHEMA
    module.verify_evaluation_semantics=verify_evaluation_semantics
    module.restore_recorded_model=restore_recorded_model
    return module


def verify_evidence(directory):return _engine().verify_evidence(directory)


def load_run(directory,baseline_saved,baseline_sha,expected_family=None):
    data=_engine().load_run(directory,baseline_saved,baseline_sha,expected_family)
    data['registry_reader_sha256']=sha(__file__)
    data['frozen_predicate_engine_sha256']=ENGINE_SHA
    data['training_qualification_granted']=False
    return data


def verify_saved_datasets(metadata,directories):return _engine().verify_saved_datasets(metadata,directories)
