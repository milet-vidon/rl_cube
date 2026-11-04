"""Strict V7 consumer of V4 wrist-success and V5/V6 failed diagnostics.

This preserves the original success/failure outcome and returns BC candidates.
It grants no training, whole-robot neural or generalization claim. A reviewed
V7 qualification must authorize the exact candidates, sources, mask and bounds.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
import dagger_collection_core_v1 as core
import phase_residual_policy_v5 as policy
import left_dagger_baseline_runtime_v3 as runtime
from dagger_residual_dataset_v5 import bits_equal,verified_projector

SCHEMA='completed_diagnostic_eval_safe_prefix_v7'
TRUSTED_BEHAVIORS={
 'v4':{'checkpoints':('d2493b00df27b8d84ae69da3b399b3beb926b77fdd87a9b51aa6a2b1b23a36a4',),
       'qualification':'71f98935291201335bfe24f8a19b133ee475f7e0dc5060ce553c97f93fd617d5'},
 'v5':{'checkpoints':('1c663e17f6157273ee4c80b1e6ebf1c292b6c9972278041936b56943212db98c',),
       'qualification':'71f98935291201335bfe24f8a19b133ee475f7e0dc5060ce553c97f93fd617d5'},
 'v6':{'checkpoints':('523e7163d90da38ace84eb07597ccc0274a5ced14f2ae74d05ddb6dbc30eff04',
                      '0bedb191b59e302abc7a273c6c0071c0400d3687bc5d893ad0a84c566de02d48'),
       'qualification':'5cdd1214aa7fd295e9851d2ffeaee4fca28838c8b3b989bbce499cf303fac117'}}
REQUIRED_FILES=('result.json','evaluation_protocol.json','runtime_protocol.json','physics_identity.json',
 'config.json','initial_state.json','projection_parameters.json','evaluation_transitions.jsonl','trajectory.jsonl')

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda text:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+text)))
def parse_line(line):
    return json.loads(line,parse_constant=lambda text:(_ for _ in ()).throw(ValueError('Nonfinite transition: '+text)))
def inside(root,name):
    path=(root/name).resolve()
    require(path.is_relative_to(root) and path!=root,'Artifact path escapes dataset')
    return path

def verify_evidence(directory):
    run=Path(directory).resolve();manifest=read_json(run/'artifact_manifest.json')['sha256']
    require(all(name in manifest for name in REQUIRED_FILES),'Missing required artifact hash')
    files={}
    for name,expected in manifest.items():
        files[name]=sha(inside(run,name));require(files[name]==expected,'Artifact manifest mismatch: '+name)
    identity=read_json(run/'evaluation_protocol.json');protocol=read_json(run/'runtime_protocol.json')
    snapshots={}
    for folder,sources in [('evaluation_source_snapshot',identity['source_sha256']),('source_snapshot',protocol['source_sha256'])]:
        require(bool(sources),'Missing source snapshot provenance')
        for name,expected in sources.items():
            path=inside(run,folder+'/'+name);digest=sha(path)
            require(digest==expected,'Source snapshot mismatch: '+folder+'/'+name);snapshots[folder+'/'+name]=digest
    for module in (core,policy,runtime):
        name=Path(module.__file__).name
        expected=identity['source_sha256'].get(name,protocol['source_sha256'].get(name))
        require(sha(module.__file__)==expected,'Frozen consumer source mismatch: '+name)
    physics=read_json(run/'physics_identity.json');family=core.physics_fingerprint(physics)
    require(family==identity['actual_physics_family_sha256']==identity['expected_physics_family_sha256'],'Exact physics family mismatch')
    require(physics['config_sha256']==files['config.json'],'Actual runtime configuration hash mismatch')
    topology=physics['cube_constraint_topology']
    require(topology.get('d6_damping_augmentation') is False and topology.get('native_articulation_facade') is False,'Cannot pool native/D6 domain')
    require(physics['environment_class']=='articulated_cube_env.DynamicPickupEnv','Wrong baseline environment')
    for name,expected in physics['source_sha256'].items():
        require(protocol['source_sha256'].get(name)==expected,'Physics/runtime source differs: '+name)
    checkpoint_hashes={name:sha(run/'checkpoints'/name) for name in ('evaluated_policy.pt','frozen_v4_baseline.pt')}
    require(checkpoint_hashes['evaluated_policy.pt']==identity['evaluated_policy_checkpoint_sha256'],'Evaluated policy bytes changed')
    require(checkpoint_hashes['frozen_v4_baseline.pt']==identity['frozen_v4_baseline_checkpoint_sha256'],'Frozen baseline bytes changed')
    require(identity['policy']in ('v4','v5','v6') and identity['split']=='development_seen_seed','Only reviewed V4/V5/V6 development diagnostic histories are supported')
    semantics=verify_evaluation_semantics(run,identity,protocol)
    require(identity['left_policy_begins_at_reset_step']==0 and identity['actor_hidden_reset_count']==1 and identity['left_teacher_position_action_steps']==0,'Missing from-reset/no-left-intervention contract')
    require(identity['expert_hidden_or_labels_in_actor'] is False and identity['executor_identity_in_actor_features'] is False,'Actor data isolation contract changed')
    expected_schema='frozen_v4_198_phase_basis_features'if identity['policy']=='v4'else policy.FEATURE_SCHEMA
    expected_features=198 if identity['policy']=='v4'else policy.FEATURES
    require(identity['actor_feature_schema']==expected_schema and identity['actor_feature_count']==expected_features,'Recorded actor feature layout changed')
    require(protocol['gate_threshold_changes'] is False and protocol['center_oracle_implemented'] is False and protocol['eligible_training_episode'] is False,'Frozen gate/episode scope changed')
    plans={}
    for path in sorted((run/'plans').glob('*.json')):
        digest=hashlib.sha256(core.canonical(read_json(path)).encode()).hexdigest()
        require(path.stem.rsplit('_',1)[-1]==digest,'Plan content/filename hash mismatch')
        plans[path.name]=digest
    require(bool(plans),'Missing archived original teacher plans')
    return {'consumer_schema':SCHEMA,'directory':str(run),'artifact_manifest_sha256':sha(run/'artifact_manifest.json'),
        'files_sha256':files,'source_snapshot_sha256':snapshots,'checkpoint_sha256':checkpoint_hashes,
        'plan_sha256':plans,'physics_family_sha256':family,'evaluation_semantics_proof':semantics}

def verify_evaluation_semantics(run,identity,protocol):
    version=identity['policy'];folder=run/'evaluation_source_snapshot'
    trusted=TRUSTED_BEHAVIORS[version]
    require(identity['evaluated_policy_checkpoint_sha256']in trusted['checkpoints'] and identity['qualification_sha256']==trusted['qualification'],
            'Behavior checkpoint/qualification is outside the reviewed trusted candidates')
    expected_runtime={'v5':'90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64',
                      'v6':'06d3ed5aaf99bea9e29898c1bc4d2e50b874533386aa0725103af77891b08094'}
    expected_entry={'v5':'9bff577c826107574f6d1e435a07ba4427b7aa5b05777b315a85b4a0261ec139',
                    'v6':'42cdd329dcdf9d5634c294d0336e8b79761e4588168936f1a4e6b84de234d8cb'}
    runtime_version='v5'if version=='v4'else version
    runtime_path=folder/('state_residual_eval_runtime_'+runtime_version+'.py')
    entry_path=folder/('evaluate_state_residual_'+runtime_version+'.py')
    require(sha(runtime_path)==expected_runtime[runtime_version] and sha(entry_path)==expected_entry[runtime_version],
            'Unreviewed evaluation runtime/entry semantics')
    normalized=runtime_path.read_text(encoding='utf-8')
    if version=='v6':
        normalized=normalized.replace('from phase_gated_residual_policy_v6 import EpisodeActor, features as v6_features',
             'from phase_residual_policy_v5 import EpisodeActor, features as v5_features')
        normalized=normalized.replace("== 'v6'","== 'v5'").replace("=='v6'","=='v5'").replace('v6_features(', 'v5_features(')
    equivalent=hashlib.sha256(normalized.encode()).hexdigest()
    require(equivalent==expected_runtime['v5'],'Evaluation control/audit/termination semantics changed')
    saved=torch.load(run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    metadata=saved['metadata']
    if version=='v4':
        qualification=run/'reviewed_transition_qualification.json';q=read_json(qualification)
        require(sha(qualification)==identity['qualification_sha256'],'V4 qualification binding differs')
        require(identity['evaluated_policy_checkpoint_sha256']==identity['frozen_v4_baseline_checkpoint_sha256']==q['frozen_baseline_checkpoint_sha256'],
                'Matched V4 behavior is not the frozen baseline')
        require(q['schema']=='v5_reviewable_development_bc_transition_qualification_v1' and q['eligible_for_rl_episode_training']is False and q['independent_generalization_claim']is False,
                'Matched V4 development qualification changed')
        require(q['physics_family_sha256']==identity['actual_physics_family_sha256'],'Matched V4 qualified physics differs')
        restore_recorded_model(run,identity,saved)
        return {'policy_version':version,'runtime_sha256':expected_runtime[runtime_version],'entry_sha256':expected_entry[runtime_version],
                'runtime_after_only_policy_routing_rename_sha256':equivalent,
                'same_original_physics_step_projection_right_action_query_order_gates_termination_and_audit':True,
                'frozen_v4_checkpoint_equals_baseline':True,'recorded_actor_feature_count':198,
                'supervision_features_reconstructed_from_actual_state_count':284,'reviewed_qualification_sha256':sha(qualification)}
    require(metadata['feature_schema']==policy.FEATURE_SCHEMA,'Checkpoint actor feature schema mismatch')
    require(metadata['physics_family_sha256']==identity['actual_physics_family_sha256'],'Checkpoint physics family mismatch')
    require(metadata['frozen_baseline_checkpoint_sha256']==identity['frozen_v4_baseline_checkpoint_sha256'],'Checkpoint baseline mismatch')
    require(metadata['teacher_hidden_state_or_labels_in_actor'] is False and metadata['previous_executed_source_in_actor'] is False,
            'Checkpoint actor isolation metadata changed')
    for name,digest in metadata['source_sha256'].items():
        candidates=[folder/name,run/'source_snapshot'/name]
        require(any(p.exists()and sha(p)==digest for p in candidates),'Checkpoint source not matched to archived bytes: '+name)
    qualification=run/'reviewed_transition_qualification.json'
    require(sha(qualification)==metadata['reviewed_qualification_sha256']==identity['qualification_sha256'],'Executed checkpoint/qualification binding differs')
    q=read_json(qualification)
    require(q['eligible_for_rl_episode_training'] is False and q['independent_generalization_claim'] is False and q['original_result_files_modified'] is False,
            'Original development qualification scope changed')
    require(q['source_sha256']==metadata['source_sha256'],'Qualification/checkpoint source binding differs')
    require(q['physics_family_sha256']==metadata['physics_family_sha256'] and q['frozen_baseline_checkpoint_sha256']==metadata['frozen_baseline_checkpoint_sha256'],
            'Qualification model domain/baseline binding differs')
    require(q['residual_bound_rad']==metadata['residual_bound_rad'],'Qualification residual bounds differ')
    require([d['evidence']for d in q['datasets']]==metadata['dataset_evidence'],'Qualification/checkpoint development evidence differs')
    if version=='v5':
        require(metadata['schema']==policy.SCHEMA and q['schema']=='v5_reviewable_development_bc_transition_qualification_v1','Unsupported V5 checkpoint/qualification schema')
    else:
        import phase_gated_residual_policy_v6 as gated
        require(sha(gated.__file__)=='874d82a6ffff395868260f48530b6fa0cfd20111cd133b827ad4809ed4967063','Reviewed V6 policy source changed')
        require(metadata['schema']==gated.SCHEMA and metadata['gate_contract']==identity['public_residual_gate_contract']==gated.GATE_CONTRACT,
                'V6 public gate contract changed')
        require(metadata['residual_bound_rad']==identity['residual_bound_rad']==q['residual_bound_rad'],'V6 bounds contract changed')
        require(q['schema']=='v6_reviewable_mixed_development_bc_transition_qualification_v1' and q['model_schema']==metadata['schema'] and q['feature_schema']==metadata['feature_schema'] and q['gate_contract']==metadata['gate_contract'],
                'V6 qualification model/feature/gate contract changed')
    restore_recorded_model(run,identity,saved)
    return {'policy_version':version,'runtime_sha256':expected_runtime[version],'entry_sha256':expected_entry[version],
            'runtime_after_only_policy_routing_rename_sha256':equivalent,
            'same_original_physics_step_projection_right_action_query_order_gates_termination_and_audit':True,
            'checkpoint_sources_match_recorded_snapshots':True,'checkpoint_frozen_baseline_and_control_buffers_verified':True,
            'reviewed_qualification_sha256':sha(qualification)}


def restore_recorded_model(run,identity,saved):
    """Reconstruct pinned behavior model; no unchecked saved control buffers."""
    import phase_basis_policy_v4,phase_contact_policy
    metadata=saved['metadata'];state=saved['model']
    for module in (phase_basis_policy_v4,phase_contact_policy):
        require(sha(module.__file__)==metadata['source_sha256'][Path(module.__file__).name],
                'Actual frozen model dependency differs from checkpoint')
    baseline=torch.load(run/'checkpoints/frozen_v4_baseline.pt',map_location='cpu',weights_only=True)
    if identity['policy']=='v4':
        require(sha(run/'checkpoints/evaluated_policy.pt')==sha(run/'checkpoints/frozen_v4_baseline.pt'),'V4 snapshot bytes differ')
        require(all(torch.isfinite(value).all()for value in state.values()),'Nonfinite V4 behavior')
        return phase_basis_policy_v4.restored(baseline).cpu().eval()
    for name,value in baseline['model'].items():
        candidate=state['baseline.'+name]
        require(candidate.dtype==value.dtype and candidate.shape==value.shape and candidate.numpy().tobytes()==value.numpy().tobytes(),
                'Checkpoint frozen baseline tensor changed: '+name)
    if identity['policy']=='v5':
        cls=policy.PhaseResidualPolicy;control_buffers=('residual_bound',)
    else:
        from phase_gated_residual_policy_v6 import PhaseGatedResidualPolicy
        cls=PhaseGatedResidualPolicy;control_buffers=('residual_bound','residual_phase_active','squeeze_active_progress')
    model=cls(baseline,state['mean'],state['std'],metadata['residual_bound_rad'],metadata['hidden_size'])
    for key in control_buffers:
        expected=model.state_dict()[key];actual=state[key]
        require(expected.dtype==actual.dtype and expected.shape==actual.shape and expected.numpy().tobytes()==actual.numpy().tobytes(),
                'Checkpoint control buffer differs from canonical metadata: '+key)
    model.load_state_dict(state,strict=True)
    require(all(torch.isfinite(value).all()for value in model.state_dict().values()),'Nonfinite checkpoint model state')
    return model.cpu().eval()


def _has_applied(node):
    return (isinstance(node.get('targets'),dict) and any(k.startswith('_applied_') for k in node['targets'])) or any(_has_applied(v) for v in node.values() if isinstance(v,dict))

def _sync(audit,names,applied):
    before=deepcopy(audit['teacher_state_before_sync']);mode=audit['execution_memory_mode']
    require(mode==('sync_actual_applied' if _has_applied(before) else 'preserve'),'Memory sync mode differs from frozen initialization rule')
    if mode=='preserve':require(not audit['execution_memory_changes'],'Preserve query changed execution memory')
    seen=set()
    for change in audit['execution_memory_changes']:
        parts=change['path'].split('.')
        require(len(parts)>=3 and parts[0]=='teacher' and parts[-2]=='targets' and parts[-1].startswith('_applied_'),'Only explicit applied-memory sync allowed')
        require(change['path'] not in seen,'Duplicate memory-sync field');seen.add(change['path'])
        name=parts[-1][len('_applied_'):];require(name in names,'Unknown memory-sync joint')
        require(change['after']==applied[names.index(name)],'Memory sync does not use actual prior command')
        node=before
        for key in parts[1:-1]:node=node[key]
        require(node.get(parts[-1])==change['before'],'Memory-sync before value changed');node[parts[-1]]=change['after']
    require(core.canonical(before)==core.canonical(audit['teacher_state_after_sync']),'Unlogged teacher-memory mutation')
    if mode=='sync_actual_applied':
        def check(node):
            for key,value in node.get('targets',{}).items():
                if key.startswith('_applied_'):
                    name=key[len('_applied_'):];require(name in names and value==applied[names.index(name)],'Stale synchronized applied memory')
            for value in node.values():
                if isinstance(value,dict):check(value)
        check(before)

def _gate(row,reset,sources):
    proxy=SimpleNamespace(env=SimpleNamespace(cube=SimpleNamespace(pose_reset_count=reset if row is None else row['pose_reset_count'])),initial_reset_count=reset,source_sha256=sources)
    return runtime.BaselineRuntime._gate(proxy,row)


def verify_successful_v4_wrist(rows,names,teacher_snapshot,result,protocol):
    import palm_handover_audit,wrist_audit
    for module in (palm_handover_audit,wrist_audit):
        require(sha(module.__file__)==protocol['source_sha256'][Path(module.__file__).name],'Original final auditor source changed')
    measured=palm_handover_audit.assess_handover(rows,names,regrasp=False,full_episode=True)
    wrist=wrist_audit.assess_wrist([r if r['layer_measurement']['assembled_for_grasp']else r|{'phase':'cube_not_assembled'}for r in rows],names)
    hold=longest=0
    for r in rows:
        hold=hold+1 if r['phase']=='tilt_hold'and r['layer_measurement']['assembled_for_grasp']else 0
        longest=max(longest,hold)
    accepted=bool(teacher_snapshot['accepted']and measured['accepted']and wrist['accepted']and longest>=90)
    report={'teacher_accepted':bool(teacher_snapshot['accepted']),'measured':measured,'wrist':wrist,'assembled_hold_s':longest/30.,'qualified':accepted}
    require(accepted and report==result['original_grasp_gate'],'Original successful V4 wrist/grasp auditors do not reproduce')
    return report

def load_run(directory,baseline_saved,baseline_sha,expected_family=None):
    evidence=verify_evidence(directory);run=Path(evidence['directory']);family=evidence['physics_family_sha256']
    if expected_family is not None:require(family==expected_family,'Different exact physics family')
    identity=read_json(run/'evaluation_protocol.json');protocol=read_json(run/'runtime_protocol.json');result=read_json(run/'result.json')
    require(identity['frozen_v4_baseline_checkpoint_sha256']==baseline_sha,'Different frozen v4 base')
    for key in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim','whole_robot_pure_neural_episode'):
        require(result.get(key) is False,'Original diagnostic False flag changed: '+key)
    successful_v4=identity['policy']=='v4'
    require((result['failure']is None and result['original_wrist_gate_qualified']is True)if successful_v4 else
            (bool(result['failure'])and result['original_wrist_gate_qualified']is False),'Behavior outcome is outside reviewed success/failed-prefix scope')
    require(result['evaluation_identity']==identity and result['left_teacher_position_action_steps']==0,'Result/evaluation identity mismatch')
    metadata=baseline_saved['metadata'];names=metadata['joint_names'];left=metadata['output_indices']
    require(len(names)==56 and len(set(names))==56 and left==[i for i,n in enumerate(names) if n.endswith('_L') or n.startswith('left_')],'Exact ordered left joint contract invalid')
    require(len(left)==28 and read_json(run/'config.json')['joint_names']==names,'Dataset joint order mismatch')
    saved=torch.load(run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    require(saved['metadata']['joint_names']==names and saved['metadata']['output_indices']==left,'Evaluated model joint order mismatch')
    behavior_model=restore_recorded_model(run,identity,saved)
    if successful_v4:
        import state_residual_eval_runtime_v5 as old_eval
        require(sha(old_eval.__file__)=='90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64','V4 behavior actor source changed')
        behavior=old_eval.V4EpisodeActor(behavior_model,left)
    else:behavior=policy.EpisodeActor(behavior_model,left)
    behavior.reset(identity['episode_name'])
    require(len(metadata['sources'])==1 and metadata['sources'][0]['manifest_sha256']==protocol['source_manifest']['manifest_sha256'],'Frozen phase-source manifest mismatch')
    parameters=read_json(run/'projection_parameters.json');require(parameters['joint_names']==names,'Projection joint order mismatch')
    physics=read_json(run/'physics_identity.json')
    projector=verified_projector(parameters,{'source_sha256':protocol['source_sha256'],'physics_identity':physics})
    tensors={k:torch.tensor(parameters[k],dtype=torch.float32) for k in ('limits','scales','step_limit','command_velocity_limits')}
    def project(previous,native):
        request=previous+tensors['scales']*native
        target,hit=projector(previous,request,tensors['limits'],tensors['step_limit'],tensors['command_velocity_limits'],parameters['control_dt'])
        return request,target,hit
    initial=read_json(run/'initial_state.json');reset=initial['oracle_env_inputs']['pose_reset_count']
    require(initial['oracle_env_inputs']['episode_length']==0 and initial['previous_executed_source']=='reset','Prefix does not start at actual reset')
    xs=[];previous=[];labels=[];masks=[];excluded=[];seen=Counter();modes=Counter();memory_changes=0;gains=0
    old_post=initial;old_raw=None;old_metrics=initial['oracle_env_inputs']['metrics'];old_teacher=None;old_settings=initial['control_settings']
    ended=False;final=None;query_count=0;abort_seen=False;terminal_reason=None;physical_rows=[];success_audit=None
    plans=set(evidence['plan_sha256'].values())
    with (run/'evaluation_transitions.jsonl').open() as stream,(run/'trajectory.jsonl').open() as raw_stream:
        for line in stream:
            require(line.endswith('\n'),'Incomplete journal line');record=parse_line(line);kind=record['record_type']
            require(final is None,'Record after final summary')
            if kind=='evaluation_final':require(ended,'Final summary before terminal event');final=record['summary'];continue
            require(not ended,'Record after terminal event');require(record['step']==len(xs),'Skipped/reordered actual step')
            require(not record['invalid_evidence_fields'],'Invalid evidence fields cannot supervise')
            if kind=='abort':
                require(record['physical_steps']==len(xs) and record['fully_audited_transition_steps']==len(xs) and record['trajectory_rows_recorded']==len(xs),'Abort counter mismatch')
                require(record['physics_step_completed_for_unfinished_decision'] is False,'Unrecorded physical step at abort')
                decision=record['unfinished_decision'];require(decision['step']==len(xs) and decision['left_teacher_position_action_executed'] is False,'Invalid unexecuted decision')
                require('execution' not in decision and 'predicted_left_target' not in decision,'Only pre-execution teacher-failure abort is supported')
                audit=decision['original_controller_audit']
                require(audit['query_index']==query_count and audit['failure_reason']==record['reason']==result['failure'],'Abort query/reason mismatch')
                require(audit['oracle_env_inputs']['episode_length']==len(xs),'Abort is not on actual final state')
                _sync(audit,names,old_post['applied_target'])
                if old_teacher is not None:require(core.canonical(audit['teacher_state_before_sync'])==core.canonical(old_teacher),'Teacher state discontinuity at abort')
                query_count+=1;abort_seen=True;ended=True;terminal_reason=record['reason']
                excluded.append({'step':record['step'],'reason':record['reason'],'type':'unexecuted_abort_no_label'});continue
            require(kind=='executed_transition' and record['transition_finished'] is True,'Unknown/nonexecuted transition')
            raw_line=next(raw_stream,None);require(raw_line is not None and raw_line.endswith('\n'),'Missing complete trajectory row');raw=parse_line(raw_line)
            physical_rows.append(raw)
            require(raw['step']==record['step'] and record['environment_episode_counter_before_step']==record['step'],'Actual counter mismatch')
            pre=record['actor_observation_pre'];post=record['actor_observation_post'];audit=record['original_controller_audit'];label=record['counterfactual_expert_projection_diagnostic_only'];execution=record['execution']
            require(record['left_teacher_position_action_executed'] is False and execution['left_teacher_position_action_executed'] is False and record['corrective_labels_qualified_for_training'] is False,'Original diagnostic/intervention flags changed')
            for key in core.STATE_FIELDS:require(core.canonical(pre[key])==core.canonical(old_post[key]),'Actual-history discontinuity: '+key)
            core.validate_gate(_gate(old_raw,reset,protocol['source_sha256']))
            require(audit['query_index']==query_count and not audit['failure_reason'],'Oracle query discontinuity/failure');query_count+=1
            oracle=audit['oracle_env_inputs'];require(oracle['episode_length']==record['step'] and oracle['pose_reset_count']==reset,'Oracle actual-state counter/reset mismatch')
            require(core.canonical(oracle['metrics'])==core.canonical(old_metrics),'Oracle metrics not from actual preceding execution')
            if old_teacher is not None:require(core.canonical(audit['teacher_state_before_sync'])==core.canonical(old_teacher),'Teacher hidden-state discontinuity')
            schedule=audit['public_same_action_schedule'];phase=pre['phase'];duration=protocol['phase_counts'][phase]
            require(phase==raw['phase']==schedule['phase']==audit['teacher_state_after_query']['name'],'Action phase mismatch')
            require(pre['phase_progress']==schedule['phase_progress']==min(1.,seen[phase]/max(1,duration-1)),'Action progress/clock mismatch');seen[phase]+=1
            _sync(audit,names,pre['applied_target']);modes[audit['execution_memory_mode']]+=1;memory_changes+=len(audit['execution_memory_changes'])
            for digest in (audit['teacher_state_after_query'].get('path_sha256'),audit['teacher_state_after_query'].get('base',{}).get('phases_sha256')):
                if digest is not None:require(digest in plans,'Referenced teacher plan was not archived')
            before=audit['control_settings_before'];after=audit['control_settings_after'];require(before==old_settings,'Unlogged gain-state change')
            changed=before!=after;require(audit['teacher_runtime_control_setting_change']==changed,'Gain event audit mismatch')
            if changed:
                proxy=SimpleNamespace(joint_names=names)
                require(runtime.BaselineRuntime.validate_control_change(proxy,before,after,audit['teacher_state_before_sync'],audit['teacher_state_after_query']),'Unapproved teacher gain mutation');gains+=1
            old=torch.tensor(pre['applied_target'],dtype=torch.float32).reshape(1,56);native=torch.tensor(audit['normalized_action'],dtype=torch.float32).reshape(1,56)
            request,target,hit=project(old,native)
            require(bits_equal(request[0],label['requested_target']) and bits_equal(request[0],audit['requested_target']),'Expert requested action mismatch')
            require(bits_equal(target[0],label['projected_target']) and label['projection_soft_limit_hit']==hit[0].tolist(),'Expert projected label mismatch')
            require(bits_equal(label['projection_previous_applied_target'],pre['applied_target']),'Expert label used wrong previous command')
            require(label['projection_parameters_file']=='projection_parameters.json' and label['projection_parameters_file_sha256']==evidence['files_sha256']['projection_parameters.json'] and label['projection_source_sha256']==parameters['source_sha256'],'Expert projection provenance mismatch')
            require(label['projected_label_dtype']=='torch.float32' and label['projected_label_scope']=='expert_from_actual_prestate_not_student_actual_action','Expert label scope changed')
            selected=torch.tensor(execution['native_action'],dtype=torch.float32).reshape(1,56);composed=native.clone()
            composed[:,left]=(torch.tensor(record['predicted_left_target'],dtype=torch.float32).reshape(1,28)-old[:,left])/tensors['scales'][:,left]
            require(bits_equal(selected,composed) and bits_equal(raw['native_action'],selected[0]),'Left composition/right native action mismatch')
            require(execution['right_native_bits_preserved'] is True and execution['actual_projection_bits_verified'] is True,'Execution bit audit missing')
            requested,actual,actual_hit=project(old,selected)
            require(bits_equal(execution['previous_applied_target'],pre['applied_target']) and bits_equal(execution['requested_target'],requested[0]) and execution['soft_limit_hit']==actual_hit[0].tolist(),'Executed projection request/limit audit mismatch')
            require(all(bits_equal(actual[0],value) for value in (post['applied_target'],raw['applied_target'],execution['expected_applied_target'])),'Actual applied command mismatch')
            for key in ('q','joint_velocity','cube_position','cube_quaternion_wxyz','finger_contact_impulses','finger_contact_points'):
                require(core.canonical(post[key])==core.canonical(raw[key]),'Post-state and independent trajectory differ: '+key)
            require(record['raw_environment_termination']==raw['termination'],'Raw termination streams disagree')
            termination=raw['termination'];timeout=record['step']+1>=oracle['max_episode_length'];done=bool(raw['success'] or raw['drop'] or timeout)
            require(termination==runtime.termination_status(done,raw,record['step']+1,oracle['max_episode_length']),'Original complete termination contract differs')
            require(termination['raw_done']==done and bool(raw['terms']['terminal'])==done and bool(raw['terms']['timeout'])==timeout and bool(raw['terms']['drop'])==bool(raw['drop']),'Original done/terms mismatch')
            require(termination['done_matches_original_causes'] is True and termination['sampling_terminal']==done and termination['derived_timeout_truncated']==timeout and termination['derived_drop_terminated']==bool(raw['drop']),'Raw terminal causes inconsistent')
            rederived=_gate(raw,reset,protocol['source_sha256']);post_gates=record['original_stage_post_gates']
            for key,value in rederived.items():
                if key!='stage_gate_qualified':require(post_gates[key]==value,'Original post-gate differs: '+key)
            qualified=bool(record['original_wrist_gate_qualified']);require(post_gates['stage_gate_qualified']==qualified,'Stage gate/qualification mismatch')
            if qualified:
                require(successful_v4,'Failed-prefix consumer cannot accept a successful wrist boundary')
                success_audit=verify_successful_v4_wrist(physical_rows,names,audit['teacher_state_after_query'],result,protocol)
            safe=all(rederived[k] for k in ('robot_safety','joint_limits','cube_integrity','no_pose_write')) and not done and not record['failure_reason']
            require(record['terminal']==bool(record['failure_reason'] or qualified),'Terminal flag disagrees with frozen driver')
            if not safe:
                require(record['terminal'] is True,'Unsafe nonterminal evaluation continued');excluded.append({'step':record['step'],'reason':record['failure_reason'] or 'raw_done_or_post_gate','type':'excluded_executed_transition'})
            x=policy.features({**pre,'previous_executed_source':'student'})
            if successful_v4:
                from phase_contact_policy import features as v4_features
                recorded_features=v4_features(pre,phase,pre['phase_progress'])
            else:recorded_features=x
            require(bits_equal(recorded_features,record['actual_actor_features']),'Recorded actor features differ')
            predicted=behavior.act({**pre,'previous_executed_source':'student'},episode_id=identity['episode_name'],step=len(xs))
            require(bits_equal(predicted,record['predicted_left_target']),'Recorded prediction differs from actual complete-history GRU replay')
            policy.features({**post,'phase':phase,'phase_progress':pre['phase_progress'],'previous_executed_source':'student'})
            xs.append(x);previous.append(np.asarray(pre['applied_target'],dtype=np.float32)[left]);labels.append(target[0,left].numpy());masks.append(safe)
            old_post=post;old_raw=raw;old_metrics=record['raw_environment_metrics'];old_teacher=audit['teacher_state_after_query'];old_settings=after;ended=bool(record['terminal'])
            if ended:terminal_reason=record['failure_reason']
        require(next(raw_stream,None) is None,'Unrecorded physics row after transition stream')
    require(ended and final is not None and xs,'Incomplete evaluation prefix')
    for key in ('steps','physical_steps','environment_episode_counter','fully_audited_transition_steps','action_execution_dispatch_attempts','trajectory_rows_recorded'):
        require(result[key]==final[key]==len(xs),'Completed physical count mismatch: '+key)
    require(result['teacher_query_count']==final['teacher_query_count']==query_count,'Oracle query total mismatch')
    require(result['original_teacher_gain_schedule_events']==final['original_teacher_gain_schedule_events']==gains,'Gain total mismatch')
    require(final['failure']==result['failure']==terminal_reason and final['corrective_labels_qualified_for_training'] is False and final['pure_student_episode_claim'] is False,'Final diagnostic scope/reason differs')
    require(final['original_wrist_gate_qualified']==successful_v4 and (success_audit is not None)==successful_v4,'Final success boundary audit differs')
    require(any(masks),'No safe candidate transitions')
    return {'x':np.stack(xs),'previous_left':np.stack(previous),'labels':np.stack(labels),'mask':np.asarray(masks,dtype=bool),
        'left_indices':left,'joint_names':names,'parameters':parameters,'projector':projector,'executed_sources':['student']*len(xs),
        'physics_family_sha256':family,'evidence':evidence,'excluded_labels':excluded,'collection_source_sha256':protocol['source_sha256'],
        'episode_id':str(run)+':'+identity['episode_name'],'result':result,'consumer_schema':SCHEMA,
        'audit':{'oracle_queries':query_count,'excluded_unexecuted_abort':abort_seen,'execution_memory_modes':dict(modes),'memory_changes':memory_changes,'gain_events':gains,
                 'behavior_predictions_replayed_bitwise':True,'recorded_actor_feature_count':identity['actor_feature_count'],
                 'supervision_feature_count':284,'original_successful_wrist_audit':success_audit},
        'scope':'development_actual_state_bc_candidates_original_success_failure_and_episode_flags_unchanged'}

def verify_saved_datasets(metadata,directories):
    expected=metadata['dataset_evidence'];require(len(directories)==len(expected),'All eval-prefix datasets required')
    for directory,old in zip(directories,expected):
        current=verify_evidence(directory)
        require(current==old,'Eval-prefix dataset provenance changed')
        require(current['physics_family_sha256']==metadata['physics_family_sha256'],'Checkpoint physics family differs')
