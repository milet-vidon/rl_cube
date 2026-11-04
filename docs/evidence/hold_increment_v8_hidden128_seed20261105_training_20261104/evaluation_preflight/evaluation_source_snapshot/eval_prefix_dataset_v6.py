"""Strict read-only consumer of completed diagnostic V5 evaluation prefixes.

This returns BC candidates, never permission to train or a successful episode.
Original diagnostic/episode flags remain false. A separately pinned, reviewed
V6 qualification must authorize the exact candidates, sources, mask and bounds.
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

SCHEMA='completed_diagnostic_eval_safe_prefix_v6'
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
    require(identity['policy']=='v5' and identity['split']=='development_seen_seed','This consumer supports the declared development V5 diagnostic only')
    require(identity['left_policy_begins_at_reset_step']==0 and identity['actor_hidden_reset_count']==1 and identity['left_teacher_position_action_steps']==0,'Missing from-reset/no-left-intervention contract')
    require(identity['expert_hidden_or_labels_in_actor'] is False and identity['executor_identity_in_actor_features'] is False,'Actor data isolation contract changed')
    require(identity['actor_feature_schema']==policy.FEATURE_SCHEMA and identity['actor_feature_count']==policy.FEATURES,'Recorded actor feature layout changed')
    require(protocol['gate_threshold_changes'] is False and protocol['center_oracle_implemented'] is False and protocol['eligible_training_episode'] is False,'Frozen gate/episode scope changed')
    plans={}
    for path in sorted((run/'plans').glob('*.json')):
        digest=hashlib.sha256(core.canonical(read_json(path)).encode()).hexdigest()
        require(path.stem.rsplit('_',1)[-1]==digest,'Plan content/filename hash mismatch')
        plans[path.name]=digest
    require(bool(plans),'Missing archived original teacher plans')
    return {'consumer_schema':SCHEMA,'directory':str(run),'artifact_manifest_sha256':sha(run/'artifact_manifest.json'),
        'files_sha256':files,'source_snapshot_sha256':snapshots,'checkpoint_sha256':checkpoint_hashes,
        'plan_sha256':plans,'physics_family_sha256':family}

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

def load_run(directory,baseline_saved,baseline_sha,expected_family=None):
    evidence=verify_evidence(directory);run=Path(evidence['directory']);family=evidence['physics_family_sha256']
    if expected_family is not None:require(family==expected_family,'Different exact physics family')
    identity=read_json(run/'evaluation_protocol.json');protocol=read_json(run/'runtime_protocol.json');result=read_json(run/'result.json')
    require(identity['frozen_v4_baseline_checkpoint_sha256']==baseline_sha,'Different frozen v4 base')
    for key in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim','whole_robot_pure_neural_episode'):
        require(result.get(key) is False,'Original diagnostic False flag changed: '+key)
    require(bool(result['failure']) and result['original_wrist_gate_qualified'] is False,'This failed-prefix consumer cannot relabel success')
    require(result['evaluation_identity']==identity and result['left_teacher_position_action_steps']==0,'Result/evaluation identity mismatch')
    metadata=baseline_saved['metadata'];names=metadata['joint_names'];left=metadata['output_indices']
    require(len(names)==56 and len(set(names))==56 and left==[i for i,n in enumerate(names) if n.endswith('_L') or n.startswith('left_')],'Exact ordered left joint contract invalid')
    require(len(left)==28 and read_json(run/'config.json')['joint_names']==names,'Dataset joint order mismatch')
    saved=torch.load(run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    require(saved['metadata']['joint_names']==names and saved['metadata']['output_indices']==left,'Evaluated model joint order mismatch')
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
    ended=False;final=None;query_count=0;abort_seen=False;terminal_reason=None
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
            require(not qualified,'Failed-prefix consumer cannot accept a successful wrist boundary')
            safe=all(rederived[k] for k in ('robot_safety','joint_limits','cube_integrity','no_pose_write')) and not done and not record['failure_reason']
            require(record['terminal']==bool(record['failure_reason'] or qualified),'Terminal flag disagrees with frozen driver')
            if not safe:
                require(record['terminal'] is True,'Unsafe nonterminal evaluation continued');excluded.append({'step':record['step'],'reason':record['failure_reason'] or 'raw_done_or_post_gate','type':'excluded_executed_transition'})
            x=policy.features({**pre,'previous_executed_source':'student'});require(bits_equal(x,record['actual_actor_features']),'Recorded actor features differ')
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
    require(any(masks),'No safe candidate transitions')
    return {'x':np.stack(xs),'previous_left':np.stack(previous),'labels':np.stack(labels),'mask':np.asarray(masks,dtype=bool),
        'left_indices':left,'joint_names':names,'parameters':parameters,'projector':projector,'executed_sources':['student']*len(xs),
        'physics_family_sha256':family,'evidence':evidence,'excluded_labels':excluded,'collection_source_sha256':protocol['source_sha256'],
        'episode_id':str(run)+':'+identity['episode_name'],'result':result,'consumer_schema':SCHEMA,
        'audit':{'oracle_queries':query_count,'excluded_unexecuted_abort':abort_seen,'execution_memory_modes':dict(modes),'memory_changes':memory_changes,'gain_events':gains},
        'scope':'development_safe_prefix_bc_candidates_only_original_failed_diagnostic_flags_unchanged'}

def verify_saved_datasets(metadata,directories):
    expected=metadata['dataset_evidence'];require(len(directories)==len(expected),'All eval-prefix datasets required')
    for directory,old in zip(directories,expected):
        current=verify_evidence(directory)
        require(current==old,'Eval-prefix dataset provenance changed')
        require(current['physics_family_sha256']==metadata['physics_family_sha256'],'Checkpoint physics family differs')
