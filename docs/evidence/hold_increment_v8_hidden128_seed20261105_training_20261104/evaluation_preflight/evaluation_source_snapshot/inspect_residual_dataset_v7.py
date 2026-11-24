"""Five verified development histories; inspection only, no optimizer or training."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,math
import numpy as np
import torch
import dagger_residual_dataset_v5 as v3
import eval_prefix_dataset_v7 as reader
import train_phase_residual_v5 as frozen
import train_phase_gated_residual_v6 as candidate
import phase_gated_residual_policy_v6 as policy
import zero_gated_residual_policy_v7 as zero_policy

RUN_NAMES=('left_dagger_baseline_mixture_v3_9501_20261104','left_state_residual_v5_long_eval_9501_20261104_v2',
           'left_phase_gated_residual_v6_eval_9501_20261104','left_phase_gated_residual_v6_long_eval_9501_20261104',
           'left_phase_v4_matched_eval_9501_20261104_v2')
ARM_BOUNDS={f'Joint{i+1}_L':value for i,value in enumerate((.0248,.0243,.0249,.0076,.0257,.0256,.0244))}
BOUND_SCHEMA='five_development_history_observed_arm_range_plus_1mrad_v7_candidate'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return reader.read_json(path)
def source_hashes():
    return {Path(module.__file__).name:sha(module.__file__)for module in (reader,v3,policy,candidate,frozen,zero_policy)}|{Path(__file__).name:sha(__file__)}
def lines(path):
    with Path(path).open()as stream:
        for line in stream:yield reader.parse_line(line)

def supplement_mixture(run,data):
    """Recheck query/hidden memory, source plans and original gate reconstruction."""
    protocol=read(run/'runtime_protocol.json');initial=read(run/'initial_state.json');manifest=read(run/'artifact_manifest.json')['sha256']
    for name,digest in manifest.items():reader.require(sha(run/name)==digest,'Mixture manifest mismatch: '+name)
    for name,digest in protocol['source_sha256'].items():reader.require(sha(run/'source_snapshot'/name)==digest,'Mixture snapshot mismatch: '+name)
    plans=set()
    for path in (run/'plans').glob('*.json'):
        digest=hashlib.sha256(reader.core.canonical(read(path)).encode()).hexdigest()
        reader.require(path.stem.rsplit('_',1)[-1]==digest,'Mixture plan content differs');plans.add(digest)
    old_teacher=None;old_raw=None;old_settings=initial['control_settings'];seen=Counter();modes=Counter();changes=gains=0
    reset=initial['oracle_env_inputs']['pose_reset_count'];count=0
    for record,raw in zip(lines(run/'transitions.jsonl'),lines(run/'trajectory.jsonl'),strict=True):
        expert=record['expert_supervision'];pre=record['actor_observation_pre'];post=record['actor_observation_post'];step=record['step']
        reader.require(expert['query_index']==step==count,'Mixture once-query counter mismatch')
        if old_teacher is not None:reader.require(reader.core.canonical(expert['teacher_state_before_sync'])==reader.core.canonical(old_teacher),'Mixture teacher-memory continuity differs')
        reader._sync(expert,data['joint_names'],pre['applied_target']);modes[expert['execution_memory_mode']]+=1;changes+=len(expert['execution_memory_changes'])
        phase=pre['phase'];reader.require(pre['phase_progress']==min(1.,seen[phase]/max(1,protocol['phase_counts'][phase]-1)),'Mixture public clock differs');seen[phase]+=1
        reader.require(record['pre_gates']==reader._gate(old_raw,reset,protocol['source_sha256']),'Mixture original pre-gate differs')
        recomputed=reader._gate(raw,reset,protocol['source_sha256']);recomputed['stage_gate_qualified']=record['stage_complete']
        reader.require(record['post_gates']==recomputed,'Mixture original post-gate differs')
        oracle=expert['oracle_env_inputs'];metrics=oracle['metrics']
        reader.require(oracle['episode_length']==step and oracle['pose_reset_count']==reset,'Mixture oracle actual-state counter differs')
        if old_raw is None:reader.require(metrics==initial['oracle_env_inputs']['metrics'],'Mixture initial metrics differ')
        else:
            for key,value in metrics.items():reader.require(key in old_raw and reader.core.canonical(value)==reader.core.canonical(old_raw[key]),'Mixture actual prior metrics differ: '+key)
        term=raw['termination'];reader.require(term==reader.runtime.termination_status(term['raw_done'],raw,step+1,oracle['max_episode_length']),'Mixture full raw termination contract differs')
        for key in ('q','joint_velocity','cube_position','cube_quaternion_wxyz','finger_contact_impulses','finger_contact_points'):
            reader.require(reader.core.canonical(post[key])==reader.core.canonical(raw[key]),'Mixture physical post-state differs: '+key)
        before=expert['control_settings_before'];after=expert['control_settings_after'];reader.require(before==old_settings,'Mixture unlogged gain change')
        changed=before!=after;reader.require(changed==expert['teacher_runtime_control_setting_change'],'Mixture gain event mismatch')
        if changed:
            reader.require(reader.runtime.BaselineRuntime.validate_control_change(SimpleNamespace(joint_names=data['joint_names']),before,after,expert['teacher_state_before_sync'],expert['teacher_state_after_query']),'Mixture invalid gain event');gains+=1
        for digest in (expert['teacher_state_after_query'].get('path_sha256'),expert['teacher_state_after_query'].get('base',{}).get('phases_sha256')):
            if digest is not None:reader.require(digest in plans,'Mixture missing plan snapshot')
        old_teacher=expert['teacher_state_after_query'];old_settings=after;old_raw=raw;count+=1
    for key in ('eligible_training_episode','runtime_bridge_validated','pure_student_episode_claim'):
        reader.require(data['result'][key]is False,'Mixture original False flag changed')
    return {'queries':count,'memory_modes':dict(modes),'changed_memory_fields':changes,'gain_events':gains,
            'actual_state_and_teacher_memory_contiguous':True,'all_source_plan_and_gate_checks_passed':True}

def labels_diagnostic(run,data):
    is_eval=(run/'evaluation_protocol.json').exists();file='evaluation_transitions.jsonl'if is_eval else 'transitions.jsonl'
    nominal=[];requested=[];arrival=[];contact=[];plans=[];phases=[]
    for record in lines(run/file):
        if is_eval and record['record_type']!='executed_transition':continue
        expert=record['original_controller_audit']if is_eval else record['expert_supervision']
        state=expert['teacher_state_after_query'];target=state['nominal_target']
        nominal.append(np.asarray(target,dtype=np.float32).reshape(56)[data['left_indices']])
        requested.append(np.asarray(expert['requested_target'],dtype=np.float32)[data['left_indices']])
        arrival.append(state['base']['finger_arrival']['tick'])
        contact.append(record['actor_observation_pre']['finger_contact_impulses']['left'])
        phases.append(record['actor_observation_pre']['phase'])
        plans.append({'pickup_sha256':state['base'].get('phases_sha256'),'tilt_path_sha256':state.get('path_sha256'),
                      'mode':state.get('mode'),'tick':state.get('tick'),'wait':state.get('wait')})
    reader.require(len(nominal)==len(data['x']),'Diagnostic label stream length mismatch')
    return {'nominal':np.stack(nominal),'requested':np.stack(requested),'arrival_tick':np.asarray(arrival),'contacts':contact,'phases':phases,'plans':plans}

def zero_statistics(label,baseline):
    """Descriptive bins only; no small difference is automatically a zero label."""
    label=np.asarray(label,dtype=np.float32);baseline=np.asarray(baseline,dtype=np.float32)
    a=label.view(np.uint32);b=baseline.view(np.uint32);equal=a==b
    delta=np.abs(label.astype(np.float64)-baseline.astype(np.float64));small=(delta>0)&(delta<=1e-6)
    def ordered(bits):return np.where(bits&np.uint32(0x80000000),~bits,bits^np.uint32(0x80000000)).astype(np.int64)
    ulps=np.abs(ordered(a)-ordered(b))
    return {'elements':int(equal.size),'strict_bitwise_equal':int(equal.sum()),
        'numeric_zero_different_bits':int(((delta==0)&~equal).sum()),
        'nonzero_abs_le_1e_6':int(small.sum()),'nonzero_abs_gt_1e_6':int((delta>1e-6).sum()),
        'small_nonzero_max_abs_rad':float(delta[small].max())if small.any()else 0.,
        'small_nonzero_max_ulp_distance':int(ulps[small].max())if small.any()else 0,
        'small_nonzero_exactly_one_ulp':int((small&(ulps==1)).sum()),
        'small_nonzero_over_one_ulp':int((small&(ulps>1)).sum()),
        'grouping_changes_labels':False,'small_values_proven_to_be_rounding_only':False}

def delta_distribution(values):
    values=np.asarray(values,dtype=np.float64)
    choices=np.asarray([-.005,0.,.005]);distance=np.abs(values[:,None]-choices[None,:]);nearest=np.argmin(distance,axis=1)
    close=distance[np.arange(len(values)),nearest]<=1e-6
    return {'count':len(values),'min_rad':float(values.min()),'max_rad':float(values.max()),
        'zero_within_1e_6':int(np.sum(close&(nearest==1))),
        'negative_005_within_1e_6':int(np.sum(close&(nearest==0))),
        'positive_005_within_1e_6':int(np.sum(close&(nearest==2))),
        'other_than_0_or_plus_minus_005_count':int(np.sum(~close)),
        'noncanonical_max_nearest_distance_rad':float(distance.min(1).max()),
        'abs_quantiles':{str(p):float(np.quantile(np.abs(values),p))for p in (.5,.9,.99,1.)}}

def classification_summary(labels, projected, eligible, active):
    classes=zero_policy.classify_correction_labels(torch.from_numpy(labels),torch.from_numpy(projected))
    arrays={key:value.numpy()for key,value in classes.items()if key!='absolute_difference_rad'}
    result={'contract':zero_policy.CLASSIFICATION_CONTRACT,
            'full_history_boolean_array_sha256':{key:hashlib.sha256(value.tobytes(order='C')).hexdigest()for key,value in arrays.items()},
            'array_shape':list(labels.shape),'array_encoding':'numpy bool, C order, one byte per element',
            'continuous_labels_modified':False}
    for name,mask in [('all_eligible',eligible),('public_active_eligible',eligible&active),('public_masked_eligible',eligible&~active)]:
        result[name]={key:int(value[mask].sum())for key,value in arrays.items()}
        result[name]['elements']=int(mask.sum())*28
    return result

def distributions(data,model,extra):
    x,previous,labels,mask=frozen.tensor_episode(data,'cpu')
    with torch.no_grad():
        base=model.baseline_targets(x,previous)
        projected=policy.project_left_targets(base,previous,data['parameters'],data['left_indices'],data['projector'])[0].numpy()
        active=model.residual_active(x)[0].numpy()
        bound=model.effective_bounds(x)
        low=torch.clamp(base-bound,model.baseline.lower.float(),model.baseline.upper.float())
        high=torch.clamp(base+bound,model.baseline.lower.float(),model.baseline.upper.float())
        low=policy.project_left_targets(low,previous,data['parameters'],data['left_indices'],data['projector'])[0].numpy()
        high=policy.project_left_targets(high,previous,data['parameters'],data['left_indices'],data['projector'])[0].numpy()
        gap=np.maximum(np.maximum(np.minimum(low,high)-data['labels'],0),np.maximum(data['labels']-np.maximum(low,high),0))
    select=data['mask']&active
    names=[data['joint_names'][i]for i in data['left_indices']];reports={}
    for i,name in enumerate(names):
        if name not in policy.RECOVERY_JOINTS:continue
        target=data['labels'][:,i];prev=data['previous_left'][:,i];req=extra['requested'][:,i]
        finger=name.split('_')[1];touch=np.asarray([float(c.get(finger,0.))>1e-5 for c in extra['contacts']])
        reports[name]={'expert_projected_minus_frozen_base_projected':delta_distribution((target-projected[:,i])[select]),
            'expert_projected_minus_previous_actual_command':delta_distribution((target-prev)[select]),
            'expert_requested_minus_previous_actual_command':delta_distribution((req-prev)[select]),
            'expert_projected_minus_teacher_nominal':delta_distribution((target-extra['nominal'][:,i])[select]),
            'contact_active_frames':int(np.sum(select&touch)),
            'contact_active_expert_hold_within_1e_6':int(np.sum(select&touch&(np.abs(target-prev)<=1e-6))),
            'no_contact_active_frames':int(np.sum(select&~touch)),
            'max_expert_previous_distance_when_touching_rad':float(np.max(np.abs(target-prev)[select&touch]))if np.any(select&touch)else None}
    zero_by_joint={name:zero_statistics(data['labels'][:,i][data['mask']],projected[:,i][data['mask']])for i,name in enumerate(names)}
    zero_by_phase={phase:zero_statistics(data['labels'][data['mask']&(np.asarray(extra['phases'])==phase)],projected[data['mask']&(np.asarray(extra['phases'])==phase)])for phase in dict.fromkeys(extra['phases'])}
    phase_ranges={}
    for phase in dict.fromkeys(extra['phases']):
        selected=data['mask']&(np.asarray(extra['phases'])==phase)
        phase_ranges[phase]={name:{'expert_minus_base_projected_min_rad':float((data['labels'][:,i]-projected[:,i])[selected].min()),
                                  'expert_minus_base_projected_max_rad':float((data['labels'][:,i]-projected[:,i])[selected].max()),
                                  'first_step':int(np.flatnonzero(selected)[0]),'last_step':int(np.flatnonzero(selected)[-1])}for i,name in enumerate(names)}
    unreachable={}
    for i,name in enumerate(names):
        bad=select&(gap[:,i]>0)
        if not bad.any():continue
        ids=np.flatnonzero(bad);worst=int(ids[np.argmax(gap[ids,i])])
        unreachable[name]={'count':len(ids),'first_step':int(ids[0]),'last_step':int(ids[-1]),'phase_counts':dict(Counter(extra['phases'][j]for j in ids)),
            'max_distance_rad':float(gap[worst,i]),'worst_step':worst,'worst_phase':extra['phases'][worst],
            'worst_public_progress':float(data['x'][worst,197]),'worst_original_plan_state':extra['plans'][worst],
            'expert_projected':float(data['labels'][worst,i]),'expert_requested':float(extra['requested'][worst,i]),
            'baseline_projected':float(projected[worst,i]),'previous_actual_command':float(data['previous_left'][worst,i]),
            'teacher_nominal':float(extra['nominal'][worst,i]),
            'expert_minus_base':float(data['labels'][worst,i]-projected[worst,i]),
            'expert_minus_previous_actual':float(data['labels'][worst,i]-data['previous_left'][worst,i]),
            'expert_minus_teacher_nominal':float(data['labels'][worst,i]-extra['nominal'][worst,i])}
    return {'active_steps':int(select.sum()),'joints':reports,'feedback_arrival_tick_range':[int(extra['arrival_tick'].min()),int(extra['arrival_tick'].max())],
            'zero_equivalence_all_eligible':zero_statistics(data['labels'][data['mask']],projected[data['mask']]),
            'zero_equivalence_active':zero_statistics(data['labels'][select],projected[select]),
            'zero_equivalence_public_masked':zero_statistics(data['labels'][data['mask']&~active],projected[data['mask']&~active]),
            'zero_equivalence_by_joint':zero_by_joint,'zero_equivalence_by_phase':zero_by_phase,
            'reviewed_classification':classification_summary(data['labels'],projected,data['mask'],active),
            'phase_ranges_by_joint':phase_ranges,
            'all_28_targets_bitwise_baseline_frames':int(np.all(data['labels'].view(np.uint32)==projected.view(np.uint32),axis=1)[data['mask']].sum()),
            'unreachable_active_joint_details':unreachable}

def main():
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);initial_sources=source_hashes()
    baseline,baseline_hash=frozen.load_baseline(a.code/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
    episodes=[];family=None;diagnostics=[];supplements=[]
    for index,name in enumerate(RUN_NAMES):
        run=a.code/'runs'/name
        data=(v3 if index==0 else reader).load_run(run,baseline,baseline_hash,family)
        data['consumer_schema']='qualified_v3_mixture_with_v7_supplement'if index==0 else reader.SCHEMA
        family=data['physics_family_sha256'];episodes.append(data)
        supplements.append(supplement_mixture(run,data)if index==0 else data['audit'])
        diagnostics.append(labels_diagnostic(run,data));print('V7_INTEGRITY_PASSED',name,len(data['x']),int(data['mask'].sum()),flush=True)
    initial_model=candidate.build_model(baseline,episodes,64)
    left_names=[episodes[0]['joint_names'][i]for i in episodes[0]['left_indices']]
    bounds=zero_policy.candidate_bounds(left_names)
    reader.require(zero_policy.ARM_BOUND_CANDIDATE==ARM_BOUNDS,'Model and inspection arm candidate differ')
    model=policy.PhaseGatedResidualPolicy(baseline,initial_model.mean,initial_model.std,bounds,64)
    bound_report=candidate.inspect(episodes,model)
    observed={name:{'min_rad':bound_report['expert_projected_minus_frozen_base_min_rad'][i],
                    'max_rad':bound_report['expert_projected_minus_frozen_base_max_rad'][i]}for i,name in enumerate(left_names)if name in ARM_BOUNDS}
    for name,limits in observed.items():
        expected=math.ceil(max(abs(limits['min_rad']),abs(limits['max_rad']))*10000)/10000+.001
        reader.require(np.float32(expected)==np.float32(ARM_BOUNDS[name]),'Reviewed observed-range bound construction differs: '+name)
    bound_contract={'schema':BOUND_SCHEMA,'arm_bounds_rad':ARM_BOUNDS,'observed_arm_label_minus_base_proposal_ranges':observed,
        'rounding_quantum_rad':.0001,'margin_rad':.001,'algorithm':'ceil(max(abs(observed_min),abs(observed_max))/0.0001)*0.0001 + 0.001',
        'observations':'All five complete actual development histories, eligible original expert projected labels relative to frozen base proposal.',
        'finger_bounds':'Original 11 named feedback joints 0.006; remaining 10 finger joints 0.000001.',
        'original_soft_limits_velocity_projection_gates_changed':False,'counterfactual_label_coverage_only_not_executed_safety':True}
    compatible=bound_report['active_max_unreachable_distance_rad']==0. and bound_report['masked_corrections_gt_1e_6']==0
    distribution=[{'run':RUN_NAMES[i],**distributions(data,model,diagnostics[i])}for i,data in enumerate(episodes)]
    bound_contract['observed_arm_projected_ranges_by_run_and_phase']={d['run']:{phase:{name:values for name,values in joints.items()if name in ARM_BOUNDS}for phase,joints in d['phase_ranges_by_joint'].items()}for d in distribution}
    bound_contract['prior_v6_bound_failure_inspection_sha256']='e62017aab4b24cb69d9425def7553e5d95dbf722ed9233065b57bdee3e8b6791'
    sources=source_hashes();reader.require(sources==initial_sources,'Inspection source changed while running')
    inspection={'schema':'five_development_histories_residual_structure_inspection_v7','training_started':False,
        'source_sha256':sources,'physics_family_sha256':family,'frozen_baseline_checkpoint_sha256':baseline_hash,
        'bound_inspection':bound_report,'candidate_bound_contract':bound_contract,'compatible_with_declared_v7_candidate_bounds':compatible,'distributions':distribution,
        'classification_contract':zero_policy.CLASSIFICATION_CONTRACT,
        'definition':'All deltas use original labels without clipping. 1e-6 is a training classification-confidence boundary, never an inference deadband. Non-bitwise-equal values within it are ambiguous and retain continuous regression labels. Active reachability remains strictly zero.'}
    qualification={'schema':'v7_five_history_bc_integrity_candidate_sidecar_v1','training_authorized':False,
        'requires_root_review_before_training':True,'candidate_bc_transitions_passed_integrity':True,
        'eligible_under_declared_v7_candidate_bounds':compatible,'eligible_for_rl_episode_training':False,
        'pure_student_episode_success_claim':False,'independent_generalization_claim':False,'original_result_files_modified':False,
        'source_sha256':sources,'physics_family_sha256':family,'frozen_baseline_checkpoint_sha256':baseline_hash,
        'data_split':'All five full actual histories are development only; no holdout or independent generalization.',
        'current_candidate_gate':policy.GATE_CONTRACT,'candidate_bound_contract':bound_contract,'current_candidate_bounds_by_name':dict(zip(bound_report['joint_names'],bound_report['residual_bound_rad'])),
        'original_projection_source_sha256':episodes[0]['parameters']['source_sha256'],
        'original_gate_source_sha256':{n:episodes[0]['collection_source_sha256'][n]for n in ('environment.py','joint_safety.py','wrist_audit.py','elbow_gate.py','palm_handover_audit.py','tilt_audit.py','layer_turn_metrics.py')},
        'zero_class_definition_approved':True,'classification_contract':zero_policy.CLASSIFICATION_CONTRACT,
        'zero_statistics_policy':'Strict bitwise zero is class zero; absolute difference >1e-6 is certain correction; remaining non-bitwise equal elements are ambiguous and excluded from BCE only. All original continuous labels remain unchanged. No inference magnitude deadband.',
        'classification_by_dataset':{d['run']:d['reviewed_classification']for d in distribution},
        'active_max_unreachable_distance_rad':bound_report['active_max_unreachable_distance_rad'],'datasets':[]}
    for data,supp in zip(episodes,supplements):
        qualification['datasets'].append({'evidence':data['evidence'],'consumer_schema':data['consumer_schema'],'supplemental_audit':supp,
          'original_result_flags':{k:data['result'].get(k)for k in ('eligible_training_episode','runtime_bridge_validated','pure_student_episode_claim','corrective_labels_qualified_for_training','original_wrist_gate_qualified','failure','collection_mode')},
          'eligible_transition_steps':np.flatnonzero(data['mask']).tolist(),'excluded_records':data['excluded_labels'],
          'teacher_steps':sum(bool(ok)and s=='teacher'for ok,s in zip(data['mask'],data['executed_sources'])),
          'student_steps':sum(bool(ok)and s=='student'for ok,s in zip(data['mask'],data['executed_sources']))})
    for name,value in [('inspection.json',inspection),('qualification_candidate.json',qualification)]:
        path=a.output/name;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(name,sha(path),flush=True)
    print('V7_BOUNDS',json.dumps({'eligible_steps':bound_report['eligible_steps'],'active_max_unreachable_distance_rad':bound_report['active_max_unreachable_distance_rad'],'masked_max_floor':bound_report['masked_original_label_floor_max_rad'],'compatible':compatible}),flush=True)

if __name__=='__main__':main()
