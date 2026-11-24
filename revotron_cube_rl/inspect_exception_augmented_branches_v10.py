"""Eleven reviewed development histories, unchanged V9 execution representation.

Only completed audited transitions precede the two pinned teacher exceptions.
This tool performs no optimizer, simulator, existing-registry mutation or grant.
The separate V10 trainer must obtain its own reviewed qualification/authorization.
"""
from pathlib import Path
from types import ModuleType,MappingProxyType
import argparse
import hashlib
import json
import numpy as np
import torch
import train_arm_hold_increment_v9 as v9
import inspect_arm_hold_increment_branches_v9 as original
import exception_prefix_dataset_v1 as prefix

ROOT=prefix.ROOT
WORK=prefix.WORK
RUN_NAMES=v9.RUN_NAMES+tuple(prefix.RUNS)
SCHEMA='eleven_development_histories_unchanged_v9_branch_coverage_candidate_v1'
QUAL_SCHEMA='eleven_history_exception_augmented_unchanged_v9_data_candidate_v1'
CONTRACT=original.CONTRACT
coverage=original.coverage
require=v9.require
sha=prefix.sha
PINNED_PREFIX_SOURCE='3cf97ff2ee274b96091e27b8f8311f6c61f3feae0d5f83c6b003b6c6876bdeb2'
PRIOR_QUALIFICATION=('v9_trainer_preflight_20261104_01/qualification_candidate.json',
 '6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7')
PREFIX_PEER=('exception_prefix_independent_review_20261104.json',
 '56ea5f4423692ca5a6edadec06c3f0b325ad19d1a5e5a9b28afc83a64cc4bdb6')
PREFIX_REPORTS={
 tuple(prefix.RUNS)[0]:('exception_prefix_candidate_hidden64_seed05_20261104_01.json',
  '2e14372acd71dae84ae263e56393e91995625d8ba3b70fcde04f11725a749975'),
 tuple(prefix.RUNS)[1]:('exception_prefix_candidate_wide128_seed05_20261104_01.json',
  'a90afc2ee408d4fe58704eb2d92e9adb484f6addd37d6951a0186dcbfd6cb6aa')}


def source_hashes():
    require(sha(prefix.__file__)==PINNED_PREFIX_SOURCE,'Reviewed exception reader changed')
    return v9.source_hashes()|{Path(prefix.__file__).name:sha(prefix.__file__),Path(__file__).name:sha(__file__)}


def read_pinned(item):
    path=WORK/item[0]
    require(sha(path)==item[1],'Reviewed evidence changed: '+item[0])
    return json.loads(path.read_text())


def reviewed_inputs():
    prior=read_pinned(PRIOR_QUALIFICATION)
    read_pinned(PREFIX_PEER)
    require(prior['schema']==v9.QUAL_SCHEMA and prior['training_authorized']is False,'Original V9 qualification changed')
    require(prior['source_sha256']==v9.source_hashes(),'Frozen V9 source graph changed')
    require(prior['action_contract']==v9.policy.ACTION_CONTRACT and
            prior['classification_contract']==v9.policy.CLASSIFICATION_CONTRACT and
            prior['public_gate_contract']==v9.policy.PUBLIC_GATE_CONTRACT,'Original V9 policy semantics changed')
    reports={name:read_pinned(item)for name,item in PREFIX_REPORTS.items()}
    return prior,reports


def array_hashes(data):
    return {key:hashlib.sha256(np.asarray(data[key]).tobytes()).hexdigest()
            for key in ('x','previous_left','labels','mask')}


def run_name(data):
    return Path(data['evidence']['directory']).name


def validate_prefix_against_completed_report(data,report,name):
    require(name in PREFIX_REPORTS and run_name(data)==name and data['episode_id']==report['episode_id'],'Wrong exact candidate episode')
    count=prefix.RUNS[name]['steps']
    require(len(data['x'])==count and data['mask'].dtype==np.bool_ and bool(data['mask'].all()),'Prefix length or mask changed')
    require(report['selected_completed_candidate_transitions']==count and report['arrays_sha256']==array_hashes(data),
            'Completed prefix array bytes do not reproduce')
    for key in ('evidence','excluded_labels','audit','result','parameters','joint_names','left_indices',
                'collection_source_sha256','physics_family_sha256','consumer_schema'):
        require(data[key]==report[key],'Completed prefix evidence differs: '+key)
    for key in ('training_qualification_granted','eligible_for_rl_episode_training','current_V9_training_dataset_admission',
                'exception_query_after_state_known','exception_query_label_fabricated'):
        require(data[key]is False and report[key]is False,'Prefix promoted or fabricated a claim: '+key)
    require(len(data['excluded_labels'])==1 and data['excluded_labels'][0]['step']==count and
            data['excluded_labels'][0]['expert_label_present']is False,'Unreturned query exclusion changed')


def verify_saved_exception_evidence(old_evidence,directory):
    """Reverify exact completed proof and all current bytes, with no GRU forward.

    This is a checkpoint load check, not a new transition qualification. The
    pinned report remains the authority for the completed transition predicates.
    """
    source_hashes();_,reports=reviewed_inputs()
    run,spec=prefix.trust_run(directory);report=reports[run.name]
    require(old_evidence==report['evidence'],'Saved exception evidence differs from completed reviewed candidate')
    trust=prefix.private_trust_registry()
    module=ModuleType('_v10_exception_evidence_verifier')
    module.__dict__.update(__file__=prefix.prior.prior.frozen.__file__,
        REGISTRY=MappingProxyType({'v8':'exact_failed_exception_candidate'}),
        admitted_successful_wrist=lambda identity,result:False,check_unreturned_abort=prefix.check_unreturned_abort)
    source=prefix.engine_source()
    exec(compile(source,__file__+'::frozen_exception_evidence','exec'),module.__dict__)
    module.SCHEMA=prefix.SCHEMA
    module.verify_evaluation_semantics=trust.verify_evaluation_semantics
    module.restore_recorded_model=trust.restore_recorded_model
    current=module.verify_evidence(run)
    abort=prefix.scan_exact_boundary(run,spec['steps'])
    require(report['excluded_labels']==[abort],'Saved exception exclusion no longer reproduces')
    if 'actor_report'in spec:
        actor_proof=prefix.RecordedPredictionProof(run,spec,current).proof
    else:
        actor_proof={'mode':'fresh_complete_single_actual_history_replay','prior_completed_proof_available':False,
                     'new_gru_calls':spec['steps']}
    current.update(exception_prefix_contract_sha256=sha(prefix.__file__),original_registry2_sha256=prefix.PRIOR_SHA,
        private_engine_sha256=hashlib.sha256(source.encode()).hexdigest(),original_frozen_engine_sha256=prefix.prior.ENGINE_SHA,
        exact_run_whitelist=run.name,actor_proof=actor_proof,
        original_assets_sha256={str(ROOT/k):v for k,v in prefix.ASSETS.items()})
    require(current==old_evidence,'Complete current exception evidence differs')
    require(current['physics_family_sha256']==v9.FAMILY_SHA,'Exception physics family differs')
    prefix.trust_run(run);read_pinned(PREFIX_REPORTS[run.name])
    return current


def verify_history_contract(episodes):
    require(len(episodes)==11 and tuple(run_name(d)for d in episodes)==RUN_NAMES,'Exactly eleven ordered histories required')
    names,left=episodes[0]['joint_names'],episodes[0]['left_indices']
    for data in episodes:
        require(data['joint_names']==names and data['left_indices']==left and
                data['physics_family_sha256']==v9.FAMILY_SHA,'Mixed joint order or physics domain')
    require(sum(int(d['mask'].sum())for d in episodes)==13051,'Expected 9523 + 1725 + 1803 completed eligible transitions')


def load_histories(root):
    """Return the original nine plus two exact, independently reviewed prefixes."""
    root=Path(root)
    require(root.resolve()==ROOT/'revotron_cube_rl','Canonical CODE working directory required')
    _,reports=reviewed_inputs()
    baseline,base_sha=v9.prior.frozen.load_baseline(root/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
    require(base_sha==v9.BASELINE_SHA,'Frozen baseline bytes changed')
    episodes,supplements=v9.load_histories([root/'runs'/name for name in v9.RUN_NAMES],baseline,base_sha)
    for name in tuple(prefix.RUNS):
        data=prefix.load_candidate(root/'runs'/name,baseline,base_sha,v9.FAMILY_SHA)
        validate_prefix_against_completed_report(data,reports[name],name)
        episodes.append(data);supplements.append(data['audit'])
        print('EXACT_EXCEPTION_PREFIX_REPRODUCED',name,len(data['x']),flush=True)
    verify_history_contract(episodes)
    return baseline,base_sha,episodes,supplements


def classify_and_check(data,model,report):
    """Independently compare the frozen policy's labels with frozen coverage."""
    with torch.no_grad():
        x,previous,labels,mask=v9.prior.frozen.tensor_episode(data,'cpu')
        project=lambda value:v9.policy.project_left_targets(value,previous,data['parameters'],data['left_indices'],data['projector'])
        classes=v9.policy.classify_branch_labels(labels,project(model.baseline_targets(x,previous)),project(previous))
        arrays={key:classes[value]for key,value in
                (('BASE','exact_base'),('HOLD','exact_hold'),('UPDATE','confident_update'),('ambiguous','ambiguous'))}
        hashes={key:hashlib.sha256(value[0].numpy().tobytes()).hexdigest()for key,value in arrays.items()}
        active=model.residual_active(x)&mask
        counts={key:int((value&active.unsqueeze(-1)).sum())for key,value in arrays.items()}
        require(hashes==report['classification_boolean_array_sha256'],'Frozen coverage/policy class bits differ')
        require(counts==report['active_classes'],'Frozen coverage/policy class counts differ')
    return {'episode_id':data['episode_id'],'boolean_array_sha256':hashes,'active_class_counts':counts}


def representation_pass(reports):
    return all(r['active_max_union_gap_rad']==0. and r['active_unreachable_elements']==0 and
               r['active_no_float32_branch_witness_elements']==0 and r['masked_corrections_gt_1e_6']==0 for r in reports)


def build_reports(baseline,base_sha,episodes,supplements,initial_sources):
    prior,prefix_reports=reviewed_inputs()
    verify_history_contract(episodes)
    require(len(supplements)==11 and base_sha==v9.BASELINE_SHA,'History audit count or baseline changed')
    before=[array_hashes(d)for d in episodes]
    model=v9.build_model(baseline,episodes,64)
    v9.verify_model_named_contract(model,episodes)
    reports=[coverage(data,model)for data in episodes]
    require(reports[:9]==prior['branch_coverage'],'Original nine-history coverage changed')
    classification=[classify_and_check(d,model,r)for d,r in zip(episodes,reports)]
    entries=[v9.dataset_entry(d,s)for d,s in zip(episodes,supplements)]
    require(entries[:9]==prior['datasets'],'Original nine-history data, flags or audits changed')
    require(before==[array_hashes(d)for d in episodes],'Coverage changed original labels/features/history')
    require(initial_sources==source_hashes(),'Sources changed during eleven-history coverage')
    reviewed_inputs()
    for data in episodes[9:]:
        validate_prefix_against_completed_report(data,prefix_reports[run_name(data)],run_name(data))
    bindings={'prior_v9_qualification':dict(file=PRIOR_QUALIFICATION[0],sha256=PRIOR_QUALIFICATION[1]),
              'exception_prefix_independent_peer':dict(file=PREFIX_PEER[0],sha256=PREFIX_PEER[1]),
              'completed_exception_prefix_reports':{name:dict(file=item[0],sha256=item[1])for name,item in PREFIX_REPORTS.items()}}
    report={'schema':SCHEMA,'source_sha256':initial_sources,'architecture_contract':CONTRACT,
        'physics_family_sha256':v9.FAMILY_SHA,'frozen_baseline_checkpoint_sha256':base_sha,
        'training_started':False,'training_qualification_granted':False,'current_V9_training_dataset_admission':False,
        'reviewed_input_bindings':bindings,'histories':reports,'datasets':entries,'classification':classification,
        'total_eligible_frames':sum(r['eligible_frames']for r in reports),
        'max_active_union_gap_rad':max(r['active_max_union_gap_rad']for r in reports),
        'active_unreachable_elements':sum(r['active_unreachable_elements']for r in reports),
        'active_no_float32_branch_witness_elements':sum(r['active_no_float32_branch_witness_elements']for r in reports),
        'masked_corrections_gt_1e_6':sum(r['masked_corrections_gt_1e_6']for r in reports),
        'all_original_continuous_labels_retained':True,'arrays_sha256':dict(zip(RUN_NAMES,before)),
        'unchanged_original_nine_reports_and_evidence':True,'representation_checks_passed':representation_pass(reports),
        'scope':'Eleven same-physics seed9501 development histories. Exact execution branch witnesses only, not neural fit, safety or successful episode validation.'}
    candidate={'schema':QUAL_SCHEMA,'training_authorized':False,'training_qualification_granted':False,
        'requires_root_review_and_separate_trainer_qualification':True,'eligible_for_rl_episode_training':False,
        'pure_student_episode_success_claim':False,'independent_generalization_claim':False,'original_result_files_modified':False,
        'current_V9_training_dataset_admission':False,'source_sha256':initial_sources,'reviewed_input_bindings':bindings,
        'candidate_representation_checks_passed':representation_pass(reports),
        'physics_family_sha256':v9.FAMILY_SHA,'frozen_baseline_checkpoint_sha256':base_sha,
        'model_schema':v9.policy.SCHEMA,'feature_schema':v9.policy.FEATURE_SCHEMA,
        'public_gate_contract':v9.policy.PUBLIC_GATE_CONTRACT,'action_contract':v9.policy.ACTION_CONTRACT,
        'classification_contract':v9.policy.CLASSIFICATION_CONTRACT,'coverage_architecture_contract':CONTRACT,
        'left_joint_names':list(model.left_joint_names),'update_bound_rad':model.residual_bound.tolist(),
        'update_anchor_is_previous':model.update_anchor_is_previous.tolist(),
        'original_arm_projection_proofs':[v9.policy.verify_projection_arm_bounds(d['parameters'],d['left_indices'])for d in episodes],
        'datasets':entries,'dataset_consumers':[d['consumer_schema']for d in episodes],
        'classification':classification,'branch_coverage':reports,'arrays_sha256':report['arrays_sha256'],
        'total_eligible_frames':report['total_eligible_frames'],'all_original_continuous_labels_preserved':True,
        'unreturned_exception_queries_excluded':[d['excluded_labels']for d in episodes[9:]],
        'data_split':'Eleven actual development histories, two failed episode prefixes; no holdout or independent validation'}
    return report,candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output_directory.exists(),'Use a new output directory; preserve prior evidence')
    torch.set_num_threads(1)
    initial_sources=source_hashes()
    baseline,base_sha,episodes,supplements=load_histories(Path.cwd())
    report,candidate=build_reports(baseline,base_sha,episodes,supplements,initial_sources)
    args.output_directory.mkdir(parents=True,exist_ok=False)
    report_path=args.output_directory/'inspection.json'
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    candidate['coverage_sha256']=sha(report_path)
    candidate_path=args.output_directory/'qualification_candidate.json'
    candidate_path.write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'coverage_path':str(report_path),'coverage_sha256':sha(report_path),
        'qualification_candidate_path':str(candidate_path),'qualification_candidate_sha256':sha(candidate_path),
        'total_eligible_frames':report['total_eligible_frames'],'representation_checks_passed':report['representation_checks_passed'],
        'max_active_union_gap_rad':report['max_active_union_gap_rad'],
        'active_no_float32_branch_witness_elements':report['active_no_float32_branch_witness_elements'],
        'masked_corrections_gt_1e_6':report['masked_corrections_gt_1e_6'],'training_authorized':False},indent=2),flush=True)


if __name__=='__main__':main()
