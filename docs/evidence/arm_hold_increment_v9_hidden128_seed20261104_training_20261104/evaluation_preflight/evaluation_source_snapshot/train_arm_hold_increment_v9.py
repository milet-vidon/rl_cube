"""Nine-history reviewed development BC for BASE/HOLD/UPDATE; no simulator.

Default execution creates a qualification candidate and authorization template.
Optimization requires separately reviewed, exact candidate and authorization
bytes. Original failures, labels and episode-ineligibility flags are preserved.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,math,random,time
import numpy as np
import torch
import train_zero_gated_residual_v7 as prior
import inspect_residual_dataset_v7 as old_inspection
import trusted_eval_dataset_registry_v2 as registry
import train_hold_increment_v8 as v8
import inspect_arm_hold_increment_branches_v9 as coverage
import arm_hold_increment_policy_v9 as policy

AUTH_SCHEMA='root_reviewed_v9_nine_history_development_bc_authorization_v1'
QUAL_SCHEMA='v9_nine_history_arm_previous_increment_bc_candidate_v1'
BASELINE_SHA=prior.BASELINE_SHA;FAMILY_SHA=prior.FAMILY_SHA
RUN_NAMES=v8.RUN_NAMES+coverage.NEW_RUNS
ZERO_BOUNDARY_RUNS=v8.ZERO_BOUNDARY_RUNS
HOLD_BOUNDARY_RUNS=v8.HOLD_BOUNDARY_RUNS
HOLD_BOUNDARY_JOINTS=v8.HOLD_BOUNDARY_JOINTS
INDEX_HOLD_RUNS=coverage.NEW_RUNS
INDEX_HOLD_JOINTS=('left_index_MCP_joint','left_index_PIP_joint','left_index_DIP_joint')
SELECTION_CONTRACT={'ordering':['reviewed_step499_history_violations','reviewed_step545_projected_HOLD_joint_violations',
    'reviewed_step560_index_projected_HOLD_joint_violations','all_three_class_confusion_errors','hard_decoded_projected_rmse_rad'],
    'zero_step499_histories':list(ZERO_BOUNDARY_RUNS),'hold_step545_histories':list(HOLD_BOUNDARY_RUNS),
    'hold_step545_joint_names':list(HOLD_BOUNDARY_JOINTS),
    'index_hold_step560_histories':list(INDEX_HOLD_RUNS),'index_hold_step560_joint_names':list(INDEX_HOLD_JOINTS),'other_histories_not_relabelled':True,
    'initial_untrained_epoch_is_candidate':False,'candidate_boundary_check':'CPU sequential actual prefix through step560',
    'final_metrics':'CPU sequential full actual histories','no_independent_validation_claim':True}
sha=prior.sha;require=prior.require;dump=prior.dump;bits_equal=prior.bits_equal;element_bits=prior.element_bits

def source_hashes():
    return v8.source_hashes()|{Path(m.__file__).name:sha(m.__file__)for m in (registry,coverage,policy)}|{Path(__file__).name:sha(__file__)}

def dataset_consumer(index):
    require(isinstance(index,int)and not isinstance(index,bool)and 0<=index<len(RUN_NAMES),'Unknown ordered dataset index')
    return prior.mixture_reader if index==0 else prior.reader if index<5 else v8.registry if index<8 else registry

def load_histories(directories,baseline,baseline_hash):
    require(tuple(Path(d).name for d in directories)==RUN_NAMES,'Exactly nine ordered reviewed histories are required')
    require(len({str(Path(d).resolve())for d in directories})==9,'Duplicate history path')
    episodes,supplements=v8.load_histories(directories[:8],baseline,baseline_hash)
    for directory in directories[8:]:
        data=registry.load_run(directory,baseline,baseline_hash,FAMILY_SHA);episodes.append(data);supplements.append(data['audit'])
    names,left=episodes[0]['joint_names'],episodes[0]['left_indices']
    require(all(d['joint_names']==names and d['left_indices']==left and d['physics_family_sha256']==FAMILY_SHA for d in episodes),'Mixed joint order or physics domain')
    return episodes,supplements

def build_model(baseline,episodes,hidden_size):
    for data in episodes:policy.verify_projection_arm_bounds(data['parameters'],data['left_indices'])
    values=np.concatenate([d['x'][d['mask']]for d in episodes]);names=[episodes[0]['joint_names'][i]for i in episodes[0]['left_indices']]
    return policy.HoldIncrementPolicy(baseline,values.mean(0),np.maximum(values.std(0),.001),policy.candidate_bounds(names),hidden_size,
                                     left_joint_names=names)

def dataset_entry(data,supplement):
    return {'evidence':data['evidence'],'original_result_flags':{k:data['result'].get(k)for k in prior.RESULT_FLAGS},
        'eligible_transition_steps':np.flatnonzero(data['mask']).tolist(),'excluded_records':data['excluded_labels'],'audit':supplement}

def verify_model_named_contract(model,episodes):
    names=[episodes[0]['joint_names'][i]for i in episodes[0]['left_indices']]
    expected=torch.tensor([n in policy.PREVIOUS_ANCHOR_JOINTS for n in names],dtype=torch.bool)
    require(tuple(model.left_joint_names)==tuple(names),'Model actual left-name order differs')
    require(bits_equal(model.update_anchor_is_previous,expected),'Previous-anchor mask differs from actual joint names')
    require(bits_equal(model.residual_bound,torch.tensor(policy.candidate_bounds(names),dtype=torch.float32)),'Named UPDATE bounds differ')
    for data in episodes:policy.verify_projection_arm_bounds(data['parameters'],data['left_indices'])

def make_qualification(episodes,supplements,model,coverage_path,coverage_sha):
    require(sha(coverage_path)==coverage_sha,'Reviewed coverage bytes changed')
    proof=json.loads(Path(coverage_path).read_text())
    require(proof['schema']=='nine_development_histories_arm_hold_increment_coverage_v9_candidate_v1','Wrong coverage schema')
    require(proof['training_started']is False and proof['training_qualification_granted']is False,'Coverage promoted itself')
    require(proof['architecture_contract']==coverage.CONTRACT,'Coverage action semantics differ')
    require(proof['source_sha256']==coverage.source_hashes(),'Coverage sources differ')
    entries=[dataset_entry(d,s)for d,s in zip(episodes,supplements)]
    require(len(episodes)==len(supplements)==len(proof['histories'])==len(proof['datasets'])==9,'Nine full actual-history coverage records required')
    require(proof['datasets']==entries,'Coverage data/audit/flags changed')
    require(proof['physics_family_sha256']==FAMILY_SHA and proof['frozen_baseline_checkpoint_sha256']==BASELINE_SHA,'Coverage domain changed')
    verify_model_named_contract(model,episodes)
    reports=[];classes=[];class_summaries=[]
    with torch.no_grad():
        for data,old_report in zip(episodes,proof['histories']):
            report=coverage.coverage(data,model)
            require(report==old_report,'Independent branch coverage does not reproduce')
            require(report['active_max_union_gap_rad']==0. and report['active_no_float32_branch_witness_elements']==0,'Missing exact branch execution witness')
            require(report['masked_corrections_gt_1e_6']==0,'Public gate would mask genuine correction')
            x,previous,labels,mask=prior.frozen.tensor_episode(data,'cpu')
            project=lambda y:policy.project_left_targets(y,previous,data['parameters'],data['left_indices'],data['projector'])
            base=project(model.baseline_targets(x,previous));hold=project(previous)
            c=policy.classify_branch_labels(labels,base,hold)
            arrays={'BASE':c['exact_base'],'HOLD':c['exact_hold'],'UPDATE':c['confident_update'],'ambiguous':c['ambiguous']}
            hashes={k:hashlib.sha256(v[0].numpy().tobytes()).hexdigest()for k,v in arrays.items()}
            require(hashes==report['classification_boolean_array_sha256'],'Root model classification differs from independent coverage')
            # BASE/HOLD labels are exact; remaining UPDATE and ambiguous labels
            # therefore require the UPDATE witness, not mere union proximity.
            active=model.residual_active(x)&mask
            counts={k:int((v&active.unsqueeze(-1)).sum())for k,v in arrays.items()}
            require(counts==report['active_classes'],'Classification counts differ')
            class_summaries.append({'episode_id':data['episode_id'],'boolean_array_sha256':hashes,'active_class_counts':counts})
            classes.append(c);reports.append(report)
    q={'schema':QUAL_SCHEMA,'training_authorized':False,'requires_root_review_before_training':True,
        'eligible_for_development_bc_transition_supervision':True,'eligible_for_rl_episode_training':False,
        'pure_student_episode_success_claim':False,'independent_generalization_claim':False,'original_result_files_modified':False,
        'source_sha256':source_hashes(),'physics_family_sha256':FAMILY_SHA,'frozen_baseline_checkpoint_sha256':BASELINE_SHA,
        'model_schema':policy.SCHEMA,'feature_schema':policy.FEATURE_SCHEMA,'public_gate_contract':policy.PUBLIC_GATE_CONTRACT,
        'action_contract':policy.ACTION_CONTRACT,'classification_contract':policy.CLASSIFICATION_CONTRACT,
        'selection_contract':SELECTION_CONTRACT,'coverage_sha256':coverage_sha,'coverage_architecture_contract':coverage.CONTRACT,
        'left_joint_names':list(model.left_joint_names),'update_bound_rad':model.residual_bound.tolist(),
        'normalization_mean':model.mean.tolist(),'normalization_std':model.std.tolist(),
        'original_arm_projection_proofs':[policy.verify_projection_arm_bounds(d['parameters'],d['left_indices'])for d in episodes],
        'update_anchor_is_previous':model.update_anchor_is_previous.tolist(),'datasets':entries,
        'dataset_consumers':[d['consumer_schema']for d in episodes],
        'classification':class_summaries,'branch_coverage':reports,'total_eligible_frames':sum(int(d['mask'].sum())for d in episodes),
        'all_original_continuous_labels_preserved':True,'data_split':'nine complete actual development histories; no holdout or independent generalization'}
    return q,classes

def validate_qualification(text,expected_sha,recomputed):
    require(hashlib.sha256(text.encode()).hexdigest()==expected_sha,'Qualification bytes SHA mismatch')
    require(json.loads(text)==recomputed,'Reviewed qualification differs from recomputed data/model/class contracts')

def training_config(args):
    return {'epochs':args.epochs,'learning_rate':args.learning_rate,'hidden_size':args.hidden_size,'seed':args.seed,
        'regression_weight_rad2':args.regression_weight,'cross_entropy_weight':args.ce_weight,'weight_decay':1e-5,
        'gradient_clip_norm':1.,'device':args.device,'candidate_interval':args.candidate_interval,
        'episode_shuffle_only':True,'gru_reset_each_episode':True,'regression':'all labels, label-selected original-projected branch',
        'ambiguous_excluded_only_from_classification':True,'last_state_artifact_saved':True,'resume_cli_implemented':False}

def authorization_template(qsha,config):
    return {'schema':AUTH_SCHEMA,'approved_for_development_bc_training':True,'qualification_sha256':qsha,
        'source_sha256':source_hashes(),'training_config':config,'selection_contract':SELECTION_CONTRACT,
        'eligible_for_rl_episode_training':False,'independent_generalization_claim':False}

def validate_authorization(text,digest,qsha,config):
    require(hashlib.sha256(text.encode()).hexdigest()==digest,'Authorization SHA changed')
    require(json.loads(text)==authorization_template(qsha,config),'Authorization does not cover this exact training definition')

def assess(episodes,model,device='cpu',*,streaming=False,boundary_only=False):
    model.eval();errors=[];selected_errors=[];confusion=torch.zeros(3,3,dtype=torch.int64);runs=[];zero_bad=hold_bad=index_hold_bad=0
    with torch.no_grad():
        for data in episodes:
            x,previous,labels,mask=prior.frozen.tensor_episode(data,device)
            if boundary_only:
                require(streaming,'Boundary check must stream actual prefix');x,previous,labels,mask=(v[:,:561]for v in (x,previous,labels,mask))
            if streaming:
                hidden=None;outputs=[];diags=[]
                for step in range(x.shape[1]):
                    y,hidden,_,diag=model.forward_with_diagnostics(x[:,step:step+1],previous[:,step:step+1],hidden)
                    outputs.append(y);diags.append(diag)
                proposed=torch.cat(outputs,1);diag={k:torch.cat([d[k]for d in diags],1)for k in diags[0]}
            else:proposed,_,_,diag=model.forward_with_diagnostics(x,previous,None)
            project=lambda value:policy.project_left_targets(value,previous,data['parameters'],data['left_indices'],data['projector'])
            actual=project(proposed);base=project(diag['baseline_proposal']);hold=project(previous);update=project(diag['update_proposal'])
            c=policy.classify_branch_labels(labels,base,hold);target=c['class_targets'];pred=diag['predicted_class']
            active=mask.unsqueeze(-1)&diag['public_phase_active'];classified=active&c['classification_mask']
            for a in range(3):
                for b in range(3):confusion[a,b]+=int(((target==a)&(pred==b)&classified).sum())
            selected=torch.where(target==policy.BASE,base,torch.where(target==policy.HOLD,hold,update))
            selected=torch.where(diag['public_phase_active'],selected,base)
            errors.append((actual.double()-labels.double())[mask].cpu().numpy())
            selected_errors.append((selected.double()-labels.double())[mask].cpu().numpy())
            name=Path(data['evidence']['directory']).name
            require(x.shape[1]>545 and bool(mask[0,499]),'Required actual prefix absent')
            boundary=prior.boundary_record(499,proposed[0,499],diag['baseline_proposal'][0,499],previous[0,499],actual[0,499],base[0,499],labels[0,499],data)
            required499=name in ZERO_BOUNDARY_RUNS
            if required499:
                require(boundary['expert_projected_all28_bitwise_baseline'],'Reviewed 499 label is not exact BASE')
                zero_bad+=int(not all(boundary[k]for k in ('proposal_all28_bitwise_baseline','native_action_all28_bitwise_baseline','projected_all28_bitwise_baseline')))
            names=[data['joint_names'][i]for i in data['left_indices']];indices=[names.index(n)for n in HOLD_BOUNDARY_JOINTS]
            required545=name in HOLD_BOUNDARY_RUNS
            exact545=element_bits(actual[0,545,indices],hold[0,545,indices])
            if required545:
                require(bool(mask[0,545]),'Reviewed 545 HOLD frame is excluded')
                require(bool(element_bits(labels[0,545,indices],hold[0,545,indices]).all()),'Reviewed 545 expert is not exact HOLD')
                require(bool((~element_bits(labels[0,545,indices],base[0,545,indices])).all()),'Reviewed 545 HOLD overlaps BASE')
                hold_bad+=int((~exact545).sum())
            index_indices=[names.index(n)for n in INDEX_HOLD_JOINTS]
            required560=name in INDEX_HOLD_RUNS
            require(x.shape[1]>560,'Reviewed actual prefix through index recovery absent')
            exact560=element_bits(actual[0,560,index_indices],hold[0,560,index_indices])
            if required560:
                require(bool(mask[0,560]),'Reviewed 560 index HOLD frame is excluded')
                require(bool(element_bits(labels[0,560,index_indices],hold[0,560,index_indices]).all()),'Reviewed 560 expert is not exact HOLD')
                require(bool((~element_bits(labels[0,560,index_indices],base[0,560,index_indices])).all()),'Reviewed 560 HOLD overlaps BASE')
                index_hold_bad+=int((~exact560).sum())
            first=None;changed=(~element_bits(proposed,diag['baseline_proposal']).all(-1))&mask
            ids=torch.nonzero(changed[0]).flatten()
            if len(ids):
                k=int(ids[0]);first={'step':k,'phase':prior.PHASES[int(x[0,k,185:197].argmax())],
                    'predicted_classes':pred[0,k].tolist(),'expert_classes':target[0,k].tolist(),
                    'classification_mask':c['classification_mask'][0,k].tolist(),
                    'actual_projected_target':actual[0,k].cpu().tolist(),'expert_projected_target':labels[0,k].cpu().tolist()}
            runs.append({'episode_id':data['episode_id'],'frames_processed':int(x.shape[1]),
                'step499_zero_regression_required':required499,'step499':boundary,
                'step545_hold_regression_required':required545,'step545':{'joint_names':list(HOLD_BOUNDARY_JOINTS),
                    'predicted_classes':pred[0,545,indices].tolist(),'expert_classes':target[0,545,indices].tolist(),
                    'projection_exact_HOLD':exact545.tolist(),'expert_exact_HOLD':element_bits(labels[0,545,indices],hold[0,545,indices]).tolist()},
                'step560_index_hold_regression_required':required560,'step560':{'joint_names':list(INDEX_HOLD_JOINTS),
                    'predicted_classes':pred[0,560,index_indices].tolist(),'expert_classes':target[0,560,index_indices].tolist(),
                    'projection_exact_HOLD':exact560.tolist(),'expert_exact_HOLD':element_bits(labels[0,560,index_indices],hold[0,560,index_indices]).tolist()},
                'first_nonbaseline_proposal':first})
    def fit(values):
        value=np.concatenate(values);return {'rmse_rad':float(np.sqrt(np.mean(value**2))),'max_abs_rad':float(np.max(np.abs(value))),
            'rmse_by_joint_rad':np.sqrt(np.mean(value**2,axis=0)).tolist()}
    require(sum(r['step499_zero_regression_required']for r in runs)==7,'All seven zero histories required')
    require(sum(r['step545_hold_regression_required']for r in runs)==2,'Both reviewed HOLD histories required')
    require(sum(r['step560_index_hold_regression_required']for r in runs)==1,'Reviewed index recovery history required')
    return {'hard_decoded_projected_fit':fit(errors),'label_selected_projected_fit':fit(selected_errors),
        'class_order':['BASE','HOLD','UPDATE'],'confusion_matrix_true_by_predicted':confusion.tolist(),
        'classification_error_count':int(confusion.sum()-confusion.trace()),
        'step499_regression_violation_count':zero_bad,'step545_regression_violation_count':hold_bad,'step560_regression_violation_count':index_hold_bad,
        'step499_reviewed_zero_histories_bitwise_baseline':zero_bad==0,'step545_reviewed_hold_histories_exact_projected_hold':hold_bad==0,
        'step560_reviewed_index_history_exact_projected_hold':index_hold_bad==0,
        'runs':runs,'inference':'sequential_one_frame_hidden_state'if streaming else'complete_causal_history_batch',
        'actual_recorded_history':True,'teacher_labels_used_as_actor_input':False,'closed_loop_or_generalization_success_claim':False}

def selection_key(metrics):
    return (metrics['step499_regression_violation_count'],metrics['step545_regression_violation_count'],
            metrics['step560_regression_violation_count'],metrics['classification_error_count'],metrics['hard_decoded_projected_fit']['rmse_rad'])


def verify_device_definition(episodes,model,device):
    """Forward only: exact CPU/execution branch definitions must survive CUDA."""
    started=time.monotonic();other=deepcopy(model).to(device).eval();reports=[]
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for data in episodes:
            projections=[];classes=[];bases=[]
            for candidate,where in ((model,'cpu'),(other,device)):
                x,previous,labels,_=prior.frozen.tensor_episode(data,where)
                base=candidate.baseline_targets(x,previous)
                project=lambda v:policy.project_left_targets(v,previous,data['parameters'],data['left_indices'],data['projector'])
                bp,hp=project(base),project(previous)
                bases.append(base.cpu());projections.append((bp.cpu(),hp.cpu()))
                classes.append(policy.classify_branch_labels(labels,bp,hp))
            diffs={k:int((classes[0][k].cpu()!=classes[1][k].cpu()).sum()) for k in
                ('class_targets','classification_mask','exact_base','exact_hold','confident_update','ambiguous')}
            row={'episode_id':data['episode_id'],'baseline_proposal_different_bits':int((~element_bits(*bases)).sum()),
                'baseline_projected_different_bits':int((~element_bits(projections[0][0],projections[1][0])).sum()),
                'hold_projected_different_bits':int((~element_bits(projections[0][1],projections[1][1])).sum()),
                'classification_differences':diffs}
            require(not any(diffs.values()) and not any(row[k] for k in
                ('baseline_proposal_different_bits','baseline_projected_different_bits','hold_projected_different_bits')),
                'CPU/training-device branch definitions differ: '+str(row))
            reports.append(row)
    memory={'allocated':torch.cuda.max_memory_allocated(device),'reserved':torch.cuda.max_memory_reserved(device)} if str(device).startswith('cuda') else None
    del other
    return {'device':str(device),'elapsed_s':time.monotonic()-started,'cuda_peak_bytes':memory,
        'optimizer_created':False,'training_started':False,'all_branch_and_class_arrays_bitwise_identical':True,'runs':reports}


def checkpoint_metadata(model,episodes,q,qtext,qsha,authtext,authsha,config):
    return {'schema':policy.SCHEMA,'feature_schema':policy.FEATURE_SCHEMA,'physics_family_sha256':FAMILY_SHA,
        'frozen_baseline_checkpoint_sha256':BASELINE_SHA,'joint_names':episodes[0]['joint_names'],
        'output_indices':episodes[0]['left_indices'],'left_joint_names':list(model.left_joint_names),
        'source_sha256':source_hashes(),'dataset_evidence':[d['evidence']for d in episodes],
        'dataset_consumers':[d['consumer_schema']for d in episodes],
        'qualification_text':qtext,'qualification_sha256':qsha,'root_authorization_text':authtext,
        'root_authorization_sha256':authsha,'training_config':config,'selection_contract':SELECTION_CONTRACT,
        'gate_contract':policy.PUBLIC_GATE_CONTRACT,'action_contract':policy.ACTION_CONTRACT,
        'classification_contract':policy.CLASSIFICATION_CONTRACT,'update_bound_rad':model.residual_bound.tolist(),
        'residual_bound_rad':model.residual_bound.tolist(),'bound_semantics':'UPDATE increment about named anchor; not total distance from BASE',
        'update_anchor_is_previous':model.update_anchor_is_previous.tolist(),'hidden_size':model.hidden_size,
        'normalization_mean':model.mean.tolist(),'normalization_std':model.std.tolist(),
        'original_arm_projection_proofs':deepcopy(q['original_arm_projection_proofs']),
        'actor_inputs_are_privileged_actual_state_not_vision':True,'eligible_for_rl_episode_training':False,
        'independent_generalization_demonstrated':False,'closed_loop_success_demonstrated':False,'right28_learned':False}


def atomic_torch_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    torch.save(value,temporary);temporary.replace(path)


def save_candidate(path,model,metadata,epoch,optimizer_steps,metrics,streamed_boundary):
    require(epoch>=1 and optimizer_steps>=1,'Untrained baseline cannot be saved as a learned candidate')
    require(metrics['inference']==streamed_boundary['inference']=='sequential_one_frame_hidden_state','Candidate requires CPU sequential metrics')
    atomic_torch_save({'model':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
        'metadata':metadata|{'best_epoch':epoch,'optimizer_steps':optimizer_steps,'selection_metrics':metrics,
            'cpu_streamed_boundary':streamed_boundary,'offline_candidate_qualified':False}},path)


def verify_metadata(metadata,baseline,baseline_hash,directories,expected_family):
    require(metadata['schema']==policy.SCHEMA and metadata['feature_schema']==policy.FEATURE_SCHEMA,'V9 checkpoint schema differs')
    require(metadata['source_sha256']==source_hashes(),'V9 training/reader/inference source changed')
    require(metadata['physics_family_sha256']==expected_family==FAMILY_SHA,'Wrong physical family')
    require(metadata['frozen_baseline_checkpoint_sha256']==baseline_hash==BASELINE_SHA,'Wrong frozen baseline')
    for key in ('joint_names','output_indices'):
        require(metadata[key]==baseline['metadata'][key],'Checkpoint joint order changed: '+key)
    for key,value in (('gate_contract',policy.PUBLIC_GATE_CONTRACT),('action_contract',policy.ACTION_CONTRACT),
                      ('classification_contract',policy.CLASSIFICATION_CONTRACT),('selection_contract',SELECTION_CONTRACT)):
        require(metadata[key]==value,'Checkpoint contract changed: '+key)
    require(metadata['best_epoch']>=1 and metadata['optimizer_steps']>=1,'Untrained checkpoint')
    require(metadata['hidden_size']==metadata['training_config']['hidden_size'],'Hidden size differs from authorization')
    require(metadata['best_epoch']<=metadata['training_config']['epochs'] and
            metadata['optimizer_steps']==len(RUN_NAMES)*metadata['best_epoch'],'Checkpoint training epoch/update counters differ')
    qtext=metadata['qualification_text'];qsha=metadata['qualification_sha256'];q=json.loads(qtext)
    require(hashlib.sha256(qtext.encode()).hexdigest()==qsha,'Embedded qualification altered')
    require(q['schema']==QUAL_SCHEMA and q['training_authorized']is False and q['source_sha256']==source_hashes(),'Original candidate qualification changed')
    for key in ('eligible_for_rl_episode_training','pure_student_episode_success_claim','independent_generalization_claim','original_result_files_modified'):
        require(q[key]is False,'Original qualification False flag changed: '+key)
    require(q['eligible_for_development_bc_transition_supervision']is True,'Candidate transition scope missing')
    for key,value in (('physics_family_sha256',FAMILY_SHA),('frozen_baseline_checkpoint_sha256',BASELINE_SHA),
                      ('public_gate_contract',policy.PUBLIC_GATE_CONTRACT),('action_contract',policy.ACTION_CONTRACT),
                      ('classification_contract',policy.CLASSIFICATION_CONTRACT),('selection_contract',SELECTION_CONTRACT)):
        require(q[key]==value,'Embedded qualification differs: '+key)
    validate_authorization(metadata['root_authorization_text'],metadata['root_authorization_sha256'],qsha,metadata['training_config'])
    require(tuple(Path(d).name for d in directories)==RUN_NAMES and len({str(Path(d).resolve())for d in directories})==9,'Exactly nine distinct ordered histories required')
    require(len(metadata['dataset_evidence'])==len(metadata['dataset_consumers'])==len(q['datasets'])==9,'Nine evidence records required')
    require(metadata['dataset_consumers']==q['dataset_consumers'],'Checkpoint data consumer routing changed')
    for i,(directory,old,qd)in enumerate(zip(directories,metadata['dataset_evidence'],q['datasets'])):
        require(old==qd['evidence'],'Checkpoint qualification dataset differs')
        consumer=dataset_consumer(i)
        consumer.verify_saved_datasets({'dataset_evidence':[old],'physics_family_sha256':FAMILY_SHA},[directory])
    projection_proofs=[policy.verify_projection_arm_bounds(json.loads((Path(d)/'projection_parameters.json').read_text()),metadata['output_indices'])for d in directories]
    require(metadata['original_arm_projection_proofs']==q['original_arm_projection_proofs']==projection_proofs,'Original per-history arm projection cap changed')
    names=[metadata['joint_names'][i]for i in metadata['output_indices']]
    require(metadata['left_joint_names']==q['left_joint_names']==names,'Actual left joint order differs')
    expected_bounds=torch.tensor(policy.candidate_bounds(names),dtype=torch.float32)
    expected_anchor=torch.tensor([n in policy.PREVIOUS_ANCHOR_JOINTS for n in names],dtype=torch.bool)
    for key in ('update_bound_rad','residual_bound_rad'):
        require(bits_equal(expected_bounds,torch.tensor(metadata[key],dtype=torch.float32)),'Named increment bounds differ')
    require(bits_equal(expected_bounds,torch.tensor(q['update_bound_rad'],dtype=torch.float32)),'Qualified increment bounds differ')
    require(bits_equal(expected_anchor,torch.tensor(metadata['update_anchor_is_previous'],dtype=torch.bool)) and
            bits_equal(expected_anchor,torch.tensor(q['update_anchor_is_previous'],dtype=torch.bool)),'Named previous-anchor mask differs')
    for key in ('normalization_mean','normalization_std'):
        require(bits_equal(torch.tensor(metadata[key],dtype=torch.float32),torch.tensor(q[key],dtype=torch.float32)),'Qualified normalization differs')
    for key in ('actor_inputs_are_privileged_actual_state_not_vision',):require(metadata[key]is True,'Actor scope missing')
    for key in ('eligible_for_rl_episode_training','independent_generalization_demonstrated','closed_loop_success_demonstrated','right28_learned'):
        require(metadata[key]is False,'Checkpoint scope promoted: '+key)


def load_checkpoint(path,baseline_path,dataset_directories,*,expected_physics_family,expected_checkpoint_sha256,
                    qualification_path,expected_qualification_sha256,authorization_path,expected_authorization_sha256,
                    require_offline_qualified=True):
    require(sha(path)==expected_checkpoint_sha256,'V9 checkpoint bytes SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True);metadata=saved['metadata'];state=saved['model']
    for file,digest,text_key,hash_key in (
        (qualification_path,expected_qualification_sha256,'qualification_text','qualification_sha256'),
        (authorization_path,expected_authorization_sha256,'root_authorization_text','root_authorization_sha256')):
        raw=Path(file).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==digest==metadata[hash_key] and raw.decode('utf-8')==metadata[text_key],
                'External reviewed evidence differs: '+hash_key)
    baseline,basehash=prior.frozen.load_baseline(baseline_path)
    verify_metadata(metadata,baseline,basehash,dataset_directories,expected_physics_family)
    if require_offline_qualified:
        require(metadata['offline_candidate_qualified']is True,'Candidate failed offline qualification')
        final=metadata['final_cpu_sequential_metrics']
        require(final['inference']=='sequential_one_frame_hidden_state' and
            final['step499_reviewed_zero_histories_bitwise_baseline']is True and
            final['step545_reviewed_hold_histories_exact_projected_hold']is True and
            final['step560_reviewed_index_history_exact_projected_hold']is True and
            final['step499_regression_violation_count']==final['step545_regression_violation_count']==final['step560_regression_violation_count']==0,
            'Candidate missing final CPU development regressions')
        initial=metadata['initial_cpu_sequential_metrics']
        require(final['classification_error_count']<initial['classification_error_count'] and
            final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'],
            'Qualified candidate does not improve decoded development metrics')
    model=policy.HoldIncrementPolicy(baseline,metadata['normalization_mean'],metadata['normalization_std'],
        metadata['update_bound_rad'],metadata['hidden_size'],metadata['left_joint_names'],metadata['gate_contract'],metadata['action_contract'])
    original=model.state_dict()
    # All nontrainable tensors, including the entire frozen baseline, are exact.
    trainable={name for name,p in model.named_parameters()if p.requires_grad}
    require(set(state)==set(original),'Checkpoint state keys differ')
    for key,value in original.items():
        if key not in trainable:require(bits_equal(value,state[key]),'Checkpoint static tensor differs bitwise: '+key)
        require(state[key].shape==value.shape and state[key].dtype==value.dtype,'Checkpoint tensor definition differs: '+key)
        require(bool(torch.isfinite(state[key]).all()),'Nonfinite checkpoint tensor: '+key)
    model.load_state_dict(state,strict=True)
    return model.eval(),metadata


def save_last_state(path,model,optimizer,rng,epoch,steps,metadata,best):
    atomic_torch_save({'schema':'v9_last_optimizer_state_not_deployment_candidate_v1','epoch':epoch,'optimizer_steps':steps,
        'model':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
        'episode_shuffle_random_state':rng.getstate(),'torch_cpu_rng_state':torch.get_rng_state(),
        'torch_cuda_rng_states':torch.cuda.get_rng_state_all()if torch.cuda.is_available()else[],
        'numpy_rng_state':{'algorithm':np.random.get_state()[0],'keys':torch.tensor(np.random.get_state()[1].astype(np.int64)),
            'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},
        'metadata':metadata,'best_selection_key':list(best)if best is not None else None,
        'resume_cli_implemented':False,'exact_resume_demonstrated':False},path)


def train(model,episodes,classes,args,metadata):
    initial=assess(episodes,model,'cpu',streaming=True)
    dump(args.output/'initial_untrained_baseline_metrics.json',initial)
    device=args.device;model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=args.learning_rate,weight_decay=1e-5)
    rng=random.Random(args.seed);best=None;steps=0;started=time.monotonic()
    with (args.output/'training.jsonl').open('w',encoding='utf-8',newline='\n')as journal:
        for epoch in range(1,args.epochs+1):
            model.train();order=list(range(len(episodes)));rng.shuffle(order);loss_rows=[]
            for i in order:
                data=episodes[i];x,previous,labels,mask=prior.frozen.tensor_episode(data,device)
                optimizer.zero_grad(set_to_none=True)
                proposed,_,_,diag=model.forward_with_diagnostics(x,previous,None)
                losses=policy.supervision_losses(proposed,diag,previous,labels,mask,
                    classes[i]['class_targets'].to(device),classes[i]['classification_mask'].to(device),
                    data['parameters'],data['left_indices'],data['projector'])
                loss=args.regression_weight*losses['regression_mse_rad2']+args.ce_weight*losses['branch_cross_entropy']
                require(bool(torch.isfinite(loss)),'Nonfinite training objective')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();steps+=1
                loss_rows.append({'episode_id':data['episode_id'],'objective':float(loss.detach()),
                    **{k:float(losses[k].detach())for k in ('regression_mse_rad2','branch_cross_entropy')}})
            metrics=assess(episodes,model,device);selected=False;cpu_metrics=None
            consider=epoch==1 or epoch%args.candidate_interval==0 or epoch==args.epochs
            if consider:
                cpu_model=deepcopy(model).cpu()
                cpu_metrics=assess(episodes,cpu_model,'cpu',streaming=True);score=selection_key(cpu_metrics)
                if best is None or score<best:
                    best=score;selected=True
                    save_candidate(args.output/'model.pt',cpu_model,metadata,epoch,steps,cpu_metrics,cpu_metrics)
                del cpu_model
                save_last_state(args.output/'last_state.pt',model,optimizer,rng,epoch,steps,metadata,best)
            row={'epoch':epoch,'optimizer_steps':steps,'episode_losses':loss_rows,'metrics':metrics,
                'cpu_candidate_metrics':cpu_metrics,'selected':selected,'selection_key':list(selection_key(cpu_metrics))if cpu_metrics else None,
                'elapsed_s':time.monotonic()-started}
            journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
            if epoch==1 or epoch%10==0:print('V9_DEVELOPMENT_FIT',json.dumps({'epoch':epoch,'selected':selected,
                'decoded_rmse':metrics['hard_decoded_projected_fit']['rmse_rad'],'class_errors':metrics['classification_error_count'],
                'step499_violations':metrics['step499_regression_violation_count'],'step545_violations':metrics['step545_regression_violation_count'],
                'step560_violations':metrics['step560_regression_violation_count'],
                'elapsed_s':row['elapsed_s']}),flush=True)
    require(metadata['source_sha256']==source_hashes(),'Sources changed during optimization')
    saved=torch.load(args.output/'model.pt',map_location='cpu',weights_only=True);model.cpu().load_state_dict(saved['model'])
    final=assess(episodes,model,'cpu',streaming=True)
    require(final==saved['metadata']['selection_metrics'],'Saved candidate CPU sequential metrics do not reproduce')
    qualified=(final['step499_regression_violation_count']==final['step545_regression_violation_count']==final['step560_regression_violation_count']==0 and
        final['classification_error_count']<initial['classification_error_count'] and
        final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'])
    saved['metadata'].update(offline_candidate_qualified=qualified,final_cpu_sequential_metrics=final,
        initial_cpu_sequential_metrics=initial,qualification_basis='trained development candidate: seven step499 BASE and two step545 projected HOLD and one step560 index HOLD regressions; fewer class errors and lower decoded RMSE than initial; no physical/generalization claim')
    atomic_torch_save(saved,args.output/'model.pt')
    return {'offline_candidate_qualified':qualified,'final_cpu_sequential_metrics':final,'best_epoch':saved['metadata']['best_epoch'],
        'checkpoint_sha256':sha(args.output/'model.pt'),'last_state_sha256':sha(args.output/'last_state.pt'),
        'closed_loop_success_demonstrated':False,'independent_generalization_demonstrated':False,
        'failure_if_unqualified':None if qualified else'candidate_selection_contract_not_met'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train',action='store_true')
    p.add_argument('--coverage',type=Path,required=True);p.add_argument('--coverage-sha256',required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-sha256')
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256')
    p.add_argument('--epochs',type=int,default=5000);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--hidden-size',type=int,default=64);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--regression-weight',type=float,default=1e6);p.add_argument('--ce-weight',type=float,default=1.)
    p.add_argument('--candidate-interval',type=int,default=100);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    require(args.epochs>=1 and args.hidden_size>=1 and args.candidate_interval>=1 and all(math.isfinite(v)and v>0 for v in
        (args.learning_rate,args.regression_weight,args.ce_weight)),'Positive finite training parameters required')
    torch.set_num_threads(1);sources=source_hashes();baseline,basehash=prior.frozen.load_baseline(args.baseline)
    require(basehash==BASELINE_SHA,'Unreviewed frozen baseline')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    episodes,supplements=load_histories(args.run,baseline,basehash);model=build_model(baseline,episodes,args.hidden_size)
    q,classes=make_qualification(episodes,supplements,model,args.coverage,args.coverage_sha256)
    qtext=json.dumps(q,indent=2,allow_nan=False)+'\n';qsha=hashlib.sha256(qtext.encode()).hexdigest()
    if args.qualification is not None:
        require(bool(args.qualification_sha256),'Qualification requires expected SHA')
        text=args.qualification.read_bytes().decode('utf-8');validate_qualification(text,args.qualification_sha256,q)
        qtext,qsha=text,args.qualification_sha256
    config=training_config(args);authtext=authsha=None
    if args.train:
        require(args.qualification is not None and args.authorization is not None and args.authorization_sha256,
            'Optimization requires explicit reviewed qualification and root authorization files with pinned hashes')
        authtext=args.authorization.read_bytes().decode('utf-8');authsha=args.authorization_sha256
        validate_authorization(authtext,authsha,qsha,config)
    device_proof=verify_device_definition(episodes,model,args.device)
    initial=assess(episodes,model,'cpu',streaming=True,boundary_only=True)
    initial['untrained_reference_only_not_candidate']=True
    require(sources==source_hashes(),'Sources changed during qualification')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'qualification_candidate.json').write_text(qtext,encoding='utf-8',newline='\n')
    dump(args.output/'training_authorization_template.json',authorization_template(qsha,config))
    dump(args.output/'training_device_definition.json',device_proof);dump(args.output/'initial_boundary_metrics.json',initial)
    dump(args.output/'verified_inputs.json',{'source_sha256':sources,'qualification_sha256':qsha,'eligible_steps':q['total_eligible_frames'],
        'training_started':bool(args.train),'original_candidate_training_authorized':q['training_authorized'],'all_original_continuous_labels_retained':True})
    print('V9_INPUTS_VERIFIED',json.dumps({'eligible_steps':q['total_eligible_frames'],'qualification_sha256':qsha,
        'authorization_template':str(args.output/'training_authorization_template.json')}),flush=True)
    if args.train:
        metadata=checkpoint_metadata(model,episodes,q,qtext,qsha,authtext,authsha,config)
        result=train(model,episodes,classes,args,metadata);dump(args.output/'result.json',result)
        print('V9_COMPLETE',json.dumps({k:v for k,v in result.items()if k!='final_cpu_sequential_metrics'}),flush=True)


if __name__=='__main__':main()
