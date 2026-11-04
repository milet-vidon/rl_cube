"""Inspect/qualify V6 development BC transitions; training requires pinned review.

V3 mixed data and completed evaluation prefixes have separate verified readers.
No original result flag is changed. Every eligible original label enters loss,
including the explicitly reported float32 floor under the public phase mask.
"""
from pathlib import Path
import argparse,hashlib,json,random,time
import numpy as np
import torch
import phase_gated_residual_policy_v6 as policy
import train_phase_residual_v5 as frozen
import dagger_residual_dataset_v5 as v3_reader
import eval_prefix_dataset_v6 as eval_reader

V3_CONSUMER='qualified_v3_projected_mixture_reader_v5'
EVAL_CONSUMER='completed_diagnostic_eval_safe_prefix_v6'
QUALIFICATION_SCHEMA='v6_reviewable_mixed_development_bc_transition_qualification_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def source_hashes():
    return frozen.source_hashes()|{Path(module.__file__).name:sha(module.__file__)for module in(policy,eval_reader)}|{Path(__file__).name:sha(__file__)}

def load_episode(directory,baseline,baseline_hash,family=None):
    directory=Path(directory)
    is_eval=(directory/'evaluation_protocol.json').exists()
    reader=eval_reader if is_eval else v3_reader
    result=reader.load_run(directory,baseline,baseline_hash,family)
    result['consumer_schema']=EVAL_CONSUMER if is_eval else V3_CONSUMER
    result['evidence']={**result['evidence'],'consumer_schema':result['consumer_schema']}
    return result

def verify_saved_datasets(metadata,directories):
    expected=metadata['dataset_evidence']
    if len(expected)!=len(directories):raise ValueError('All V6 source datasets must be supplied')
    for prior,directory in zip(expected,directories):
        reader={V3_CONSUMER:v3_reader,EVAL_CONSUMER:eval_reader}.get(prior['consumer_schema'])
        if reader is None:raise ValueError('Unknown V6 dataset consumer schema')
        reader.verify_saved_datasets({'dataset_evidence':[prior],
            'physics_family_sha256':metadata['physics_family_sha256']},[directory])

def prove_boundary(episodes):
    proofs=[]
    for episode in episodes:
        directory=Path(episode['evidence']['directory'])
        protocol=read(directory/'runtime_protocol.json')
        if protocol['phase_counts']['squeeze']!=85:raise ValueError('Public source squeeze clock changed')
        if episode['collection_source_sha256']['opposition_teacher.py']!=policy.GATE_CONTRACT['original_teacher_source_sha256']:
            raise ValueError('Original feedback activation source changed')
        plans=[]
        for path in sorted((directory/'plans').glob('pickup_*.json')):
            data=read(path)
            if isinstance(data,list)and data and data[0][0]=='ready':
                counts={name:len(values)for name,values in data}
                if counts.get('squeeze')!=73:raise ValueError('Original squeeze target path length changed')
                plans.append({'file':str(path.relative_to(directory)),'sha256':sha(path),'target_counts':counts})
        if not plans:raise ValueError('Missing archived initial pickup path for activation proof')
        proofs.append({'run':str(directory),'runtime_protocol_sha256':sha(directory/'runtime_protocol.json'),
            'source_squeeze_clock_frames':85,'initial_pickup_plans':plans,
            'feedback_first_squeeze_target_index':72,'public_progress_threshold':policy.GATE_CONTRACT['squeeze_active_progress_float32']})
    return proofs

def build_model(baseline,episodes,hidden_size):
    eligible=np.concatenate([d['x'][d['mask']]for d in episodes])
    names=episodes[0]['joint_names'];left=episodes[0]['left_indices']
    for data in episodes:
        if data['joint_names']!=names or data['left_indices']!=left:raise ValueError('Dataset joint order differs')
    if baseline['metadata']['joint_names']!=names or baseline['metadata']['output_indices']!=left:
        raise ValueError('Baseline and actual dataset joint order differ')
    bounds=policy.candidate_bounds([names[i]for i in left])
    return policy.PhaseGatedResidualPolicy(baseline,eligible.mean(0),np.maximum(eligible.std(0),.001),bounds,hidden_size)

def inspect(episodes,model):
    differences=[];distances=[];active_parts=[];run_parts=[]
    with torch.no_grad():
        for episode in episodes:
            x,previous,labels,mask=frozen.tensor_episode(episode,'cpu')
            base=model.baseline_targets(x,previous)
            bound=model.effective_bounds(x)
            low=torch.clamp(base-bound,model.baseline.lower.float(),model.baseline.upper.float())
            high=torch.clamp(base+bound,model.baseline.lower.float(),model.baseline.upper.float())
            low=policy.project_left_targets(low,previous,episode['parameters'],episode['left_indices'],episode['projector'])
            high=policy.project_left_targets(high,previous,episode['parameters'],episode['left_indices'],episode['projector'])
            distance=torch.maximum((torch.minimum(low,high)-labels).clamp_min(0),(labels-torch.maximum(low,high)).clamp_min(0))
            active=model.residual_active(x)[mask].numpy()
            delta=(labels-base)[mask].numpy();gap=distance[mask].numpy()
            differences.append(delta);distances.append(gap);active_parts.append(active)
            run_parts.append({'episode_id':episode['episode_id'],'eligible_steps':int(mask.sum()),
                'masked_steps':int((~active).sum()),'active_steps':int(active.sum()),
                'masked_original_label_floor_max_rad':float(gap[~active].max())if (~active).any()else 0.,
                'masked_original_nonzero_label_elements':int(np.count_nonzero(gap[~active])),
                'masked_corrections_gt_1e_6':int(np.count_nonzero(gap[~active]>1e-6)),
                'active_unreachable_max_distance_rad':float(gap[active].max())if active.any()else 0.,
                'active_unreachable_elements_strict':int(np.count_nonzero(gap[active]))})
    delta=np.concatenate(differences);gap=np.concatenate(distances);active=np.concatenate(active_parts)
    active_gap=gap[active];masked_gap=gap[~active]
    return {'eligible_steps':len(delta),'eligible_student_executed_steps':sum(sum(bool(ok)and source=='student'for ok,source in zip(d['mask'],d['executed_sources']))for d in episodes),
        'joint_names':[episodes[0]['joint_names'][i]for i in episodes[0]['left_indices']],
        'residual_bound_rad':model.residual_bound.tolist(),'gate_contract':policy.GATE_CONTRACT,'runs':run_parts,
        'expert_projected_minus_frozen_base_min_rad':delta.min(0).tolist(),
        'expert_projected_minus_frozen_base_max_rad':delta.max(0).tolist(),
        'absolute_difference_quantiles':{str(q):np.quantile(abs(delta),q,axis=0).tolist()for q in (.5,.9,.95,.99,1.)},
        'active_unreachable_count_by_joint_strict':np.count_nonzero(active_gap,axis=0).tolist(),
        'active_max_unreachable_distance_rad_by_joint':active_gap.max(0).tolist(),
        'active_max_unreachable_distance_rad':float(active_gap.max()),
        'masked_original_label_floor_max_rad':float(masked_gap.max()),
        'masked_original_nonzero_label_elements':int(np.count_nonzero(masked_gap)),
        'masked_corrections_gt_1e_6':int(np.count_nonzero(masked_gap>1e-6)),
        'all_labels_retained_in_loss':True,'labels_clipped_or_misses_filtered':False}

def assert_reachable(report):
    if report['masked_corrections_gt_1e_6']:
        raise ValueError('Public mask hides genuine expert corrections; reject candidate')
    if report['active_max_unreachable_distance_rad']!=0.:
        raise ValueError('Active original projected labels are not strictly reachable under declared bounds')

def qualification(episodes,baseline_hash,model,report,boundary_proof):
    return {'schema':QUALIFICATION_SCHEMA,'eligible_for_bc_transition_supervision':True,
        'eligible_for_rl_episode_training':False,'pure_student_episode_success_claim':False,
        'independent_generalization_claim':False,'requires_root_review_before_training':True,
        'original_result_files_modified':False,'physics_family_sha256':episodes[0]['physics_family_sha256'],
        'frozen_baseline_checkpoint_sha256':baseline_hash,'source_sha256':source_hashes(),
        'model_schema':policy.SCHEMA,'feature_schema':policy.FEATURE_SCHEMA,'gate_contract':policy.GATE_CONTRACT,
        'boundary_proof':boundary_proof,'residual_bound_joint_names':report['joint_names'],
        'residual_bound_rad':model.residual_bound.tolist(),'bound_and_mask_inspection':report,
        'data_split':'both complete histories are development fitting; no holdout or independent generalization claim',
        'masked_label_floor_policy':'Retain every original label in loss; explicitly allow only reported float32 floor <=1e-6 under masked output. This does not change any runtime safety gate.',
        'datasets':[{'evidence':d['evidence'],'consumer_schema':d['consumer_schema'],
            'collection_source_sha256':d['collection_source_sha256'],
            'original_result_flags':{k:d['result'].get(k)for k in ('eligible_training_episode','runtime_bridge_validated',
                'pure_student_episode_claim','corrective_labels_qualified_for_training','original_wrist_gate_qualified','failure','collection_mode')},
            'eligible_transition_steps':np.flatnonzero(d['mask']).tolist(),'excluded_records':d['excluded_labels'],
            'eligible_student_steps':sum(bool(ok)and s=='student'for ok,s in zip(d['mask'],d['executed_sources'])),
            'eligible_teacher_steps':sum(bool(ok)and s=='teacher'for ok,s in zip(d['mask'],d['executed_sources']))}for d in episodes]}

def load_checkpoint(path,baseline_path,dataset_directories,*,expected_physics_family):
    saved=torch.load(path,map_location='cpu',weights_only=True);metadata=saved['metadata'];state=saved['model']
    if metadata['schema']!=policy.SCHEMA or metadata['feature_schema']!=policy.FEATURE_SCHEMA:
        raise ValueError('V6 checkpoint schema mismatch')
    if metadata['gate_contract']!=policy.GATE_CONTRACT:raise ValueError('Checkpoint phase gate changed')
    if metadata['source_sha256']!=source_hashes():raise ValueError('V6 source hashes changed')
    if metadata['physics_family_sha256']!=expected_physics_family:raise ValueError('Different physics domain')
    baseline,basehash=frozen.load_baseline(baseline_path)
    if metadata['frozen_baseline_checkpoint_sha256']!=basehash:raise ValueError('Frozen baseline hash differs')
    if metadata['joint_names']!=baseline['metadata']['joint_names']or metadata['output_indices']!=baseline['metadata']['output_indices']:
        raise ValueError('Checkpoint joint names or ordered output indices differ')
    expected_bounds=torch.tensor(policy.candidate_bounds([metadata['joint_names'][i]for i in metadata['output_indices']]),dtype=torch.float32)
    if not torch.equal(torch.tensor(metadata['residual_bound_rad']),expected_bounds):raise ValueError('V6 declared joint bounds changed')
    verify_saved_datasets(metadata,dataset_directories)
    for name,value in baseline['model'].items():
        if not torch.equal(value,state['baseline.'+name]):raise ValueError('Frozen baseline weights changed')
    model=policy.PhaseGatedResidualPolicy(baseline,state['mean'],state['std'],metadata['residual_bound_rad'],metadata['hidden_size'],metadata['gate_contract'])
    for key in ('residual_bound','residual_phase_active','squeeze_active_progress'):
        if not torch.equal(model.state_dict()[key],state[key]):raise ValueError('Gate/bound state differs from metadata: '+key)
    model.load_state_dict(state)
    if any(not torch.isfinite(value).all()for value in model.state_dict().values()):raise ValueError('Nonfinite V6 weights')
    return model.eval(),metadata

def train(model,episodes,args,metadata):
    model.to(args.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=args.learning_rate,weight_decay=1e-5)
    rng=random.Random(args.seed);best=float('inf');started=time.monotonic()
    with (args.output/'training.jsonl').open('w')as journal:
        for epoch in range(args.epochs):
            model.train();order=list(range(len(episodes)));rng.shuffle(order);losses=[]
            for index in order:
                data=episodes[index];x,previous,labels,mask=frozen.tensor_episode(data,args.device)
                optimizer.zero_grad(set_to_none=True)
                proposed,_,_=model(x,previous,None)
                actual=policy.project_left_targets(proposed,previous,data['parameters'],data['left_indices'],data['projector'])
                loss=(((actual-labels)[mask]/.001)**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],1.)
                optimizer.step();losses.append(float(loss.detach()))
            assessed=frozen.metrics(episodes,model,args.device);score=assessed['expert_projected_fit']['rmse_rad']
            entry={'epoch':epoch+1,'loss_mrad_squared':float(np.mean(losses)),
                'selection_scope':'two_development_histories_fit_no_holdout','metrics':assessed,'elapsed_s':time.monotonic()-started}
            journal.write(json.dumps(entry,allow_nan=False)+'\n');journal.flush()
            if score<best:
                best=score;torch.save({'model':{k:v.detach().cpu()for k,v in model.state_dict().items()},
                    'metadata':metadata|{'best_epoch':epoch+1,'selection_scope':entry['selection_scope']}},args.output/'model.pt')
            if (epoch+1)%10==0 or epoch==0:print('V6_DEVELOPMENT_FIT',json.dumps(entry),flush=True)
    saved=torch.load(args.output/'model.pt',map_location='cpu',weights_only=True);model.cpu().load_state_dict(saved['model'])
    return {'training_fit_cpu_original_projection':frozen.metrics(episodes,model),'checkpoint_sha256':sha(args.output/'model.pt'),
        'best_epoch':saved['metadata']['best_epoch'],'independent_generalization_demonstrated':False,'closed_loop_success_demonstrated':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train',action='store_true')
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-sha256')
    p.add_argument('--epochs',type=int,default=2000);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--hidden-size',type=int,default=64);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.epochs<1 or args.learning_rate<=0:p.error('Positive training parameters required')
    if len(args.run)!=2 or len({str(d.resolve())for d in args.run})!=2:p.error('V6 candidate requires exactly two distinct development histories')
    initial_sources=source_hashes()
    baseline,basehash=frozen.load_baseline(args.baseline);episodes=[];family=None
    for path in args.run:
        data=load_episode(path,baseline,basehash,family);episodes.append(data);family=data['physics_family_sha256']
    if {d['consumer_schema']for d in episodes}!={V3_CONSUMER,EVAL_CONSUMER}:
        raise ValueError('V6 candidate requires one qualified v3 mixture and one independently verified evaluation prefix')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    boundary=prove_boundary(episodes);model=build_model(baseline,episodes,args.hidden_size)
    bound_report=inspect(episodes,model);assert_reachable(bound_report)
    q=qualification(episodes,basehash,model,bound_report,boundary)
    if q['source_sha256']!=initial_sources:raise ValueError('Source changed during inspection; rerun on frozen files')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'scope':'two_development_history_inspection_only','physics_family_sha256':family,
        'gate_and_bound_inspection':bound_report,'boundary_proof':boundary,
        'source_sha256':source_hashes(),'frozen_baseline_checkpoint_sha256':basehash,'training_started':False,
        'normalization_uses_only_these_two_development_histories':True,'no_labels_clipped_or_filtered_for_bound_fit':True}
    (args.output/'inspection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'qualification.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
    if not args.train:print('V6_INSPECTION_ONLY',args.output,flush=True);return
    frozen.require_reviewed_qualification(args.qualification,args.qualification_sha256,q)
    metadata={'schema':policy.SCHEMA,'feature_schema':policy.FEATURE_SCHEMA,'hidden_size':args.hidden_size,
        'gate_contract':policy.GATE_CONTRACT,'residual_bound_rad':model.residual_bound.tolist(),
        'joint_names':episodes[0]['joint_names'],'output_indices':episodes[0]['left_indices'],
        'physics_family_sha256':family,'frozen_baseline_checkpoint_sha256':basehash,'source_sha256':source_hashes(),
        'dataset_evidence':[d['evidence']for d in episodes],'seed':args.seed,
        'reviewed_qualification_sha256':args.qualification_sha256,'eligible_for_rl_episode_training':False,
        'teacher_hidden_state_or_labels_in_actor':False,'previous_executed_source_in_actor':False,
        'phase_source':'original_stage_controller_same_action_public_phase_progress',
        'reward_learning':False,'development_only':True,'independent_generalization_demonstrated':False,
        'normalization_scope':'two_development_histories','loss_weighting':'one full-history optimizer step per episode; mean over eligible step/joint labels; masked label floor retained'}
    result=train(model,episodes,args,metadata)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('V6_DEVELOPMENT_TRAINING_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
