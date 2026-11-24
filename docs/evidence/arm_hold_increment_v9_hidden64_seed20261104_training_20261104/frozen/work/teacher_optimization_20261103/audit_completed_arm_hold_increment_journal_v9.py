"""Completed V9 journal/checkpoint/optimizer inventory; trainer-author self-check, no policy inference.

Valid RNG and optimizer artifacts do not demonstrate exact resumed training.
The frozen trainer intentionally exposes no resume CLI.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,math,random
import numpy as np
import torch
import train_arm_hold_increment_v9 as trainer


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v:(_ for _ in()).throw(ValueError('Nonfinite JSON: '+v)))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_recorded_shuffle(rng,ordered_episode_ids,loss_rows):
    order=list(range(len(ordered_episode_ids)));rng.shuffle(order)
    require([row['episode_id']for row in loss_rows]==[ordered_episode_ids[i]for i in order],
            'Epoch episode order differs from authorized private shuffle RNG')


def validate_last_state(last,model,config,metadata,best_key):
    require(last['schema']=='v9_last_optimizer_state_not_deployment_candidate_v1','Wrong last-state schema')
    require(last['epoch']==config['epochs'] and last['optimizer_steps']==9*config['epochs'],'Incomplete last-state optimizer epoch')
    require(last['resume_cli_implemented']is False and last['exact_resume_demonstrated']is False,'Unproven resume claim')
    require(last['best_selection_key']==list(best_key),'Last-state remembered best differs from journal')
    require(all(metadata[k]==v for k,v in last['metadata'].items()),'Last-state immutable metadata differs from candidate')
    expected=model.state_dict();state=last['model'];trainable=dict((n,p)for n,p in model.named_parameters()if p.requires_grad)
    require(set(state)==set(expected),'Last-state model keys differ')
    for name,value in state.items():
        require(value.shape==expected[name].shape and value.dtype==expected[name].dtype,'Last-state model tensor layout differs: '+name)
        require(bool(torch.isfinite(value).all()),'Nonfinite last-state tensor: '+name)
        if name not in trainable:
            require(trainer.bits_equal(value,expected[name]),'Last-state frozen tensor changed: '+name)
    optimizer=last['optimizer'];groups=optimizer['param_groups']
    require(len(groups)==1,'Unexpected optimizer groups')
    group=groups[0];ids=group['params']
    require(len(ids)==len(trainable) and len(set(ids))==len(ids),'Optimizer parameter count/order malformed')
    require(set(optimizer['state'])==set(ids),'Missing AdamW moment state')
    require(group['lr']==config['learning_rate'] and group['weight_decay']==config['weight_decay'],'Optimizer authorized learning rate/decay differs')
    require(tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['amsgrad']is False,'Optimizer defaults differ')
    for index,(name,parameter) in zip(ids,trainable.items()):
        item=optimizer['state'][index]
        require(set(item)=={'step','exp_avg','exp_avg_sq'},'Unexpected AdamW state fields')
        require(float(item['step'])==last['optimizer_steps'],'AdamW per-parameter steps incomplete: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            value=item[key]
            require(value.shape==parameter.shape and value.dtype==parameter.dtype and bool(torch.isfinite(value).all()),'Invalid optimizer moment: '+name+'/'+key)
        require(bool((item['exp_avg_sq']>=0).all()),'Negative AdamW second moment')
    # Validate restoration into private host generators; never alter global RNG.
    shuffle=random.Random();shuffle.setstate(last['episode_shuffle_random_state'])
    cpu_rng=last['torch_cpu_rng_state']
    require(cpu_rng.dtype==torch.uint8 and cpu_rng.ndim==1,'Invalid CPU RNG bytes')
    generator=torch.Generator(device='cpu');generator.set_state(cpu_rng)
    cuda_states=last['torch_cuda_rng_states']
    if config['device'].startswith('cuda'):
        require(len(cuda_states)>int(config['device'].split(':')[-1]),'Missing active-device CUDA RNG state')
    require(all(v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0 for v in cuda_states),'Invalid CUDA RNG state bytes')
    nr=last['numpy_rng_state'];keys=nr['keys']
    require(keys.dtype==torch.int64 and keys.ndim==1 and bool(((keys>=0)&(keys<2**32)).all()),'Invalid NumPy RNG keys')
    numpy_rng=np.random.RandomState();numpy_rng.set_state((nr['algorithm'],keys.numpy().astype(np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
    return {'epoch':last['epoch'],'optimizer_steps':last['optimizer_steps'],'adamw_trainable_parameter_tensors':len(ids),
        'all_model_tensor_shapes_dtypes_finite':True,'all_frozen_model_tensors_bitwise_unchanged':True,
        'adamw_moments_and_steps_validated':True,'private_host_rng_restoration_validated':True,
        'cuda_rng_byte_vectors_validated':True,'cuda_rng_restore_or_exact_resume_executed':False,
        'resume_cli_implemented':False,'exact_resume_demonstrated':False,
        'last_optimizer_state_is_separate_from_selected_deployment_checkpoint':True}


def validate_metrics(metrics,qualified_class_counts,episode_frames,inference):
    require(metrics['inference']==inference,'Metrics inference mode differs')
    require(metrics['class_order']==['BASE','HOLD','UPDATE'],'Class order changed')
    matrix=np.asarray(metrics['confusion_matrix_true_by_predicted'])
    require(matrix.shape==(3,3) and np.issubdtype(matrix.dtype,np.integer) and (matrix>=0).all(),'Invalid confusion matrix')
    require(matrix.sum(1).tolist()==qualified_class_counts,'Class support changed from qualified labels')
    require(int(matrix.sum()-np.trace(matrix))==metrics['classification_error_count'],'Confusion error count differs')
    require({r['episode_id']:r['frames_processed']for r in metrics['runs']}==episode_frames,'History omitted/shortened in metrics')
    require(len(metrics['runs'])==9,'Duplicated or missing history metrics')
    zero_bad=hold_bad=index_bad=0;zero_names=[];hold_names=[];index_names=[]
    for row in metrics['runs']:
        name=Path(row['episode_id'].split(':')[0]).name
        require(row['step499_zero_regression_required']==(name in trainer.ZERO_BOUNDARY_RUNS),'Wrong step499 scope')
        require(row['step545_hold_regression_required']==(name in trainer.HOLD_BOUNDARY_RUNS),'Wrong step545 scope')
        if row['step499_zero_regression_required']:
            zero_names.append(name);require(row['step499']['expert_projected_all28_bitwise_baseline']is True,'Reviewed BASE label changed')
            zero_bad+=int(not all(row['step499'][k]for k in ('proposal_all28_bitwise_baseline','native_action_all28_bitwise_baseline','projected_all28_bitwise_baseline')))
        if row['step545_hold_regression_required']:
            hold_names.append(name);h=row['step545']
            require(h['joint_names']==list(trainer.HOLD_BOUNDARY_JOINTS) and h['expert_exact_HOLD']==[True]*3,'Reviewed HOLD label changed')
            require(all(type(v)is bool for v in h['projection_exact_HOLD']) and len(h['projection_exact_HOLD'])==3,'Invalid HOLD boolean checks')
            hold_bad+=sum(not v for v in h['projection_exact_HOLD'])
        require(row['step560_index_hold_regression_required']==(name in trainer.INDEX_HOLD_RUNS),'Wrong step560 scope')
        if row['step560_index_hold_regression_required']:
            index_names.append(name);h=row['step560']
            require(h['joint_names']==list(trainer.INDEX_HOLD_JOINTS) and h['expert_exact_HOLD']==[True]*3,'Reviewed index HOLD label changed')
            require(all(type(v)is bool for v in h['projection_exact_HOLD']) and len(h['projection_exact_HOLD'])==3,'Invalid index HOLD boolean checks')
            index_bad+=sum(not v for v in h['projection_exact_HOLD'])
    require(sorted(index_names)==sorted(trainer.INDEX_HOLD_RUNS),'Missing reviewed index recovery boundary')
    require(index_bad==metrics['step560_regression_violation_count'],'Index HOLD boundary count differs')
    require(metrics['step560_reviewed_index_history_exact_projected_hold']==(index_bad==0),'Index HOLD pass flag differs')
    require(sorted(zero_names)==sorted(trainer.ZERO_BOUNDARY_RUNS) and sorted(hold_names)==sorted(trainer.HOLD_BOUNDARY_RUNS),'Missing reviewed development boundaries')
    require(zero_bad==metrics['step499_regression_violation_count'] and hold_bad==metrics['step545_regression_violation_count'],'Boundary counts differ')
    require(metrics['step499_reviewed_zero_histories_bitwise_baseline']==(zero_bad==0) and metrics['step545_reviewed_hold_histories_exact_projected_hold']==(hold_bad==0),'Boundary pass flags differ')
    for key in ('hard_decoded_projected_fit','label_selected_projected_fit'):
        values=metrics[key]
        require(all(math.isfinite(v) and v>=0 for v in [values['rmse_rad'],values['max_abs_rad'],*values['rmse_by_joint_rad']]),'Invalid projected fit metrics')
        require(len(values['rmse_by_joint_rad'])==28,'Wrong fitted output width')
    require(metrics['teacher_labels_used_as_actor_input']is False and metrics['closed_loop_or_generalization_success_claim']is False,'Metrics promoted scope')


def audit(run,authorization_path,output):
    require(not output.exists(),'Preserve existing audit')
    result=read(run/'result.json');checkpoint=run/'model.pt';last_path=run/'last_state.pt'
    require(sha(checkpoint)==result['checkpoint_sha256'] and sha(last_path)==result['last_state_sha256'],'Completed artifact digest differs from result')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);metadata=saved['metadata'];config=metadata['training_config']
    require(metadata['source_sha256']==trainer.source_hashes(),'Frozen source changed')
    require(len(trainer.RUN_NAMES)==9 and len(metadata['source_sha256'])==22,'Wrong V9 history/source graph size')
    require(metadata['hidden_size']==config['hidden_size'],'Authorized model width differs')
    authtext=authorization_path.read_text();authsha=sha(authorization_path)
    require(metadata['root_authorization_sha256']==authsha and metadata['root_authorization_text']==authtext,'Root authorization differs')
    trainer.validate_authorization(authtext,authsha,metadata['qualification_sha256'],config)
    q=read(run/'qualification_candidate.json')
    require(sha(run/'qualification_candidate.json')==metadata['qualification_sha256'],'Original qualification changed')
    require(q['training_authorized']is False and q['eligible_for_rl_episode_training']is False,'Candidate eligibility promoted')
    initial=read(run/'initial_untrained_baseline_metrics.json')
    require(initial==metadata['initial_cpu_sequential_metrics'],'Initial reference metrics changed')
    episode_frames={r['episode_id']:r['frames_processed']for r in initial['runs']}
    names_to_ids={Path(eid.split(':')[0]).name:eid for eid in episode_frames}
    require(set(names_to_ids)==set(trainer.RUN_NAMES),'Unreviewed history name')
    ordered_episode_ids=[names_to_ids[name]for name in trainer.RUN_NAMES]
    shuffle_rng=random.Random(config['seed'])
    require(sum(episode_frames.values())==q['total_eligible_frames']==9523,'Reviewed history count changed')
    counts=[sum(c['active_class_counts'][k]for c in q['classification'])for k in ('BASE','HOLD','UPDATE')]
    validate_metrics(initial,counts,episode_frames,'sequential_one_frame_hidden_state')
    epochs=config['epochs'];interval=config['candidate_interval'];selected=[];best=None;best_row=None;elapsed=-1.;count=0
    losses={k:[float('inf'),-float('inf')]for k in ('objective','regression_mse_rad2','branch_cross_entropy')}
    with (run/'training.jsonl').open()as stream:
        for line in stream:
            require(line.endswith('\n'),'Incomplete training JSONL row')
            row=json.loads(line,parse_constant=lambda v:(_ for _ in()).throw(ValueError('Nonfinite training row')))
            count+=1;require(row['epoch']==count and row['optimizer_steps']==9*count,'Missing/duplicated epoch or optimizer step')
            require(len(row['episode_losses'])==9 and {v['episode_id']for v in row['episode_losses']}==set(episode_frames),'Missing/duplicated optimized history')
            validate_recorded_shuffle(shuffle_rng,ordered_episode_ids,row['episode_losses'])
            for values in row['episode_losses']:
                for key,bounds in losses.items():
                    value=values[key];require(math.isfinite(value) and value>=0,'Invalid recorded loss')
                    bounds[0]=min(bounds[0],value);bounds[1]=max(bounds[1],value)
                # Regression is float64; CE is float32 before promotion to the
                # float64 sum, exactly as in the reviewed production source.
                expected=config['regression_weight_rad2']*values['regression_mse_rad2']+float(np.float32(config['cross_entropy_weight'])*np.float32(values['branch_cross_entropy']))
                require(math.isclose(expected,values['objective'],rel_tol=1e-10,abs_tol=1e-12),'Recorded loss differs from authorized objective')
            require(math.isfinite(row['elapsed_s']) and row['elapsed_s']>=elapsed,'Elapsed training clock regressed');elapsed=row['elapsed_s']
            validate_metrics(row['metrics'],counts,episode_frames,'complete_causal_history_batch')
            consider=count==1 or count%interval==0 or count==epochs
            require((row['cpu_candidate_metrics']is not None)==consider,'CPU candidate schedule changed')
            if consider:
                cpu=row['cpu_candidate_metrics'];validate_metrics(cpu,counts,episode_frames,'sequential_one_frame_hidden_state')
                key=trainer.selection_key(cpu);require(row['selection_key']==list(key),'Selection key changed')
                improves=best is None or key<best
                require(row['selected']==improves,'Candidate selection violates exact ordering')
                if improves:
                    best=key;best_row=row;selected.append({'epoch':count,'selection_key':list(key),'elapsed_s':elapsed})
            else:require(row['selected']is False and row['selection_key']is None,'Noncandidate epoch selected')
    require(count==epochs and best_row is not None,'Training incomplete')
    require(best_row['epoch']==metadata['best_epoch']==result['best_epoch'] and metadata['optimizer_steps']==9*metadata['best_epoch'],'Saved selected epoch differs')
    final=metadata['final_cpu_sequential_metrics']
    require(final==metadata['selection_metrics']==metadata['cpu_streamed_boundary']==best_row['cpu_candidate_metrics']==result['final_cpu_sequential_metrics'],'Saved final/selection/journal metrics differ')
    qualified=(final['step499_regression_violation_count']==final['step545_regression_violation_count']==final['step560_regression_violation_count']==0 and final['classification_error_count']<initial['classification_error_count'] and final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'])
    require(qualified==metadata['offline_candidate_qualified']==result['offline_candidate_qualified'],'Incorrect offline qualification')
    require(result['failure_if_unqualified']==(None if qualified else'candidate_selection_contract_not_met'),'Wrong unqualified outcome reason')
    require(result['closed_loop_success_demonstrated']is False and result['independent_generalization_demonstrated']is False,'Overclaimed physical performance')
    baseline_path=run.parent/'teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt'
    baseline,basehash=trainer.prior.frozen.load_baseline(baseline_path);require(basehash==trainer.BASELINE_SHA,'Frozen V4 bytes changed')
    model=trainer.policy.HoldIncrementPolicy(baseline,metadata['normalization_mean'],metadata['normalization_std'],metadata['update_bound_rad'],metadata['hidden_size'],metadata['left_joint_names'],metadata['gate_contract'],metadata['action_contract'])
    model.load_state_dict(saved['model'])
    last=torch.load(last_path,map_location='cpu',weights_only=True)
    require(last['episode_shuffle_random_state']==shuffle_rng.getstate(),'Saved private shuffle RNG does not reproduce recorded epoch order')
    last_validation=validate_last_state(last,model,config,metadata,best)
    last_validation['private_shuffle_state_matches_all_recorded_epoch_orders']=True
    report={'schema':'v9_completed_journal_selection_and_last_state_trainer_author_selfcheck_v1','audit_passed':True,
        'training_run':str(run),'checkpoint_sha256':sha(checkpoint),'last_state_sha256':sha(last_path),'result_sha256':sha(run/'result.json'),
        'training_journal_sha256':sha(run/'training.jsonl'),'source_sha256':trainer.source_hashes(),
        'authorization_sha256':authsha,'qualification_sha256':metadata['qualification_sha256'],'epochs':count,
        'optimizer_steps_per_epoch':9,'best_epoch':metadata['best_epoch'],'selected_candidates':selected,'loss_ranges':losses,
        'offline_candidate_qualified':qualified,'final_cpu_sequential_metrics':final,'last_state_artifact_validation':last_validation,
        'independent_policy_inference_repeated_here':False,'reviewer_role':'trainer_author_self_check_pending_cross_agent_review',
        'independent_cross_agent_review_performed_here':False,'audit_script_sha256':sha(__file__),
        'training_resume_performed':False,'exact_resume_demonstrated':False,'original_source_or_result_modified':False,
        'closed_loop_success_claim':False,'independent_generalization_claim':False}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'checkpoint_sha256':sha(checkpoint),
        'best_epoch':metadata['best_epoch'],'offline_candidate_qualified':qualified,'epochs':count}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);audit(args.run.resolve(),args.authorization.resolve(),args.output.resolve())


if __name__=='__main__':main()
