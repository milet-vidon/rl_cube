"""Independent completed-product CPU replay; no optimizer, simulator or intervention."""
from pathlib import Path
import argparse,json,time
import torch
import train_hold_increment_v8 as t


def errors_by_actual_history(episodes,model):
    errors=[];summaries=[]
    with torch.no_grad():
        for data in episodes:
            x,previous,labels,mask=t.prior.frozen.tensor_episode(data,'cpu');hidden=None;count=0
            names=[data['joint_names'][i]for i in data['left_indices']]
            for step in range(x.shape[1]):
                observation=x[:,step:step+1];prev=previous[:,step:step+1];label=labels[:,step:step+1]
                proposed,hidden,_,diag=model.forward_with_diagnostics(observation,prev,hidden)
                project=lambda v:t.policy.project_left_targets(v,prev,data['parameters'],data['left_indices'],data['projector'])
                base,hold,actual=project(diag['baseline_proposal']),project(prev),project(proposed)
                classes=t.policy.classify_branch_labels(label,base,hold)
                active=bool(mask[0,step])&diag['public_phase_active']&classes['classification_mask']
                wrong=active&(classes['class_targets']!=diag['predicted_class'])
                for j in torch.nonzero(wrong[0,0]).flatten().tolist():
                    errors.append({'episode_id':data['episode_id'],'step':step,'joint_name':names[j],
                        'phase':t.prior.PHASES[int(observation[0,0,185:197].argmax())],
                        'public_progress':float(observation[0,0,197]),
                        'expert_class':int(classes['class_targets'][0,0,j]),'predicted_class':int(diag['predicted_class'][0,0,j]),
                        'logits':diag['mode_logits'][0,0,j].tolist(),'expert_projected_target':float(label[0,0,j]),
                        'baseline_projected_target':float(base[0,0,j]),'hold_projected_target':float(hold[0,0,j]),
                        'actual_projected_target':float(actual[0,0,j]),
                        'absolute_projected_error_rad':float((actual.double()-label.double()).abs()[0,0,j])})
                    count+=1
            summaries.append({'episode_id':data['episode_id'],'actual_frames_replayed':int(x.shape[1]),'class_errors':count})
    return errors,summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','qualification','authorization','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('checkpoint','qualification','authorization'):p.add_argument('--'+name+'-sha256',required=True)
    a=p.parse_args();t.require(not a.output.exists(),'New audit output required')
    torch.set_num_threads(1);started=time.monotonic();root=Path.cwd();sources=t.source_hashes()
    directories=[root/'runs'/n for n in t.RUN_NAMES];baseline_path=root/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt'
    model,metadata=t.load_checkpoint(a.checkpoint,baseline_path,directories,expected_physics_family=t.FAMILY_SHA,
        expected_checkpoint_sha256=a.checkpoint_sha256,qualification_path=a.qualification,
        expected_qualification_sha256=a.qualification_sha256,authorization_path=a.authorization,
        expected_authorization_sha256=a.authorization_sha256,require_offline_qualified=False)
    baseline,bh=t.prior.frozen.load_baseline(baseline_path);episodes,_=t.load_histories(directories,baseline,bh)
    final=t.assess(episodes,model,'cpu',streaming=True)
    t.require(final==metadata['final_cpu_sequential_metrics'],'Independent full CPU metrics differ from completed checkpoint')
    mistakes,counts=errors_by_actual_history(episodes,model)
    t.require(len(mistakes)==final['classification_error_count'],'Detailed class-error inventory does not sum to full replay')
    initial=metadata['initial_cpu_sequential_metrics']
    recomputed=(final['step499_regression_violation_count']==final['step545_regression_violation_count']==0 and
        final['classification_error_count']<initial['classification_error_count'] and
        final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'])
    t.require(recomputed==metadata['offline_candidate_qualified'],'Completed qualification flag differs from actual CPU metrics')
    t.require(sources==t.source_hashes(),'Sources changed during independent audit')
    t.dump(a.output,{'schema':'completed_v8_eight_history_cpu_product_replay_v1','checkpoint_sha256':a.checkpoint_sha256,
        'qualification_sha256':a.qualification_sha256,'authorization_sha256':a.authorization_sha256,
        'source_sha256':sources,'audit_source_sha256':t.sha(__file__),'elapsed_s':time.monotonic()-started,
        'best_epoch':metadata['best_epoch'],'offline_candidate_qualified':recomputed,
        'completed_metadata_metrics_reproduced_exactly':True,'final_cpu_sequential_metrics':final,
        'class_order':['BASE','HOLD','UPDATE'],'class_error_records':mistakes,'per_history_counts':counts,
        'eligible_frames':sum(int(d['mask'].sum())for d in episodes),
        'optimizer_created':False,'model_modified':False,'closed_loop_success_claim':False,'independent_generalization_claim':False,
        'scope':'Eight previously reviewed actual development histories, complete causal GRU history from reset; physical evaluation remains required.'})
    print('V8_COMPLETED_PRODUCT_AUDIT',json.dumps({'report':str(a.output),'report_sha256':t.sha(a.output),
        'qualified':recomputed,'selection_key':t.selection_key(final),'best_epoch':metadata['best_epoch']}),flush=True)


if __name__=='__main__':main()
