"""Inspect new actual-state feedback coverage; never grants training permission."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
import trusted_eval_dataset_registry_v1 as reader
import train_zero_gated_residual_v7 as trainer
import train_phase_gated_residual_v6 as prior
import inspect_residual_dataset_v7 as old_inspection
import zero_gated_residual_policy_v7 as policy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(data,model):
    x,previous,labels,mask=trainer.frozen.tensor_episode(data,'cpu')
    with torch.no_grad():
        base_proposal=model.baseline_targets(x,previous)
        base=policy.project_left_targets(base_proposal,previous,data['parameters'],data['left_indices'],data['projector'])
        hold=policy.project_left_targets(previous,previous,data['parameters'],data['left_indices'],data['projector'])
        active=model.residual_active(x)&mask
    exact_base=(labels.contiguous().view(torch.int32)==base.contiguous().view(torch.int32))[0].numpy()
    exact_hold=(labels.contiguous().view(torch.int32)==hold.contiguous().view(torch.int32))[0].numpy()
    delta=(labels-base).numpy()[0];previous_delta=(labels-hold).numpy()[0];active=active[0].numpy()
    names=[data['joint_names'][i]for i in data['left_indices']]
    phases=np.asarray([policy.PUBLIC_GATE_CONTRACT and trainer.PHASES[int(np.argmax(row[185:197]))]for row in data['x']])
    by_joint={}
    for i,name in enumerate(names):
        bad=active&(np.abs(delta[:,i])>float(model.residual_bound[i]))
        ids=np.flatnonzero(bad)
        by_joint[name]={'expert_minus_base_projected_min':float(delta[:,i][active].min()),
            'expert_minus_base_projected_max':float(delta[:,i][active].max()),
            'expert_minus_exact_hold_projected_min':float(previous_delta[:,i][active].min()),
            'expert_minus_exact_hold_projected_max':float(previous_delta[:,i][active].max()),
            'exact_base_count':int((exact_base[:,i]&active).sum()),'exact_hold_count':int((exact_hold[:,i]&active).sum()),
            'both_exact_count':int((exact_base[:,i]&exact_hold[:,i]&active).sum()),
            'outside_base_bound_frames':len(ids),'first_outside_step':int(ids[0])if len(ids)else None,
            'last_outside_step':int(ids[-1])if len(ids)else None,
            'outside_base_bound_exact_hold_frames':int((bad&exact_hold[:,i]).sum()),
            'outside_phase_counts':{str(phase):int((bad&(phases==phase)).sum())for phase in np.unique(phases[bad])}}
    return {'active_frames':int(active.sum()),'per_joint':by_joint,
        'classification_proposal_only':'BASE/HOLD strict original float32 projected equality; no class definition approved or training started',
        'labels_modified':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    torch.set_num_threads(1);start=sha(reader.__file__)
    base_path=a.run/'checkpoints/frozen_v4_baseline.pt';baseline,base_sha=trainer.frozen.load_baseline(base_path)
    data=reader.load_run(a.run,baseline,base_sha,trainer.FAMILY_SHA)
    identity=reader.read_json(a.run/'evaluation_protocol.json')
    saved=torch.load(a.run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
    model=reader.restore_recorded_model(a.run.resolve(),identity,saved)
    bounds=prior.inspect([data],model)
    detail=old_inspection.distributions(data,model,old_inspection.labels_diagnostic(a.run,data))
    report={'schema':'v7_actual_failed_prefix_bounds_and_hold_inspection_v1','training_started':False,'training_qualification_granted':False,
        'registry_reader_sha256':start,'frozen_predicate_engine_sha256':reader.ENGINE_SHA,
        'episode_evidence':data['evidence'],'original_result_flags':{k:data['result'].get(k)for k in trainer.RESULT_FLAGS},
        'eligible_transition_steps':np.flatnonzero(data['mask']).tolist(),'excluded_records':data['excluded_labels'],
        'audit':data['audit'],'bounds_inspection':bounds,'detailed_current_bound_failures':detail['unreachable_active_joint_details'],
        'projected_exact_hold_inspection':summary(data,model),'inspection_source_sha256':sha(__file__)}
    if sha(reader.__file__)!=start:raise ValueError('Registry changed during inspection')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('NEW_ACTUAL_PREFIX_INSPECTED',sha(a.output),json.dumps({'frames':len(data['x']),'eligible':int(data['mask'].sum()),
        'active_gap':bounds['active_max_unreachable_distance_rad'],'failed_joints':list(report['detailed_current_bound_failures'])}),flush=True)

if __name__=='__main__':main()
