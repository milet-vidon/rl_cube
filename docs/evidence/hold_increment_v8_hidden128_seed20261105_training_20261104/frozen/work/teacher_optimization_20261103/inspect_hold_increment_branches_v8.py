"""Eight-history original-execution-space branch coverage; no optimization."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
import trusted_eval_dataset_registry_v1 as registry
import train_zero_gated_residual_v7 as trainer
import zero_gated_residual_policy_v7 as old_policy
import inspect_residual_dataset_v7 as old_inspection

NEW_RUNS=('left_zero_gated_residual_v7_eval_9501_20261104',
 'left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2',
 'left_zero_gated_residual_v7_long_eval_9501_20261104')
CONTRACT={'schema':'candidate_base_hold_previous_increment_recovery11_v8_coverage',
 'BASE':'frozen V4 proposal through original float32 action roundtrip and project_target',
 'HOLD':'actual previous applied target through original float32 action roundtrip and project_target',
 'UPDATE_recovery11':'actual previous applied target plus bounded 0.006 rad continuous delta',
 'UPDATE_other17':'frozen V4 proposal plus previous reviewed named base-residual bound',
 'priority':['strict_bitwise_BASE','strict_bitwise_HOLD','confident_UPDATE','ambiguous'],
 'confident_UPDATE':'float64 absolute distance from both projected BASE and projected HOLD strictly greater than 1e-6',
 'ambiguous':'neither branch exact and within 1e-6 of at least one branch; excluded from classification only',
 'all_continuous_expert_labels_retained':True,'inference_amplitude_deadband':None,
 'original_environment_projection_or_safety_gates_changed':False,'public_phase_gate_unchanged':True}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):return a.contiguous().view(torch.int32)==b.contiguous().view(torch.int32)

def coverage(data,model):
    x,previous,labels,mask=trainer.frozen.tensor_episode(data,'cpu')
    names=[data['joint_names'][i]for i in data['left_indices']]
    recovery=torch.tensor([name in old_policy.RECOVERY_JOINTS for name in names],dtype=torch.bool).reshape(1,1,28)
    with torch.no_grad():
        base_proposal=model.baseline_targets(x,previous)
        active=model.residual_active(x)&mask
        project=lambda target:old_policy.project_left_targets(target,previous,data['parameters'],data['left_indices'],data['projector'])
        base=project(base_proposal);hold=project(previous)
        anchor=torch.where(recovery,previous,base_proposal)
        bounds=torch.where(recovery,torch.full_like(anchor,.006),model.residual_bound.reshape(1,1,28))
        clamp=lambda target:torch.clamp(target,model.baseline.lower.float(),model.baseline.upper.float())
        low=project(clamp(anchor-bounds));high=project(clamp(anchor+bounds))
        update_gap=torch.maximum(torch.minimum(low,high)-labels,labels-torch.maximum(low,high)).clamp_min(0.)
        base_gap=(labels.double()-base.double()).abs();hold_gap=(labels.double()-hold.double()).abs()
        exact_base=same(labels,base);exact_hold=same(labels,hold)
        class_base=exact_base;class_hold=~exact_base&exact_hold
        class_update=~exact_base&~exact_hold&(base_gap>1e-6)&(hold_gap>1e-6)
        ambiguous=~class_base&~class_hold&~class_update
        union_gap=torch.minimum(torch.minimum(base_gap,hold_gap),update_gap.double())
        # A concrete bounded float32 increment witness is stronger than interval
        # inclusion. Neighbors are diagnosed, never substituted for raw labels.
        delta=(labels-anchor).float();witness=torch.zeros_like(exact_base)
        first_witness=torch.full_like(delta,float('nan'))
        candidates=[delta]
        lower_delta=upper_delta=delta
        for _ in range(2):
            lower_delta=torch.nextafter(lower_delta,torch.full_like(delta,float('-inf')))
            upper_delta=torch.nextafter(upper_delta,torch.full_like(delta,float('inf')))
            candidates.extend((lower_delta,upper_delta))
        for candidate in candidates:
            allowed=candidate.abs()<=bounds
            matched=same(project(clamp(anchor+candidate)),labels)&allowed
            first_witness=torch.where(matched&~witness,candidate,first_witness)
            witness|=matched
        union_exact_witness=exact_base|exact_hold|witness
    selected=mask.unsqueeze(-1).expand_as(labels);active_elements=active.unsqueeze(-1).expand_as(labels)
    masked=selected&~active_elements
    def count(t,which=active_elements):return int((t&which).sum())
    per_joint={}
    for j,name in enumerate(names):
        choose=active[0];bad=choose&(union_gap[0,:,j]>0)
        d=(labels-anchor)[0,:,j]
        per_joint[name]={'recovery_finger':bool(recovery[0,0,j]),'update_bound_rad':float(bounds[0,0,j]),
            'expert_minus_update_anchor_min_rad':float(d[choose].min()),'expert_minus_update_anchor_max_rad':float(d[choose].max()),
            'BASE':int(class_base[0,:,j][choose].sum()),'HOLD':int(class_hold[0,:,j][choose].sum()),
            'UPDATE':int(class_update[0,:,j][choose].sum()),'ambiguous':int(ambiguous[0,:,j][choose].sum()),
            'both_BASE_and_HOLD_exact':int((exact_base&exact_hold)[0,:,j][choose].sum()),
            'numerically_unreachable_count':int(bad.sum()),'max_union_gap_rad':float(union_gap[0,:,j][choose].max()),
            'no_float32_branch_witness_count':int((~union_exact_witness[0,:,j]&choose).sum())}
    return {'episode_id':data['episode_id'],'eligible_frames':int(mask.sum()),'active_frames':int(active.sum()),
        'active_classes':{'BASE':count(class_base),'HOLD':count(class_hold),'UPDATE':count(class_update),'ambiguous':count(ambiguous)},
        'active_elements':int(active_elements.sum()),'masked_max_base_difference_rad':float(base_gap[masked].max())if masked.any()else 0.,
        'masked_corrections_gt_1e_6':int((base_gap[masked]>1e-6).sum()),
        'active_max_union_gap_rad':float(union_gap[active_elements].max()),
        'active_unreachable_elements':int((union_gap[active_elements]>0).sum()),
        'active_no_float32_branch_witness_elements':int((~union_exact_witness&active_elements).sum()),
        'float32_witness_scope':'BASE/HOLD exact, or bounded float32 increment and its +/-1,+/-2 ULP neighbors through the original projector; not yet a proof for a particular neural parameterization',
        'per_joint':per_joint,
        'classification_boolean_array_sha256':{key:hashlib.sha256(value[0].numpy().tobytes()).hexdigest()for key,value in
            [('BASE',class_base),('HOLD',class_hold),('UPDATE',class_update),('ambiguous',ambiguous)]}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    torch.set_num_threads(1);root=Path.cwd();work=root.parent/'work/teacher_optimization_20261103'
    source_start={Path(m.__file__).name:sha(m.__file__)for m in (registry,trainer,old_policy)}|{Path(__file__).name:sha(__file__)}
    baseline,base_sha=trainer.frozen.load_baseline(root/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
    episodes,supplements=trainer.load_histories([root/'runs'/n for n in old_inspection.RUN_NAMES],baseline,base_sha)
    model=trainer.build_model(baseline,episodes,64)
    qpath=work/'v7_five_history_inspection_20261104_03/qualification_candidate.json'
    trainer.validate_qualification(qpath.read_text(),sha(qpath),episodes,supplements,model,base_sha)
    for name in NEW_RUNS:
        data=registry.load_run(root/'runs'/name,baseline,base_sha,trainer.FAMILY_SHA);episodes.append(data)
        print('NEW_PREFIX_ORIGINAL_PREDICATES_PASSED',name,len(data['x']),flush=True)
    reports=[coverage(d,model)for d in episodes]
    require=lambda condition,message:None if condition else (_ for _ in ()).throw(ValueError(message))
    source_end={Path(m.__file__).name:sha(m.__file__)for m in (registry,trainer,old_policy)}|{Path(__file__).name:sha(__file__)}
    require(source_start==source_end,'Sources changed during coverage inspection')
    report={'schema':'eight_development_histories_base_hold_increment_coverage_v8_candidate_v1',
        'training_started':False,'training_qualification_granted':False,'architecture_contract':CONTRACT,'source_sha256':source_end,
        'original_five_history_qualification_sha256':sha(qpath),'physics_family_sha256':trainer.FAMILY_SHA,
        'frozen_baseline_checkpoint_sha256':base_sha,'histories':reports,
        'total_eligible_frames':sum(r['eligible_frames']for r in reports),
        'max_active_union_gap_rad':max(r['active_max_union_gap_rad']for r in reports),
        'active_no_float32_branch_witness_elements':sum(r['active_no_float32_branch_witness_elements']for r in reports),
        'datasets':[{'evidence':d['evidence'],'original_result_flags':{k:d['result'].get(k)for k in trainer.RESULT_FLAGS},
                     'eligible_transition_steps':np.flatnonzero(d['mask']).tolist(),'excluded_records':d['excluded_labels'],
                     'audit':d.get('audit',supplements[0]if i==0 else None)}for i,d in enumerate(episodes)],
        'scope':'All original labels and failures preserved; same physics/seed9501 development. Union representability only, not neural fitting, executed safety or independent generalization.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BRANCH_COVERAGE_COMPLETE',sha(a.output),json.dumps({k:report[k]for k in ('total_eligible_frames','max_active_union_gap_rad','active_no_float32_branch_witness_elements')}),flush=True)

if __name__=='__main__':main()
