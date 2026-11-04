"""Nine-history arm-prior increment candidate coverage; no optimization."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
import trusted_eval_dataset_registry_v2 as registry
import train_hold_increment_v8 as v8_trainer
import arm_hold_increment_policy_v9 as policy
import train_zero_gated_residual_v7 as trainer
import zero_gated_residual_policy_v7 as old_policy
import inspect_residual_dataset_v7 as old_inspection

NEW_RUNS=('left_hold_increment_v8_seed20261104_eval_9501_20261104',)
CONTRACT={'schema':'candidate_base_hold_arm7_recovery11_previous_increment_v9_coverage',
 'BASE':'frozen V4 proposal through original float32 action roundtrip and project_target',
 'HOLD':'actual previous applied target through original float32 action roundtrip and project_target',
 'UPDATE_recovery11':'actual previous applied target plus bounded 0.006 rad continuous delta',
 'UPDATE_arm7':'previous actual target plus original per-step min cap 0.025 rad; original projector unchanged',
 'UPDATE_other10':'frozen V4 proposal plus unchanged 1e-6 rad residual',
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
        policy.verify_projection_arm_bounds(data['parameters'],data['left_indices'])
        anchor=torch.where(model.update_anchor_is_previous.reshape(1,1,28),previous,base_proposal)
        bounds=model.residual_bound.reshape(1,1,28).expand_as(anchor)
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
        clipped_delta=torch.clamp(delta,-bounds,bounds)
        candidates=[delta,clipped_delta]
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
        per_joint[name]={'recovery_finger':bool(recovery[0,0,j]),'previous_target_anchor':bool(model.update_anchor_is_previous[j]),'update_bound_rad':float(bounds[0,0,j]),
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
        'float32_witness_scope':'BASE/HOLD exact, or original delta, cap-clipped delta, and +/-1,+/-2 ULP delta neighbors through original projector; no label edits and not yet proof of neural parameterization',
        'per_joint':per_joint,
        'classification_boolean_array_sha256':{key:hashlib.sha256(value[0].numpy().tobytes()).hexdigest()for key,value in
            [('BASE',class_base),('HOLD',class_hold),('UPDATE',class_update),('ambiguous',ambiguous)]}}

def source_hashes():
    return v8_trainer.source_hashes() | {Path(m.__file__).name:sha(m.__file__) for m in (registry,policy)} | {Path(__file__).name:sha(__file__)}


def load_histories(root):
    baseline,base_sha=trainer.frozen.load_baseline(root/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
    episodes,supplements=v8_trainer.load_histories([root/'runs'/n for n in v8_trainer.RUN_NAMES],baseline,base_sha)
    for name in NEW_RUNS:
        data=registry.load_run(root/'runs'/name,baseline,base_sha,trainer.FAMILY_SHA)
        episodes.append(data);supplements.append(data['audit'])
        print('NEW_PREFIX_ORIGINAL_PREDICATES_PASSED',name,len(data['x']),flush=True)
    return baseline,base_sha,episodes,supplements


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    torch.set_num_threads(1);root=Path.cwd();start=source_hashes()
    baseline,base_sha,episodes,supplements=load_histories(root)
    values=np.concatenate([d['x'][d['mask']] for d in episodes])
    names=[episodes[0]['joint_names'][i] for i in episodes[0]['left_indices']]
    model=policy.HoldIncrementPolicy(baseline,values.mean(0),np.maximum(values.std(0),.001),policy.candidate_bounds(names),64,left_joint_names=names)
    for data in episodes:policy.verify_projection_arm_bounds(data['parameters'],data['left_indices'])
    reports=[coverage(d,model)for d in episodes]
    if start!=source_hashes():raise ValueError('Sources changed during coverage inspection')
    report={'schema':'nine_development_histories_arm_hold_increment_coverage_v9_candidate_v1',
        'training_started':False,'training_qualification_granted':False,'architecture_contract':CONTRACT,'source_sha256':start,
        'physics_family_sha256':trainer.FAMILY_SHA,'frozen_baseline_checkpoint_sha256':base_sha,
        'histories':reports,'total_eligible_frames':sum(r['eligible_frames']for r in reports),
        'max_active_union_gap_rad':max(r['active_max_union_gap_rad']for r in reports),
        'active_no_float32_branch_witness_elements':sum(r['active_no_float32_branch_witness_elements']for r in reports),
        'datasets':[v8_trainer.dataset_entry(d,s)for d,s in zip(episodes,supplements)],
        'scope':'Original safe prefix labels and failures retained; same physics/seed9501 development. Union representation only, no optimization or closed-loop/generalization claim.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BRANCH_COVERAGE_COMPLETE',sha(a.output),json.dumps({k:report[k]for k in ('total_eligible_frames','max_active_union_gap_rad','active_no_float32_branch_witness_elements')}),flush=True)

if __name__=='__main__':main()
