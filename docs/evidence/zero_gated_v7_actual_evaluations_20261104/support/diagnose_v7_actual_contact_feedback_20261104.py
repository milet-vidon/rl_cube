"""Read-only actual-state contact/command diagnostics; no new label qualification."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import joint_safety
import train_phase_residual_v5 as frozen
import zero_gated_residual_policy_v7 as policy

ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103'
RUN=CODE/'runs/left_zero_gated_residual_v7_eval_9501_20261104'
OUTPUT=WORK/'v7_actual_contact_feedback_diagnosis_20261104.json'
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
params=json.loads((RUN/'projection_parameters.json').read_text());names=params['joint_names'];left=[i for i,n in enumerate(names)if n.endswith('_L')or n.startswith('left_')]
assert len(left)==28 and sha(joint_safety.__file__)==params['source_sha256']['joint_safety.py']
baseline,basehash=frozen.load_baseline(CODE/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
assert basehash=='d2493b00df27b8d84ae69da3b399b3beb926b77fdd87a9b51aa6a2b1b23a36a4'
m=policy.ZeroGatedResidualPolicy(baseline,np.zeros(policy.FEATURES),np.ones(policy.FEATURES),policy.candidate_bounds([names[i]for i in left]))
records=[];abort=None
for line in (RUN/'evaluation_transitions.jsonl').open():
    r=json.loads(line)
    if r['record_type']=='executed_transition':records.append(r)
    elif r['record_type']=='abort':abort=r
assert len(records)==752 and [r['step']for r in records]==list(range(752))
x=torch.tensor([r['actual_actor_features']for r in records],dtype=torch.float32).unsqueeze(0)
previous=torch.tensor([r['actor_observation_pre']['applied_target']for r in records],dtype=torch.float32)[:,left].unsqueeze(0)
expert=torch.tensor([r['counterfactual_expert_projection_diagnostic_only']['projected_target']for r in records],dtype=torch.float32)[:,left].unsqueeze(0)
actual=torch.tensor([r['actor_observation_post']['applied_target']for r in records],dtype=torch.float32)[:,left].unsqueeze(0)
with torch.no_grad():
    base=m.baseline_targets(x,previous);bound=m.effective_bounds(x)
    project=lambda p:policy.project_left_targets(p,previous,params,left,joint_safety.project_target)
    projected_base=project(base)
    low=project(torch.clamp(base-bound,m.baseline.lower.float(),m.baseline.upper.float()))
    high=project(torch.clamp(base+bound,m.baseline.lower.float(),m.baseline.upper.float()))
    gap=torch.maximum((torch.minimum(low,high)-expert).clamp_min(0),(expert-torch.maximum(low,high)).clamp_min(0))[0].numpy()
    cls=policy.classify_correction_labels(expert,projected_base)
base=base[0].numpy();base_projected=projected_base[0].numpy();old=previous[0].numpy();ex=expert[0].numpy();actual=actual[0].numpy();active=m.residual_active(x)[0].numpy()
left_names=[names[i]for i in left]
pre=np.array([[r['actor_observation_pre']['finger_contact_impulses']['left'][f]for f in ('thumb','index','middle','ring','little')]for r in records])
post=np.array([[r['actor_observation_post']['finger_contact_impulses']['left'][f]for f in ('thumb','index','middle','ring','little')]for r in records])

def intervals(mask):
    result=[];start=None
    for i,v in enumerate(mask):
        if v and start is None:start=i
        if start is not None and(not v or i==len(mask)-1):
            end=i if v else i-1;result.append([start,end]);start=None
    return result

def row(step):
    r=records[step]
    return {'step':step,'phase':r['actor_observation_pre']['phase'],'progress':r['actor_observation_pre']['phase_progress'],
        'pre_contact_impulses':r['actor_observation_pre']['finger_contact_impulses']['left'],
        'post_contact_impulses':r['actor_observation_post']['finger_contact_impulses']['left'],
        'height_m':r['raw_environment_metrics']['height_m'],'cube_position':r['actor_observation_post']['cube_position'],
        'left_joints':{name:{'previous_actual':float(old[step,j]),'expert_projected':float(ex[step,j]),'actual_applied':float(actual[step,j]),
            'baseline_proposal':float(base[step,j]),'baseline_projected':float(base_projected[step,j]),
            'expert_minus_previous':float(np.float64(ex[step,j])-old[step,j]),'actual_minus_previous':float(np.float64(actual[step,j])-old[step,j]),
            'expert_minus_base_proposal':float(np.float64(ex[step,j])-base[step,j]),'actual_minus_expert':float(np.float64(actual[step,j])-ex[step,j]),
            'unreachable_distance_rad':float(gap[step,j])}for j,name in enumerate(left_names)if name.startswith(('left_ring_','left_index_'))}}

finger_reports={}
for k,f in enumerate(('thumb','index','middle','ring','little')):
    js=[i for i,n in enumerate(left_names)if n.startswith('left_'+f+'_')]
    touch=pre[:,k]>1e-5;select=touch&active
    hold=np.max(np.abs(ex[:,js].astype(np.float64)-old[:,js]),axis=1)<=1e-6
    movement=np.max(np.abs(actual[:,js].astype(np.float64)-old[:,js]),axis=1)>1e-6
    bad=select&hold&movement
    finger_reports[f]={'post_no_contact_intervals_after_feedback':intervals((post[:,k]<=1e-5)&active),
        'post_positive_count_after_499':int((post[499:,k]>1e-5).sum()),'post_total_after_499':len(records)-499,
        'contact_active_expert_holds_previous_frames':int((select&hold).sum()),
        'contact_active_expert_hold_but_actual_moves_frames':int(bad.sum()),
        'first_contact_hold_actual_movement_step':int(np.flatnonzero(bad)[0])if bad.any()else None,
        'max_actual_motion_while_expert_holds_rad':float(np.max(np.abs(actual[:,js].astype(np.float64)-old[:,js])[select&hold]))if(select&hold).any()else 0.,
        'max_unreachable_distance_rad_after_499':float(gap[499:,js].max()),
        'first_unreachable_gt1e_6_step':int(np.flatnonzero((gap[:,js]>1e-6).any(1)&active)[0])if((gap[:,js]>1e-6).any(1)&active).any()else None}
unreachable=[]
for step,j in zip(*np.where((gap>1e-6)&active[:,None])):
    unreachable.append({'step':int(step),'joint':left_names[j],'gap_rad':float(gap[step,j]),
        'expert_minus_base_proposal_rad':float(np.float64(ex[step,j])-base[step,j]),'bound_rad':float(m.residual_bound[j])})
top=sorted(unreachable,key=lambda r:r['gap_rad'],reverse=True)[:25]
selected=sorted(set(list(range(542,558))+[499,560,575,600,625,650,700,751]+[
    item['first_contact_hold_actual_movement_step']for item in finger_reports.values()if item['first_contact_hold_actual_movement_step']is not None]+[
    item['first_unreachable_gt1e_6_step']for item in finger_reports.values()if item['first_unreachable_gt1e_6_step']is not None]))
result=json.loads((RUN/'result.json').read_text())
value={'schema':'actual_v7_contact_hold_and_original_projection_reachability_diagnostic_v1',
    'run':str(RUN),'sources':{str(p.relative_to(ROOT)):sha(p)for p in (RUN/'evaluation_transitions.jsonl',RUN/'result.json',RUN/'projection_parameters.json',
        RUN/'source_snapshot/opposition_feedback.py',RUN/'source_snapshot/opposition_teacher.py',Path(__file__))},
    'original_result':result,'fingers':finger_reports,'active_projected_label_unreachable_elements_gt1e_6':len(unreachable),
    'active_max_unreachable_distance_rad':float(gap[active].max()),'unreachable_by_joint':{n:int(np.count_nonzero((gap[:,j]>1e-6)&active))for j,n in enumerate(left_names)},
    'first_unreachable':unreachable[:20],'largest_unreachable':top,'selected_actual_rows':[row(i)for i in selected],
    'original_teacher_semantics':{'source':'source_snapshot/opposition_feedback.py:12-39','contact_threshold_Ns':1e-5,
        'positive_contact_action':'hold actual previous applied target','contact_hold_is_not_return_to_v4_baseline':True,
        'no_contact_action':'move toward cached/refit geometric target by at most 0.005 rad per step',
        'refit_every_n_feedback_ticks':3,'actual_applied_memory_sync_is_logged':True},
    'claim_scope':'actual causal ordering and counterfactual target reachability; no alternative control trajectory executed',
    'labels_qualified_for_training':False,'source_or_result_modified':False,'simulation_run_started':False}
OUTPUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(OUTPUT),'sha256':sha(OUTPUT),'finger_summary':finger_reports,
    'unreachable_elements':len(unreachable),'max_unreachable':value['active_max_unreachable_distance_rad'],'first_unreachable':unreachable[:8]},indent=2))
