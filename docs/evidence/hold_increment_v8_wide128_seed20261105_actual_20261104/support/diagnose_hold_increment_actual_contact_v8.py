"""Completed actual contact/command causality; no actor replay or label admission."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
import joint_safety
import train_phase_residual_v5 as frozen
import hold_increment_policy_v8 as policy

ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103'
FINGERS=('thumb','index','middle','ring','little')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def intervals(mask):
    out=[];start=None
    for i,value in enumerate(mask):
        if value and start is None:start=i
        if start is not None and(not value or i==len(mask)-1):
            out.append([start,i if value else i-1]);start=None
    return out

def same(a,b):return a.astype(np.float32).view(np.int32)==b.astype(np.float32).view(np.int32)

def load_rows(run):
    rows=[];abort=[];final=None
    with (run/'evaluation_transitions.jsonl').open()as f:
        for line in f:
            assert line.endswith('\n'),'Incomplete actual journal'
            r=json.loads(line)
            if r['record_type']=='executed_transition':
                assert r['step']==len(rows)and r['transition_finished']is True
                audit=r['original_controller_audit']
                rows.append({'step':r['step'],'pre':r['actor_observation_pre'],'post':r['actor_observation_post'],
                    'features':r['actual_actor_features'],'expert':r['counterfactual_expert_projection_diagnostic_only']['projected_target'],
                    'predicted':r['predicted_left_target'],'height_m':r['raw_environment_metrics']['height_m'],
                    'raw_termination':r['raw_environment_termination'],'failure':r['failure_reason'],
                    'terminal':r['terminal'],'wrist_qualified':r['original_wrist_gate_qualified'],
                    'execution_memory_mode':audit['execution_memory_mode'],
                    'execution_memory_changes':audit['execution_memory_changes']})
            elif r['record_type']=='abort':abort.append({'step':r['step'],'reason':r['reason'],
                'physics_step_completed_for_unfinished_decision':r['physics_step_completed_for_unfinished_decision']})
            elif r['record_type']=='evaluation_final':final=r['summary']
    assert final is not None and final['physical_steps']==len(rows)
    return rows,abort,final


def compare_prefix(rows,reference):
    if not reference.is_dir():return {'available':False}
    fields=('applied_target','q','cube_position','cube_quaternion_wxyz')
    first={key:None for key in (*fields,'phase','finger_contact_impulses')};checked=0
    with (reference/'evaluation_transitions.jsonl').open()as stream:
        for line in stream:
            ref=json.loads(line)
            if ref['record_type']!='executed_transition':continue
            step=ref['step']
            if step>=len(rows):break
            actual=rows[step];checked+=1
            for key in fields:
                if first[key]is None and not same(np.asarray(actual['post'][key]),np.asarray(ref['actor_observation_post'][key])).all():
                    first[key]={'step':step,'actual_phase':actual['pre']['phase'],'reference_phase':ref['actor_observation_pre']['phase'],
                        'max_abs_difference':float(np.max(np.abs(np.asarray(actual['post'][key],np.float64)-np.asarray(ref['actor_observation_post'][key],np.float64))))}
            if first['phase']is None and actual['pre']['phase']!=ref['actor_observation_pre']['phase']:
                first['phase']={'step':step,'actual_phase':actual['pre']['phase'],'reference_phase':ref['actor_observation_pre']['phase']}
            if first['finger_contact_impulses']is None and actual['post']['finger_contact_impulses']!=ref['actor_observation_post']['finger_contact_impulses']:
                first['finger_contact_impulses']={'step':step,'actual':actual['post']['finger_contact_impulses'],'reference':ref['actor_observation_post']['finger_contact_impulses']}
    return {'available':True,'reference_run':str(reference),'reference_result_sha256':sha(reference/'result.json'),
        'shared_steps_checked':checked,'first_difference':first,'state_comparison':'float32 bitwise per field; impulse dictionaries exact; after phase divergence equal step is not equal task phase'}


def diagnose(run,output):
    assert not output.exists(),'Keep prior diagnostic'
    result=read(run/'result.json');identity=read(run/'evaluation_protocol.json');params=read(run/'projection_parameters.json')
    assert identity['policy']=='v8'
    assert sha(joint_safety.__file__)==params['source_sha256']['joint_safety.py']
    rows,abort,final=load_rows(run);assert len(rows)==result['physical_steps']==result['fully_audited_transition_steps']
    assert final['failure']==result['failure'] and result['left_teacher_position_action_steps']==0
    names=params['joint_names'];left=[i for i,n in enumerate(names)if n.endswith('_L')or n.startswith('left_')]
    left_names=[names[i]for i in left];assert len(left)==28
    baseline,basehash=frozen.load_baseline(run/'checkpoints/frozen_v4_baseline.pt')
    assert basehash==identity['frozen_v4_baseline_checkpoint_sha256']
    assert identity['update_bound_rad']==np.asarray(policy.candidate_bounds(left_names),np.float32).tolist()
    m=policy.HoldIncrementPolicy(baseline,np.zeros(policy.FEATURES),np.ones(policy.FEATURES),identity['update_bound_rad'],
        hidden_size=8,left_joint_names=left_names,action_contract=identity['action_contract'])
    assert m.update_anchor_is_previous.tolist()==identity['update_anchor_is_previous']
    x=torch.tensor([r['features']for r in rows],dtype=torch.float32).unsqueeze(0)
    old=torch.tensor([r['pre']['applied_target']for r in rows],dtype=torch.float32)[:,left].unsqueeze(0)
    labels=torch.tensor([r['expert']for r in rows],dtype=torch.float32)[:,left].unsqueeze(0)
    actual=np.asarray([r['post']['applied_target']for r in rows],np.float32)[:,left]
    with torch.no_grad():
        # Frozen phase baseline only. The learned recurrent actor is NOT run.
        base=m.baseline_targets(x,old)
        project=lambda p:policy.project_left_targets(p,old,params,left,joint_safety.project_target)
        base_projected=project(base);hold_projected=project(old)
        anchor=torch.where(m.update_anchor_is_previous,old,base)
        clamp=lambda p:torch.clamp(p,m.baseline.lower.float(),m.baseline.upper.float())
        low=project(clamp(anchor-m.residual_bound));high=project(clamp(anchor+m.residual_bound))
        gap=torch.maximum(torch.minimum(low,high)-labels,labels-torch.maximum(low,high)).clamp_min(0.)
        union_gap=torch.minimum(torch.minimum((labels.double()-base_projected.double()).abs(),
            (labels.double()-hold_projected.double()).abs()),gap.double())[0].numpy()
        classes=policy.classify_branch_labels(labels,base_projected,hold_projected)
        active=m.residual_active(x)[0].numpy()
    previous=old[0].numpy();expert=labels[0].numpy();base=base[0].numpy()
    base_projected=base_projected[0].numpy();hold_projected=hold_projected[0].numpy()
    pre=np.array([[r['pre']['finger_contact_impulses']['left'][f]for f in FINGERS]for r in rows])
    post=np.array([[r['post']['finger_contact_impulses']['left'][f]for f in FINGERS]for r in rows])
    phases=np.asarray([r['pre']['phase']for r in rows]);reports={}
    def summarize(eligible):
        details={}
        for k,finger in enumerate(FINGERS):
            js=[j for j,n in enumerate(left_names)if n.startswith('left_'+finger+'_')]
            hold=same(expert[:,js],hold_projected[:,js]).all(1)
            moves=~same(actual[:,js],hold_projected[:,js]).all(1)
            selected=eligible&active&(pre[:,k]>1e-5)&hold
            bad=selected&moves;ids=np.flatnonzero(bad)
            details[finger]={'post_positive_frames':int(((post[:,k]>1e-5)&eligible).sum()),
                'eligible_frames':int(eligible.sum()),'post_no_contact_intervals':intervals((post[:,k]<=1e-5)&eligible),
                'contact_active_exact_expert_HOLD_frames':int(selected.sum()),
                'exact_expert_HOLD_but_actual_target_differs_frames':int(bad.sum()),
                'first_exact_HOLD_target_difference_step':int(ids[0])if len(ids)else None,
                'max_target_difference_while_exact_HOLD_rad':float(np.abs(actual[:,js].astype(np.float64)-hold_projected[:,js])[selected].max())if selected.any()else 0.,
                'first_union_unreachable_step':int(np.flatnonzero((union_gap[:,js]>0).any(1)&eligible&active)[0])if((union_gap[:,js]>0).any(1)&eligible&active).any()else None}
        return details
    reports['all_feedback_active']=summarize(active)
    for phase in dict.fromkeys(phases):reports[phase]=summarize(phases==phase)
    differences=np.abs(actual.astype(np.float64)-expert.astype(np.float64))
    first_error=np.argwhere((differences>1e-6)&active[:,None])
    selected=set(range(max(0,len(rows)-5),len(rows)))
    selected.update(range(540,min(551,len(rows))))
    if len(rows)>499:selected.add(499)
    if len(first_error):selected.update(range(max(0,int(first_error[0,0])-2),min(len(rows),int(first_error[0,0])+4)))
    for values in reports['all_feedback_active'].values():
        step=values['first_exact_HOLD_target_difference_step']
        if step is not None:selected.update(range(max(0,step-1),min(len(rows),step+3)))
    def row(step):
        r=rows[step]
        return {'step':step,'phase':r['pre']['phase'],'progress':r['pre']['phase_progress'],'height_m':r['height_m'],
            'pre_impulses':r['pre']['finger_contact_impulses']['left'],'post_impulses':r['post']['finger_contact_impulses']['left'],
            'cube_position':r['post']['cube_position'],'cube_quaternion_wxyz':r['post']['cube_quaternion_wxyz'],
            'execution_memory_mode':r['execution_memory_mode'],'execution_memory_changes':r['execution_memory_changes'],
            'joint_targets':{name:{'previous_actual':float(previous[step,j]),'expert_projected':float(expert[step,j]),
                'actual_applied':float(actual[step,j]),'baseline_proposal':float(base[step,j]),'baseline_projected':float(base_projected[step,j]),
                'hold_projected':float(hold_projected[step,j]),'expert_counterfactual_class':int(classes['class_targets'][0,step,j]),
                'class_confident':bool(classes['classification_mask'][0,step,j]),
                'expert_equals_HOLD_bitwise':bool(same(expert[step,j:j+1],hold_projected[step,j:j+1]).all()),
                'actual_equals_HOLD_bitwise':bool(same(actual[step,j:j+1],hold_projected[step,j:j+1]).all()),
                'actual_minus_expert_rad':float(differences[step,j]),'union_interval_gap_rad':float(union_gap[step,j])}
                for j,name in enumerate(left_names)if name.startswith(('left_ring_','left_index_','left_middle_'))}}
    comparison={label:compare_prefix(rows,CODE/'runs'/name)for label,name in (
        ('successful_v4','left_phase_v4_matched_eval_9501_20261104_v2'),
        ('first_failed_v7','left_zero_gated_residual_v7_eval_9501_20261104'))}
    value={'schema':'completed_v8_actual_contact_and_exact_teacher_HOLD_diagnostic_v1','run':str(run),
        'original_result':result,'final_evaluation_summary':final,'unexecuted_abort_evidence':abort,
        'contact_and_target_by_phase':reports,'prefix_comparisons':comparison,
        'first_active_expert_target_error_gt1e_6':None if not len(first_error)else{'step':int(first_error[0,0]),'joint':left_names[int(first_error[0,1])],
            'absolute_error_rad':float(differences[tuple(first_error[0])])},
        'active_union_interval_unreachable_elements':int(((union_gap>0)&active[:,None]).sum()),
        'active_max_union_interval_gap_rad':float(union_gap[active].max())if active.any()else 0.,
        'selected_actual_rows':[row(i)for i in sorted(selected)],
        'source_sha256':{str(p.relative_to(ROOT)):sha(p)for p in (run/'result.json',run/'evaluation_transitions.jsonl',run/'projection_parameters.json',
            run/'source_snapshot/opposition_feedback.py',run/'source_snapshot/opposition_teacher.py',Path(__file__).resolve())},
        'original_teacher_semantics':{'positive_contact_threshold_Ns':1e-5,'positive_contact_action':'HOLD previous actually applied target',
            'hold_is_not_return_to_v4_baseline':True,'no_contact_motion_toward_geometric_target_max_step_rad':.005},
        'learned_actor_replayed_here':False,'observed_equal_target_is_not_claimed_to_identify_learned_mode':True,
        'interval_coverage_is_not_neural_parameterization_or_physical_success_proof':True,
        'counterfactual_labels_qualified_for_training':False,'alternative_control_trajectory_executed':False,
        'source_or_result_modified':False,'isaac_started':False}
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'report':str(output),'sha256':sha(output),'steps':len(rows),'failure':result['failure'],
        'wrist_qualified':result['original_wrist_gate_qualified'],'fingers':reports['all_feedback_active'],
        'first_active_expert_target_error_gt1e_6':value['first_active_expert_target_error_gt1e_6'],
        'active_union_interval_unreachable_elements':value['active_union_interval_unreachable_elements'],
        'active_max_union_interval_gap_rad':value['active_max_union_interval_gap_rad']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=(20261104,20261105),required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    diagnose(CODE/'runs'/f'left_hold_increment_v8_seed{args.seed}_eval_9501_20261104',
             WORK/f'v8_seed{args.seed}_actual_contact_diagnosis_20261104.json')


if __name__=='__main__':main()
