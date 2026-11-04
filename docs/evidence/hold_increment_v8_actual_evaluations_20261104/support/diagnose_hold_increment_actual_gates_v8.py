"""Read-only completed V8 target-envelope and original hold-gate decomposition."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import diagnose_hold_increment_actual_contact_v8 as d


def compact_teacher(state):
    return {k:state.get(k) for k in ('name','mode','tick','hold','lost','tilt_error_deg','accepted','failure_reason','path_sha256')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(20261104,20261105),required=True);a=p.parse_args()
    torch.set_num_threads(1)
    run=d.CODE/'runs'/f'left_hold_increment_v8_seed{a.seed}_eval_9501_20261104'
    out=d.WORK/f'v8_seed{a.seed}_actual_hold_gate_decomposition_20261104.json'
    assert not out.exists()
    result=d.read(run/'result.json');identity=d.read(run/'evaluation_protocol.json');params=d.read(run/'projection_parameters.json')
    rows,aborts,final=d.load_rows(run)
    names=params['joint_names'];left=[i for i,n in enumerate(names)if n.endswith('_L')or n.startswith('left_')];ln=[names[i]for i in left]
    baseline,bhash=d.frozen.load_baseline(run/'checkpoints/frozen_v4_baseline.pt');assert bhash==identity['frozen_v4_baseline_checkpoint_sha256']
    assert d.sha(d.joint_safety.__file__)==params['source_sha256']['joint_safety.py']
    m=d.policy.HoldIncrementPolicy(baseline,np.zeros(d.policy.FEATURES),np.ones(d.policy.FEATURES),identity['update_bound_rad'],hidden_size=8,left_joint_names=ln,action_contract=identity['action_contract'])
    x=torch.tensor([r['features']for r in rows],dtype=torch.float32)[None]
    old=torch.tensor([r['pre']['applied_target']for r in rows],dtype=torch.float32)[:,left][None]
    expert=torch.tensor([r['expert']for r in rows],dtype=torch.float32)[:,left][None]
    actual=np.array([r['post']['applied_target']for r in rows],np.float32)[:,left]
    with torch.no_grad():
        base=m.baseline_targets(x,old);project=lambda p:d.policy.project_left_targets(p,old,params,left,d.joint_safety.project_target)
        pb=project(base);ph=project(old);anchor=torch.where(m.update_anchor_is_previous,old,base)
        clamp=lambda p:torch.clamp(p,m.baseline.lower.float(),m.baseline.upper.float())
        lo=project(clamp(anchor-m.residual_bound));hi=project(clamp(anchor+m.residual_bound))
        gap=torch.maximum(torch.minimum(lo,hi)-expert,expert-torch.maximum(lo,hi)).clamp_min(0.)
        union=torch.minimum(torch.minimum((expert.double()-pb.double()).abs(),(expert.double()-ph.double()).abs()),gap.double())[0].numpy()
        active=m.residual_active(x)[0].numpy()
    old,expert,base,pb,ph,lo,hi=(t[0].numpy()for t in (old,expert,base,pb,ph,lo,hi))
    phases=np.array([r['pre']['phase']for r in rows]);bounds=np.array(identity['update_bound_rad'])
    def target_row(t,j):
        return {'step':int(t),'phase':str(phases[t]),'joint':ln[j],'update_anchor':'previous_actual'if m.update_anchor_is_previous[j]else'frozen_base',
                'bound_rad':float(bounds[j]),'expert_projected':float(expert[t,j]),'actual_applied':float(actual[t,j]),'previous_actual':float(old[t,j]),
                'baseline_proposal':float(base[t,j]),'projected_BASE':float(pb[t,j]),'projected_HOLD':float(ph[t,j]),
                'update_interval_endpoints':[float(lo[t,j]),float(hi[t,j])],'union_gap_rad':float(union[t,j]),
                'expert_minus_previous_rad':float(expert[t,j].astype(np.float64)-old[t,j]),
                'actual_minus_previous_rad':float(actual[t,j].astype(np.float64)-old[t,j]),
                'actual_equals_BASE_bitwise':bool(d.same(actual[t,j:j+1],pb[t,j:j+1])[0]),
                'actual_equals_HOLD_bitwise':bool(d.same(actual[t,j:j+1],ph[t,j:j+1])[0])}
    unreachable={}
    for j,n in enumerate(ln):
        ids=np.flatnonzero((union[:,j]>0)&active)
        if len(ids):
            t=int(ids[np.argmax(union[ids,j])]);unreachable[n]={'count':len(ids),'first':target_row(int(ids[0]),j),'maximum':target_row(t,j),
                'by_phase':{str(p):int(((union[:,j]>0)&active&(phases==p)).sum())for p in dict.fromkeys(phases)if((union[:,j]>0)&active&(phases==p)).any()}}
    index_events=[];ring_events=[]
    for finger,sink in (('index',index_events),('ring',ring_events)):
        js=[j for j,n in enumerate(ln)if n.startswith('left_'+finger+'_')]
        for t,r in enumerate(rows):
            if active[t]and r['pre']['finger_contact_impulses']['left'][finger]>1e-5 and d.same(expert[t,js],ph[t,js]).all()and not d.same(actual[t,js],ph[t,js]).all():
                sink.append({'step':t,'phase':str(phases[t]),'pre_impulse_Ns':r['pre']['finger_contact_impulses']['left'][finger],
                    'post_impulse_Ns':r['post']['finger_contact_impulses']['left'][finger],
                    'joint_changes':[target_row(t,j)for j in js if not d.same(actual[t,j:j+1],ph[t,j:j+1])[0]]})
    # Use the exact original teacher query inputs, not post-physics state.
    queries=[];path_lengths={};gate_rows=[]
    for line in (run/'evaluation_transitions.jsonl').open():
        r=json.loads(line)
        if r['record_type']=='executed_transition':decision=r;executed=True
        elif r['record_type']=='abort'and isinstance(r.get('unfinished_decision'),dict):decision=r['unfinished_decision'];executed=False
        else:continue
        audit=decision.get('original_controller_audit')
        if not audit:continue
        before=audit['teacher_state_before_sync'];after=audit['teacher_state_after_query'];metrics=audit['oracle_env_inputs']['metrics']
        if 'actor_observation_pre'in decision:pre=decision['actor_observation_pre']
        else:
            assert not executed and decision['step']==len(rows) and r['physics_step_completed_for_unfinished_decision']is False
            pre=rows[-1]['post']  # No next physics step occurred before the rejected query.
        if after['name']!='tilt_hold':continue
        # At hold, the original code only updates the final pose, not path length.
        # An aborted query may have an unarchived new final pose; use its archived
        # before-query path for length without inventing the absent plan bytes.
        pathhash=before['path_sha256']
        if pathhash not in path_lengths:
            obj=d.read(run/'plans'/('tilt_'+pathhash+'.json'))
            assert isinstance(obj,list);path_lengths[pathhash]=len(obj)
        plen=path_lengths[pathhash];imp=pre['finger_contact_impulses']['left']
        flags={'path_finished':before['tick']>=plen,'all_fingers':all(imp.get(f,0)>1e-5 for f in d.FINGERS),
            'height_at_least_08':metrics.get('height_m',0)>=.08,'no_cube_table':not metrics.get('cube_table_contact',True),
            'tilt_error_at_most_5':after['tilt_error_deg']<=5.,'linear_speed_below_03':metrics.get('linear_speed',0)<.03,
            'angular_speed_below_20':metrics.get('angular_speed',0)<.20}
        expected_hold=before['hold']+1 if all(flags.values())else 0
        assert expected_hold==after['hold'],('Original hold reconstruction mismatch',decision['step'],flags,before['hold'],after['hold'])
        gate_rows.append({'step':decision['step'],'physics_executed':executed,'flags':flags,'hold_count':after['hold'],
            'failed':sorted(k for k,v in flags.items()if not v),'no_contact_fingers':[f for f in d.FINGERS if imp.get(f,0)<=1e-5],
            'linear_speed_m_s':metrics.get('linear_speed'),'angular_speed_rad_s':metrics.get('angular_speed'),
            'tilt_error_deg':after['tilt_error_deg'],'teacher_after':compact_teacher(after),
            'path_length_source_before_query_sha256':pathhash,'path_length':plen,
            'after_query_plan_bytes_archived':(run/'plans'/('tilt_'+after['path_sha256']+'.json')).exists()})
    assert gate_rows
    keys=list(gate_rows[0]['flags']);fail_counts={k:sum(not r['flags'][k]for r in gate_rows)for k in keys}
    combinations={}
    for r in gate_rows:
        key='+'.join(r['failed'])or'qualified_this_query';combinations[key]=combinations.get(key,0)+1
    witness_js=[j for j,n in enumerate(ln)if n.startswith(('left_index_','left_ring_'))]
    witness_steps=range(534,min(len(rows),576))
    report={'schema':'v8_actual_contact_original_hold_gate_and_interval_decomposition_v1','run':str(run),
        'physical_steps':len(rows),'original_failure':result['failure'],'original_result_sha256':d.sha(run/'result.json'),
        'source_sha256':{str(p.relative_to(d.ROOT)):d.sha(p)for p in (Path(__file__).resolve(),Path(d.__file__).resolve(),run/'evaluation_transitions.jsonl',run/'source_snapshot/tilt_regrasp.py',run/'source_snapshot/palm_handover.py')},
        'union_unreachable_by_joint':unreachable,'union_unreachable_elements':int(((union>0)&active[:,None]).sum()),
        'max_union_gap_rad':float(union[active].max()),'recovery11_max_abs_expert_minus_previous_rad':{n:float(np.abs(expert[:,j].astype(np.float64)-old[:,j])[active].max())for j,n in enumerate(ln)if m.update_anchor_is_previous[j]},
        'index_contact_exact_expert_HOLD_but_actual_differs':index_events,'ring_contact_exact_expert_HOLD_but_actual_differs':ring_events,
        'hold_gate':{'query_count':len(gate_rows),'first_step':gate_rows[0]['step'],'last_step':gate_rows[-1]['step'],
            'max_consecutive_hold':max(r['hold_count']for r in gate_rows),'required_consecutive_hold':90,
            'predicate_failure_query_counts':fail_counts,'predicate_failure_combinations':combinations,'queries':gate_rows,
            'counterfactual_if_each_single_predicate_removed':{k:max((b-a+1 for a,b in d.intervals([all(v for f,v in r['flags'].items()if f!=k)for r in gate_rows])),default=0)for k in keys},
            'counterfactual_is_diagnostic_only_no_gate_changed':True,'source_exact_hold_count_reconstruction_passed':True},
        'early_contact_target_witnesses':[{'step':t,'pre_impulses':rows[t]['pre']['finger_contact_impulses']['left'],'post_impulses':rows[t]['post']['finger_contact_impulses']['left'],
                'targets':[target_row(t,j)for j in witness_js]}for t in witness_steps],
        'learned_actor_replayed':False,'observed_action_branch_equality_not_mode_claim':True,'interval_is_not_parameterization_or_physical_success':True,
        'training_qualification_granted':False,'original_result_modified':False,'isaac_started':False}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'report':str(out),'sha256':d.sha(out),'unreachable':{k:{'count':v['count'],'first_step':v['first']['step'],'max':v['maximum']}for k,v in unreachable.items()},
        'hold_gate':{k:v for k,v in report['hold_gate'].items()if k!='queries'}},indent=2))


if __name__=='__main__':main()
