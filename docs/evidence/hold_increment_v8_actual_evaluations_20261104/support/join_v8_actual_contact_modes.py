"""Join independent recorded contact diagnostics with already verified actor modes."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from diagnose_hold_increment_actual_contact_v8 import ROOT,CODE,WORK,read,sha,same


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(20261104,20261105),required=True);args=p.parse_args();seed=args.seed
    run=CODE/'runs'/f'left_hold_increment_v8_seed{seed}_eval_9501_20261104'
    gate=WORK/f'v8_seed{seed}_actual_hold_gate_decomposition_20261104.json'
    contact=WORK/f'v8_seed{seed}_actual_contact_diagnosis_20261104.json'
    branchdir=WORK/f'v8_seed{seed}_actual_registry_v2_audit_20261104';branch=branchdir/'branch_replay.jsonl'
    out=WORK/f'v8_seed{seed}_contact_mode_causal_join_20261104.json';assert not out.exists()
    g=read(gate);c=read(contact);identity=read(run/'evaluation_protocol.json');names=identity['update_bound_joint_names']
    rows=[json.loads(x)for x in branch.open()];assert len(rows)==g['physical_steps'];assert all(x['step']==i for i,x in enumerate(rows))
    events={};counters={}
    mode=('BASE','HOLD','UPDATE')
    for finger in ('index','ring'):
        output=[];count=collections.Counter()
        for event in g[finger+'_contact_exact_expert_HOLD_but_actual_differs']:
            b=rows[event['step']];details=[]
            for change in event['joint_changes']:
                j=names.index(change['joint'])
                assert same(np.array([b['actual_model_projected'][j]]),np.array([change['actual_applied']]))[0]
                assert same(np.array([b['expert_projected'][j]]),np.array([change['expert_projected']]))[0]
                predicted=mode[b['predicted_class'][j]];expert=mode[b['expert_class'][j]]
                count[expert+'->'+predicted]+=1
                details.append(change|{'predicted_class':predicted,'expert_class':expert,'expert_class_confident':b['expert_classification_mask'][j],'mode_logits':b['mode_logits'][j]})
            output.append({k:v for k,v in event.items()if k!='joint_changes'}|{'joint_changes':details})
        events[finger]=output;counters[finger]=dict(count)
    selected=[]
    for t in (545,548,552,559,560,561,562,581):
        if t>=len(rows):continue
        b=rows[t];selected.append({'step':t,'phase':b['phase'],'index_ring':{n:{'mode':mode[b['predicted_class'][j]],'expert_mode':mode[b['expert_class'][j]],
            'confident':b['expert_classification_mask'][j],'previous':b['previous_left'][j],'expert':b['expert_projected'][j],'actual':b['actual_model_projected'][j],
            'logits':b['mode_logits'][j]}for j,n in enumerate(names)if n.startswith(('left_index_','left_ring_'))}})
    report={'schema':'v8_verified_actual_modes_contact_causality_join_v1','seed':seed,'physical_steps':len(rows),
        'original_failure':c['original_result']['failure'],'first_deviation_from_matched_v4':c['prefix_comparisons']['successful_v4']['first_difference'],
        'hold_wrong_mode_joint_element_counts':counters,'all_exact_expert_hold_deviation_events':events,
        'selected_early_modes':selected,'sources_sha256':{str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__).resolve(),contact,gate,branch,branchdir/'report.json')},
        'source_report_actor_replay_binding':'Independent registry audit verified each recorded feature and action from a single actual-history forward; this join runs no model.',
        'causal_limit':'Temporal command/contact consequences are observed; alternative control outcomes have not been executed.',
        'labels_admitted_for_training':False,'new_actor_forward_calls':0,'isaac_started':False}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'report':str(out),'sha256':sha(out),'counts':counters,'selected':selected[:6]},indent=2))


if __name__=='__main__':main()
