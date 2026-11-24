"""Generate a new V9 audit; leave the reviewed V8 audit and trainer untouched."""
from pathlib import Path
root=Path(__file__).parent
s=(root/'audit_completed_hold_increment_journal_v8.py').read_text(encoding='utf-8-sig')
def replace(old,new,count=1):
 global s
 assert s.count(old)==count,(old,s.count(old));s=s.replace(old,new)
replace('Completed V8 journal/checkpoint/optimizer inventory; no policy inference.','Completed V9 journal/checkpoint/optimizer inventory; trainer-author self-check, no policy inference.')
replace('import train_hold_increment_v8 as trainer','import train_arm_hold_increment_v9 as trainer')
replace("'v8_last_optimizer_state_not_deployment_candidate_v1'","'v9_last_optimizer_state_not_deployment_candidate_v1'")
replace("8*config['epochs']","9*config['epochs']")
replace("len(metrics['runs'])==8","len(metrics['runs'])==9")
replace('zero_bad=hold_bad=0;zero_names=[];hold_names=[]','zero_bad=hold_bad=index_bad=0;zero_names=[];hold_names=[];index_names=[]')
replace("    require(sorted(zero_names)==sorted(trainer.ZERO_BOUNDARY_RUNS)","""        require(row['step560_index_hold_regression_required']==(name in trainer.INDEX_HOLD_RUNS),'Wrong step560 scope')
        if row['step560_index_hold_regression_required']:
            index_names.append(name);h=row['step560']
            require(h['joint_names']==list(trainer.INDEX_HOLD_JOINTS) and h['expert_exact_HOLD']==[True]*3,'Reviewed index HOLD label changed')
            require(all(type(v)is bool for v in h['projection_exact_HOLD']) and len(h['projection_exact_HOLD'])==3,'Invalid index HOLD boolean checks')
            index_bad+=sum(not v for v in h['projection_exact_HOLD'])
    require(sorted(index_names)==sorted(trainer.INDEX_HOLD_RUNS),'Missing reviewed index recovery boundary')
    require(index_bad==metrics['step560_regression_violation_count'],'Index HOLD boundary count differs')
    require(metrics['step560_reviewed_index_history_exact_projected_hold']==(index_bad==0),'Index HOLD pass flag differs')
    require(sorted(zero_names)==sorted(trainer.ZERO_BOUNDARY_RUNS)""")
replace("==q['total_eligible_frames']==7681","==q['total_eligible_frames']==9523")
replace("row['optimizer_steps']==8*count","row['optimizer_steps']==9*count")
replace("len(row['episode_losses'])==8","len(row['episode_losses'])==9")
replace("metadata['optimizer_steps']==8*metadata['best_epoch']","metadata['optimizer_steps']==9*metadata['best_epoch']")
replace("final['step499_regression_violation_count']==final['step545_regression_violation_count']==0","final['step499_regression_violation_count']==final['step545_regression_violation_count']==final['step560_regression_violation_count']==0")
replace("'v8_completed_journal_selection_and_last_state_independent_audit_v1'","'v9_completed_journal_selection_and_last_state_trainer_author_selfcheck_v1'")
replace("'optimizer_steps_per_epoch':8","'optimizer_steps_per_epoch':9")
replace("'independent_policy_inference_repeated_here':False", "'independent_policy_inference_repeated_here':False,'reviewer_role':'trainer_author_self_check_pending_cross_agent_review',\n        'independent_cross_agent_review_performed_here':False")
replace("    require(metadata['source_sha256']==trainer.source_hashes(),'Frozen source changed')", """    require(metadata['source_sha256']==trainer.source_hashes(),'Frozen source changed')
    require(len(trainer.RUN_NAMES)==9 and len(metadata['source_sha256'])==22,'Wrong V9 history/source graph size')
    require(metadata['hidden_size']==config['hidden_size'],'Authorized model width differs')""")
(root/'audit_completed_arm_hold_increment_journal_v9.py').write_text(s,encoding='utf-8',newline='\n')
t=(root/'test_completed_hold_increment_journal_v8.py').read_text(encoding='utf-8-sig')
t=t.replace('train_hold_increment_v8 as trainer','train_arm_hold_increment_v9 as trainer').replace('audit_completed_hold_increment_journal_v8 as audit','audit_completed_arm_hold_increment_journal_v9 as audit').replace('test_train_hold_increment_v8_peer import fixtures','test_train_arm_hold_increment_v9_peer import fixtures')
t=t.replace('range(8)','range(9)').replace('random.Random(3),1,8,metadata,(0,0,2,.001)','random.Random(3),1,9,metadata,(0,0,0,2,.001)').replace('(0,0,2,.001)','(0,0,0,2,.001)')
t=t.replace("['update_anchor_is_previous'][0]=True","['update_anchor_is_previous'][0]=False")
t=t.replace("('confusion','499','545','frames','claim')","('confusion','499','545','560','560scope','560label','frames','claim')")
t=t.replace("            elif kind=='frames':", """            elif kind=='560':modified['step560_regression_violation_count']+=1
            elif kind=='560scope':modified['runs'][-1]['step560_index_hold_regression_required']=False
            elif kind=='560label':modified['runs'][-1]['step560']['expert_exact_HOLD'][0]=False
            elif kind=='frames':""")
(root/'test_completed_arm_hold_increment_journal_v9.py').write_text(t,encoding='utf-8',newline='\n')
