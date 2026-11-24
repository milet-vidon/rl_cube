"""One actual-history V8 actor replay, diagnostic only; no dataset admission.

This does not widen either frozen registry. The sole accepted run/checkpoint/
qualification/authorization tuple is explicitly pinned. A teacher query that
raised before returning has no expert label and receives no actor call.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json
import numpy as np
import torch
import train_hold_increment_v8 as trainer
import hold_increment_policy_v8 as policy
import state_hold_increment_eval_runtime_v8 as runtime
import eval_prefix_dataset_v7 as engine
import dagger_collection_core_v1 as core

RUN_NAME='left_hold_increment_v8_wide128_seed20261105_eval_9501_20261104'
CHECKPOINT='cba7ebc92496e6e32d33e83f29357677fcbf8d3037f456403a3734093dce82bb'
QUALIFICATION='810b793afb4b827bfcea72778c66f55883552f143b0fda4bd2245eb5aaa2d1b9'
AUTHORIZATION='0a5a9963c32a062c0cf13e0085ee3c25463bb0a7f62b4873a8717b7611147a91'
RESULT_SHA='8ac1cbe8d3225d841a909d293f1211da619dc71cb6591806c89674a047ddfb7c'
RUNTIME_SHA='8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c'
ENTRY_SHA='54a9f1ae958cacd55434397e72b04c10743c8a16c1be24d74f3518010ed497c3'
ENGINE_SHA='61c6b43cf0b2af18144723f6f3b2587dea04299feb75611e1a09c7c073b67883'
FAILURE='evaluation_protocol_error:Post-turn correction exceeded bounded pose envelope'
require=engine.require;read=engine.read_json;bits=engine.bits_equal
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def check_unreturned_abort(record,count,result):
 require(record['record_type']=='abort'and record['step']==count,'Wrong abort boundary')
 require(record['reason']==result['failure']==FAILURE,'Wrong exception outcome')
 require(record['unfinished_decision']=={'step':count,'record_type':'decision_attempt','left_teacher_position_action_executed':False},'Unreturned query must have no invented audit, action, features or label')
 for key in ('physical_steps','environment_episode_counter','fully_audited_transition_steps','action_execution_dispatch_attempts','trajectory_rows_recorded'):
  require(record[key]==count,'Abort count changed: '+key)
 require(record['physics_step_completed_for_unfinished_decision']is False and record['execution_dispatch_without_counter_advance']is False,'Unfinished query dispatched physics')
 require(record['invalid_evidence_fields']==[],'Invalid abort evidence')
 return {'step':count,'reason':record['reason'],'teacher_query_returned':False,'expert_label_present':False,
         'actor_replayed':False,'physics_executed':False,'training_qualification':False}

def load_completed_records(run,result):
 records=[];abort=None;final=None
 with (run/'evaluation_transitions.jsonl').open()as f:
  for line in f:
   require(line.endswith('\n'),'Incomplete journal line');r=engine.parse_line(line)
   require(final is None,'Record after final summary')
   if r['record_type']=='executed_transition':
    require(abort is None and r['step']==len(records)and r['transition_finished']is True,'Noncontiguous execution')
    require(r['terminal']is False and r['failure_reason']is None and r['original_wrist_gate_qualified']is False,'Unexpected earlier terminal')
    require(not r['invalid_evidence_fields'],'Invalid executed evidence');records.append(r)
   elif r['record_type']=='abort':
    require(abort is None,'Repeated abort');abort=check_unreturned_abort(r,len(records),result)
   else:
    require(r['record_type']=='evaluation_final'and abort is not None,'Unknown record or premature final');final=r['summary']
 require(abort is not None and final is not None and records,'Incomplete failed evaluation')
 for key in ('steps','physical_steps','environment_episode_counter','fully_audited_transition_steps','action_execution_dispatch_attempts','trajectory_rows_recorded','teacher_query_count'):
  require(result[key]==final[key]==len(records),'Final count mismatch: '+key)
 require(final['failure']==FAILURE and final['original_wrist_gate_qualified']is False,'Final outcome differs')
 return records,abort,final

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=a.run.resolve();require(run.name==RUN_NAME,'Diagnostic is pinned to one actual run')
 require(not a.output.exists(),'New diagnostic output required');a.output.mkdir(parents=True)
 torch.set_num_threads(1)
 sources=trainer.source_hashes()|{Path(m.__file__).name:sha(m.__file__)for m in(runtime,engine,core)}|{Path(__file__).name:sha(__file__)}
 require(sha(runtime.__file__)==RUNTIME_SHA and sha(engine.__file__)==ENGINE_SHA,'Frozen replay modules changed')
 require(sha(run/'result.json')==RESULT_SHA,'Pinned actual result changed')
 manifest=read(run/'artifact_manifest.json')['sha256'];files={}
 for name,h in manifest.items():
  path=engine.inside(run,name);files[name]=sha(path);require(files[name]==h,'Artifact hash changed: '+name)
 identity=read(run/'evaluation_protocol.json');protocol=read(run/'runtime_protocol.json');result=read(run/'result.json')
 require(result['evaluation_identity']==identity,'Result identity changed')
 for name in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim','whole_robot_pure_neural_episode'):
  require(result[name]is False,'Original False outcome changed: '+name)
 require(identity['policy']=='v8'and identity['episode_name']==RUN_NAME and identity['actor_feature_count']==284,'Wrong behavior tuple')
 require(identity['expert_hidden_or_labels_in_actor']is False and identity['executor_identity_in_actor_features']is False,'Actor input isolation changed')
 require(identity['left_policy_begins_at_reset_step']==0 and identity['actor_hidden_reset_count']==1 and result['left_teacher_position_action_steps']==0,'Not a pure left from-reset evaluation')
 for key,h in [('evaluated_policy_checkpoint_sha256',CHECKPOINT),('qualification_sha256',QUALIFICATION),('training_authorization_sha256',AUTHORIZATION)]:require(identity[key]==h,'Behavior binding changed: '+key)
 snapshots={}
 for folder,hashes in [('evaluation_source_snapshot',identity['source_sha256']),('source_snapshot',protocol['source_sha256'])]:
  for name,h in hashes.items():
   path=engine.inside(run,folder+'/'+name);require(sha(path)==h,'Snapshot changed: '+str(path));snapshots[folder+'/'+name]=h
 require(snapshots['evaluation_source_snapshot/state_hold_increment_eval_runtime_v8.py']==RUNTIME_SHA and snapshots['evaluation_source_snapshot/evaluate_state_hold_increment_v8.py']==ENTRY_SHA,'Actual control loop changed')
 physics=read(run/'physics_identity.json');family=core.physics_fingerprint(physics)
 require(family==identity['actual_physics_family_sha256']==identity['expected_physics_family_sha256']==trainer.FAMILY_SHA,'Physics domain differs')
 qpath=run/'reviewed_transition_qualification.json';apath=run/'reviewed_training_authorization.json';q=read(qpath)
 directories=[Path(d['evidence']['directory'])for d in q['datasets']]
 model,metadata=trainer.load_checkpoint(run/'checkpoints/evaluated_policy.pt',run/'checkpoints/frozen_v4_baseline.pt',directories,
  expected_physics_family=family,expected_checkpoint_sha256=CHECKPOINT,qualification_path=qpath,expected_qualification_sha256=QUALIFICATION,
  authorization_path=apath,expected_authorization_sha256=AUTHORIZATION)
 require(metadata['hidden_size']==128 and metadata['training_config']['seed']==20261105,'Wrong capacity or initialization')
 require(len(metadata['source_sha256'])==18 and metadata['source_sha256']==q['source_sha256'],'Incomplete frozen graph')
 for name,h in metadata['source_sha256'].items():require(snapshots['evaluation_source_snapshot/'+name]==h,'Missing trained dependency')
 for key,mkey,value in [('public_residual_gate_contract','gate_contract',policy.PUBLIC_GATE_CONTRACT),('action_contract','action_contract',policy.ACTION_CONTRACT),('training_classification_contract','classification_contract',policy.CLASSIFICATION_CONTRACT)]:
  require(identity[key]==metadata[mkey]==value,'Model/runtime action contract changed')
 names=metadata['joint_names'];left=metadata['output_indices'];params=read(run/'projection_parameters.json')
 baseline=torch.load(run/'checkpoints/frozen_v4_baseline.pt',map_location='cpu',weights_only=True)
 runtime.validate_joint_contract(metadata,baseline['metadata'],params['joint_names'],left)
 require(identity['update_bound_joint_names']==metadata['left_joint_names'] and identity['update_bound_rad']==metadata['update_bound_rad'] and identity['update_anchor_is_previous']==metadata['update_anchor_is_previous'],'Recorded model bounds/anchors differ')
 projector=engine.verified_projector(params,{'source_sha256':protocol['source_sha256'],'physics_identity':physics})
 tensors={k:torch.tensor(params[k],dtype=torch.float32)for k in ('limits','scales','step_limit','command_velocity_limits')}
 def project(previous,native):
  requested=previous+tensors['scales']*native
  target,hit=projector(previous,requested,tensors['limits'],tensors['step_limit'],tensors['command_velocity_limits'],params['control_dt'])
  return requested,target,hit
 records,abort,final=load_completed_records(run,result);require(len(records)==1803,'Wrong completed run length')
 rawrows=[engine.parse_line(line)for line in(run/'trajectory.jsonl').read_text().splitlines()]
 require(len(rawrows)==len(records),'Actual trajectory count differs')
 initial=read(run/'initial_state.json');require(initial['oracle_env_inputs']['episode_length']==0,'Initial history is not reset')
 oldpost=initial;oldteacher=None;oldmetrics=initial['oracle_env_inputs']['metrics'];oldsettings=initial['control_settings'];seen=Counter();gains=0;memory_changes=0
 model.cpu().eval();actor=policy.EpisodeActor(model,left);actor.reset(RUN_NAME)
 captured=[];original_forward=model.forward_with_diagnostics
 def capture(x,previous,hidden=None):
  output=original_forward(x,previous,hidden)
  captured.append((output[0].detach().clone(),{k:v.detach().clone()for k,v in output[3].items()}))
  return output
 model.forward_with_diagnostics=capture
 confusion=np.zeros((3,3),dtype=np.int64);all_confusion=np.zeros((3,3),dtype=np.int64);sumsq=0.;maxerror=0.;first_error=None;errors=[];mode_counts=Counter()
 branchpath=a.output/'branch_replay.jsonl';first_nonbase=None
 with branchpath.open('x',encoding='utf-8',newline='\n')as detail:
  for step,(record,raw)in enumerate(zip(records,rawrows)):
   pre=record['actor_observation_pre'];post=record['actor_observation_post'];audit=record['original_controller_audit'];execution=record['execution'];label=record['counterfactual_expert_projection_diagnostic_only']
   require(record['left_teacher_position_action_executed']is False and execution['left_teacher_position_action_executed']is False and record['corrective_labels_qualified_for_training']is False,'Original intervention/label flags differ')
   require(raw['step']==step==record['environment_episode_counter_before_step'],'Actual action counter differs')
   for key in core.STATE_FIELDS:require(core.canonical(pre[key])==core.canonical(oldpost[key]),'Actual input history gap: '+key)
   require(audit['query_index']==step and audit['failure_reason']is None and audit['oracle_env_inputs']['episode_length']==step,'Returned expert query counter differs')
   require(core.canonical(audit['oracle_env_inputs']['metrics'])==core.canonical(oldmetrics),'Expert did not query preceding actual metrics')
   if oldteacher is not None:require(core.canonical(audit['teacher_state_before_sync'])==core.canonical(oldteacher),'Expert memory history gap')
   engine._sync(audit,names,pre['applied_target']);memory_changes+=len(audit['execution_memory_changes'])
   schedule=audit['public_same_action_schedule'];phase=pre['phase']
   require(phase==raw['phase']==schedule['phase']==audit['teacher_state_after_query']['name'],'Same-action phase differs')
   require(pre['phase_progress']==schedule['phase_progress']==min(1.,seen[phase]/max(1,protocol['phase_counts'][phase]-1)),'Public phase clock differs');seen[phase]+=1
   before=audit['control_settings_before'];after=audit['control_settings_after'];require(before==oldsettings,'Gain continuity gap')
   changed=before!=after;require(audit['teacher_runtime_control_setting_change']==changed,'Gain event differs')
   if changed:
    require(engine.runtime.BaselineRuntime.validate_control_change(SimpleNamespace(joint_names=names),before,after,audit['teacher_state_before_sync'],audit['teacher_state_after_query']),'Unapproved gain event');gains+=1
   old=torch.tensor(pre['applied_target'],dtype=torch.float32).reshape(1,56);native=torch.tensor(audit['normalized_action'],dtype=torch.float32).reshape(1,56)
   requested,expert,hit=project(old,native)
   require(bits(requested[0],label['requested_target'])and bits(requested[0],audit['requested_target'])and bits(expert[0],label['projected_target']),'Returned expert projection changed')
   require(label['projection_soft_limit_hit']==hit[0].tolist()and bits(label['projection_previous_applied_target'],pre['applied_target']),'Expert projection previous state differs')
   require(label['projection_parameters_file']=='projection_parameters.json'and label['projection_parameters_file_sha256']==files['projection_parameters.json']and label['projection_source_sha256']==params['source_sha256'],'Returned label projection provenance differs')
   require(label['projected_label_dtype']=='torch.float32'and label['projected_label_scope']=='expert_from_actual_prestate_not_student_actual_action','Returned label scope differs')
   safe_input=runtime.actor_input(pre,schedule);x=policy.features(safe_input)
   require(bits(x,record['actual_actor_features']),'Actual 284 features differ')
   prediction=actor.act(safe_input,episode_id=RUN_NAME,step=step)
   require(len(captured)==step+1 and bits(prediction,record['predicted_left_target']),'Actual causal actor prediction differs')
   proposal,d=captured[-1];require(bits(proposal[0,0],prediction),'Captured actor proposal differs')
   composed=native.clone();composed[:,left]=(torch.tensor(prediction,dtype=torch.float32).reshape(1,28)-old[:,left])/tensors['scales'][:,left]
   require(bits(composed[0],execution['native_action'])and bits(composed[0],raw['native_action']),'Executed native action/right28 differs')
   actualrequest,actual,actualhit=project(old,composed)
   require(bits(execution['requested_target'],actualrequest[0])and execution['soft_limit_hit']==actualhit[0].tolist(),'Actual execution projection differs')
   require(all(bits(actual[0],v)for v in(post['applied_target'],raw['applied_target'],execution['expected_applied_target'])),'Actual applied target differs')
   for key in ('q','joint_velocity','cube_position','cube_quaternion_wxyz','finger_contact_impulses','finger_contact_points'):require(core.canonical(post[key])==core.canonical(raw[key]),'Actual post-state streams differ')
   require(record['raw_environment_termination']==raw['termination']and not raw['termination']['raw_done'],'Actual raw done differs or continued past terminal')
   with torch.no_grad():
    previous=old[:,left].reshape(1,1,28);target=expert[:,left].reshape(1,1,28)
    project_left=lambda value:policy.project_left_targets(value,previous,params,left,projector)
    base=project_left(d['baseline_proposal']);hold=project_left(previous);emitted=project_left(proposal)
    require(bits(emitted[0,0],actual[0,left]),'Actual left execution and branch projection differ')
    classes=policy.classify_branch_labels(target,base,hold);active=d['public_phase_active'];confident=classes['classification_mask']
    difference=(emitted.double()-target.double()).abs()[0,0];sumsq+=float(difference.square().sum());maxerror=max(maxerror,float(difference.max()))
   wrong=(d['predicted_class']!=classes['class_targets'])&confident&active
   for j in range(28):
    truth=int(classes['class_targets'][0,0,j]);pred=int(d['predicted_class'][0,0,j]);mode_counts[(phase,pred)]+=1
    if bool(confident[0,0,j]):
     all_confusion[truth,pred]+=1
     if bool(active[0,0,j]):confusion[truth,pred]+=1
    if bool(wrong[0,0,j]):errors.append({'step':step,'phase':phase,'joint':names[left[j]],'expert_class':truth,'predicted_class':pred,'expert_projected':float(target[0,0,j]),'previous':float(previous[0,0,j]),'base_projected':float(base[0,0,j]),'actual_projected':float(emitted[0,0,j]),'logits':d['mode_logits'][0,0,j].tolist()})
   if first_error is None and bool((difference>1e-6).any()):first_error={'step':step,'phase':phase,'joints':[names[left[j]]for j in torch.nonzero(difference>1e-6).flatten().tolist()]}
   if first_nonbase is None and not bits(emitted,base):first_nonbase={'step':step,'phase':phase}
   row={'step':step,'phase':phase,'eligible_transition':False,'training_qualification_granted':False,'public_phase_active':bool(active.any()),
    'predicted_class':d['predicted_class'][0,0].tolist(),'mode_logits':d['mode_logits'][0,0].tolist(),'expert_class':classes['class_targets'][0,0].tolist(),
    'expert_classification_mask':confident[0,0].tolist(),'expert_exact_BASE':classes['exact_base'][0,0].tolist(),'expert_exact_HOLD_priority':classes['exact_hold'][0,0].tolist(),
    'previous_left':previous[0,0].tolist(),'base_projected':base[0,0].tolist(),'hold_projected':hold[0,0].tolist(),'expert_projected':target[0,0].tolist(),
    'actual_model_projected':emitted[0,0].tolist(),'update_anchor':d['update_anchor'][0,0].tolist(),'bounded_update':d['bounded_update'][0,0].tolist()}
   detail.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
   oldpost=post;oldteacher=audit['teacher_state_after_query'];oldmetrics=record['raw_environment_metrics'];oldsettings=after
 require(actor.next_step==len(records)==len(captured),'GRU call total differs')
 require(gains==result['original_teacher_gain_schedule_events']==final['original_teacher_gain_schedule_events'],'Final gain count differs')
 require(sources==trainer.source_hashes()|{Path(m.__file__).name:sha(m.__file__)for m in(runtime,engine,core)}|{Path(__file__).name:sha(__file__)},'Source changed during replay')
 for name,h in files.items():require(sha(engine.inside(run,name))==h,'Actual evidence changed during replay')
 report={'schema':'single_pinned_wide128_v8_actual_causal_actor_diagnostic_v1','passed':True,'run':str(run),'original_result':result,
  'checkpoint_sha256':CHECKPOINT,'qualification_sha256':QUALIFICATION,'authorization_sha256':AUTHORIZATION,'source_sha256':sources,
  'actual_artifact_manifest_sha256':sha(run/'artifact_manifest.json'),'actual_files_sha256':files,'actual_source_snapshots_sha256':snapshots,
  'actual_completed_steps':len(records),'returned_teacher_queries':len(records),'actor_replay_calls':len(captured),'extra_recurrent_forward_calls_for_diagnostics':0,
  'all_actual_features_predictions_native_actions_projected_targets_bitwise_equal':True,'actor_hidden_resets':1,'gain_events':gains,'memory_sync_changed_keys':memory_changes,
  'excluded_unreturned_query':abort,'left_joint_names':[names[i]for i in left],'class_names':['BASE','HOLD','UPDATE'],
  'branch_detail_file':branchpath.name,'branch_detail_sha256':sha(branchpath),'active_confident_confusion_true_rows_predicted_columns':confusion.tolist(),
  'all_confident_confusion_true_rows_predicted_columns':all_confusion.tolist(),'active_class_errors':len(errors),'class_error_records':errors,
  'actual_decoded_vs_returned_expert_rmse_rad':(sumsq/(28*len(records)))**.5,'actual_decoded_vs_returned_expert_max_rad':maxerror,
  'first_projected_nonBASE':first_nonbase,'first_expert_error_gt_1e_6':first_error,'registry_modified':False,'training_qualification_granted':False,
  'original_result_modified':False,'training_started':False,'isaac_started':False,'original_safety_prefix_qualification_recomputed':False,
  'scope':'Single completed actual behavior causal replay and counterfactual classification diagnostic only. No new training eligibility, success or registry admission. The exception at step1803 had no returned query, label, actor call or physics step.'}
 write(a.output/'report.json',report)
 print(json.dumps({'report':str(a.output/'report.json'),'report_sha256':sha(a.output/'report.json'),'branch_sha256':sha(branchpath),'frames':len(records),'class_errors':len(errors),'first_error':first_error,'rmse':report['actual_decoded_vs_returned_expert_rmse_rad']},indent=2),flush=True)

if __name__=='__main__':main()
