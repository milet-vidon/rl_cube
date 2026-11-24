"""Two exact failed V8 runs: reviewed-transition candidates before an unreturned query.

No existing registry or trainer imports this module. This is not a training
authorization. Original result flags and the unknown failed-query state remain
unchanged. A separate reviewed qualification is required before any BC use.
"""
from pathlib import Path
from types import MappingProxyType,ModuleType
import argparse,hashlib,json
import numpy as np
import torch
import trusted_eval_dataset_registry_v2 as prior

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103'
SCHEMA='two_exact_unreturned_teacher_exception_prefix_bc_candidates_v1'
PRIOR_SHA='e076e877897420e651f691a10419049815d094c27b8196afeec77320d938d89e'
FAILURE='evaluation_protocol_error:Post-turn correction exceeded bounded pose envelope'
QSHA='810b793afb4b827bfcea72778c66f55883552f143b0fda4bd2245eb5aaa2d1b9'
RUNS=MappingProxyType({
 'left_hold_increment_v8_seed20261105_eval_9501_20261104':{
  'steps':1725,'hidden_size':64,'checkpoint':'56eba2c26ceff4d85513da3970bfa11ec83d87448f47e9c7b3abafc878f74456',
  'authorization':'d6d5961f23215140a21658c51c6105e0ad24eb7db1a2df0c65f688708335e6ef',
  'hashes':{'artifact_manifest.json':'1cbdc79be5e75df972f60ca17c034574fee7cd198330e2e3a11570a1e2f060b7',
   'result.json':'25f855a6e81d1c8851b1b7b66afbf6051d1476b1c7100fecdcf111df94861e0f',
   'evaluation_transitions.jsonl':'9b0ed86511ba5eff32cf5e7ee2befb06b2237c2a1a236d590bea1804303dc1eb',
   'trajectory.jsonl':'f8c5da546a5b4bd87b14ece3ffff4ba52d7218c210c238dc2e9a733d2cb1eacd',
   'evaluation_protocol.json':'3985a722e7d257c39824517fd69110f676fbf0fa4eba23b8e34150a984b49de1',
   'physics_identity.json':'57e21f505e1833564889b4b776aac4e04e5eef11d1436e959761465928aae8c7'},
  'actor_evidence':'fresh_single_pass_required_no_completed_prior_proof'},
 'left_hold_increment_v8_wide128_seed20261105_eval_9501_20261104':{
  'steps':1803,'hidden_size':128,'checkpoint':'cba7ebc92496e6e32d33e83f29357677fcbf8d3037f456403a3734093dce82bb',
  'authorization':'0a5a9963c32a062c0cf13e0085ee3c25463bb0a7f62b4873a8717b7611147a91',
  'hashes':{'artifact_manifest.json':'7a1835b5e98c6808666aeb3b0333d2e2697f3dbc45cb885d083b15604d05516b',
   'result.json':'8ac1cbe8d3225d841a909d293f1211da619dc71cb6591806c89674a047ddfb7c',
   'evaluation_transitions.jsonl':'54cd8688389db2ab48fd98d331f1c3c6a498d9165957412394f0730bc4aff505',
   'trajectory.jsonl':'ae0066e57d1fe0b81ac7f71051c759fff93ad33bcdc7aa379b287632a1919e29',
   'evaluation_protocol.json':'29ea6b1c9c6964e84d0b7939920dd40bc5fe6ac68a486b4d93d171f67c409f4b',
   'physics_identity.json':'537007f3847c6acf8285115214d223bec27c131c4c87ab6ccb67e0635a5e74ba'},
  'actor_evidence':'pinned_completed_single_pass_proof_reused',
  'actor_report':'v8_wide128_seed20261105_actual_actor_diagnostic_20261104/report.json',
  'actor_report_sha256':'ab7d6a53bba933d24f9ce9542cfc427955155b48f59d86030a14060c6468634f',
  'actor_branch_sha256':'1e43ea5cf0a3b00f63cec65ebf5c622a7a099ee955d725cde7e17ee3a04e3cd1',
  'actor_script_sha256':'3682b56de5e82c50c2699a11b60c5b13f2d4074adccd4ed5419c3ee03759f74d'}
})
ASSETS={
 'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda':'1ea674035491ae19875c61813ef8cfb1fd26c4b0b58de3ae3a2a3f04b741a143',
 'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd':'c066a65ff5ac77a8fb0a5fb3658c71f151b64777cdc9a8f66a0739d4033170a2',
 'isaac_mujoco_cube/cube_model.usda':'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'}
require=prior.require;read=prior.read_json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_unreturned_abort(record,count,result):
    """No returned query is inferred; every absent value stays absent."""
    require(record['record_type']=='abort'and type(record['step'])is int and record['step']==count,'Wrong abort boundary')
    require(record['reason']==result['failure']==FAILURE,'Wrong exception outcome')
    require(record['unfinished_decision']=={'step':count,'record_type':'decision_attempt','left_teacher_position_action_executed':False},
            'Unreturned query must have exactly the original three fields; no invented audit/label/features/action')
    require(type(record['unfinished_decision']['step'])is int and record['unfinished_decision']['left_teacher_position_action_executed']is False,
            'Wrong unreturned decision field types')
    for key in ('physical_steps','environment_episode_counter','fully_audited_transition_steps','action_execution_dispatch_attempts','trajectory_rows_recorded'):
        require(type(record[key])is int and record[key]==count,'Abort count differs: '+key)
    require(record['physics_step_completed_for_unfinished_decision']is False and record['execution_dispatch_without_counter_advance']is False,
            'Unreturned query dispatched physics')
    require(record['invalid_evidence_fields']==[],'Invalid abort evidence')
    return {'step':count,'reason':record['reason'],'type':'unreturned_teacher_exception_no_label',
        'teacher_query_returned':False,'teacher_after_state_known':False,'actor_features_or_prediction_present':False,
        'expert_label_present':False,'physics_executed':False,'excluded_from_bc_supervision':True}

def scan_exact_boundary(run,expected_steps):
    result=read(run/'result.json');count=0;abort=None;final=None
    with (run/'evaluation_transitions.jsonl').open()as stream:
        for line in stream:
            require(line.endswith('\n'),'Incomplete transition line')
            r=prior.prior.frozen.parse_line(line);require(final is None,'Record after final')
            if r['record_type']=='executed_transition':
                require(abort is None and r['step']==count and r['transition_finished']is True,'Noncontiguous or post-abort physics')
                require(r['terminal']is False and r['failure_reason']is None and r['original_wrist_gate_qualified']is False,'Unexpected earlier terminal')
                require(not r['invalid_evidence_fields'],'Invalid executed evidence');count+=1
            elif r['record_type']=='abort':
                require(abort is None,'Repeated abort');abort=check_unreturned_abort(r,count,result)
            else:
                require(r['record_type']=='evaluation_final'and abort is not None and not r['invalid_evidence_fields'],'Unknown record or premature final')
                final=r['summary']
    require(count==expected_steps and abort is not None and final is not None,'Incomplete pinned exception prefix')
    for key in ('steps','physical_steps','environment_episode_counter','fully_audited_transition_steps',
                'action_execution_dispatch_attempts','trajectory_rows_recorded','teacher_query_count'):
        require(type(result[key])is int and type(final[key])is int and result[key]==final[key]==count,'Final count differs: '+key)
    require(final['failure']==result['failure']==FAILURE and final['original_wrist_gate_qualified']is False,'Wrong final failure')
    return abort

def trust_run(directory):
    run=Path(directory).resolve();require(run.name in RUNS,'Run outside two-case whitelist')
    require(run==ROOT/'revotron_cube_rl/runs'/run.name and not Path(directory).is_symlink(),'Noncanonical run directory')
    spec=RUNS[run.name]
    require(sha(prior.__file__)==PRIOR_SHA,'Frozen registry2 source changed');prior._sources()
    for name,h in spec['hashes'].items():require(sha(run/name)==h,'Pinned actual artifact changed: '+name)
    identity=read(run/'evaluation_protocol.json')
    require(identity['episode_name']==run.name and identity['policy']=='v8','Wrong exact episode identity')
    for key,h in (('evaluated_policy_checkpoint_sha256',spec['checkpoint']),('qualification_sha256',QSHA),
                  ('training_authorization_sha256',spec['authorization'])):
        require(identity[key]==h,'Wrong exact model/qualification/authorization tuple: '+key)
    physics=read(run/'physics_identity.json');assets={str(ROOT/n):h for n,h in ASSETS.items()}
    require(physics['assets_sha256']==assets,'Wrong original physics asset set')
    for name,h in ASSETS.items():
        p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==h,'Original asset bytes changed: '+name)
    return run,spec

def private_trust_registry():
    """Private module instance only; the frozen global registry is never changed."""
    require(sha(prior.__file__)==PRIOR_SHA,'Registry2 bytes changed')
    module=ModuleType('_exception_prefix_private_v8_trust')
    module.__file__=prior.__file__
    exec(compile(Path(prior.__file__).read_text(),prior.__file__+'::private_exception_trust','exec'),module.__dict__)
    module.V8_TRUSTED=MappingProxyType({v['checkpoint']:v['authorization']for v in RUNS.values()})
    return module

class RecordedPredictionProof:
    """Reuse an immutable completed actor replay, with zero additional GRU calls."""
    def __init__(self,run,spec,evidence):
        report_path=WORK/spec['actor_report'];require(sha(report_path)==spec['actor_report_sha256'],'Actor proof changed')
        report=read(report_path);branch=report_path.parent/'branch_replay.jsonl'
        require(sha(branch)==report['branch_detail_sha256']==spec['actor_branch_sha256'],'Actor mode proof changed')
        script=WORK/'replay_wide128_v8_actual_diagnostic_20261104.py'
        require(sha(script)==spec['actor_script_sha256'] and report['source_sha256'][script.name]==spec['actor_script_sha256'],'Actor audit source changed')
        require(report['passed']is True and report['run']==str(run),'Actor proof is not a completed exact run')
        require(report['all_actual_features_predictions_native_actions_projected_targets_bitwise_equal']is True,'Actor proof lacks exact prediction verification')
        require(report['actual_completed_steps']==report['returned_teacher_queries']==report['actor_replay_calls']==spec['steps'],'Actor proof count differs')
        require(report['actor_hidden_resets']==1 and report['extra_recurrent_forward_calls_for_diagnostics']==0,'Actor replay reset/call semantics differ')
        require(report['checkpoint_sha256']==spec['checkpoint'] and report['qualification_sha256']==QSHA and report['authorization_sha256']==spec['authorization'],'Actor proof behavior tuple differs')
        require(report['actual_artifact_manifest_sha256']==evidence['artifact_manifest_sha256']and report['actual_files_sha256']==evidence['files_sha256'],
                'Actor proof not bound to all current actual file bytes')
        require(report['actual_source_snapshots_sha256']==evidence['source_snapshot_sha256'],'Actor proof source snapshots differ')
        for name,h in report['actual_files_sha256'].items():
            require(sha(prior.prior.frozen.inside(run,name))==h,'Actor proof current input bytes changed: '+name)
        self.values=[]
        with (run/'evaluation_transitions.jsonl').open()as stream:
            for line in stream:
                r=prior.prior.frozen.parse_line(line)
                if r['record_type']=='executed_transition':
                    self.values.append((np.asarray(r['actual_actor_features'],np.float32),np.asarray(r['predicted_left_target'],np.float32)))
        require(len(self.values)==spec['steps'],'Actor proof record length differs')
        self.episode=run.name;self.next_step=None
        self.proof={'mode':'reuse_pinned_complete_actor_replay','report_sha256':spec['actor_report_sha256'],
                    'branch_sha256':spec['actor_branch_sha256'],'prior_gru_calls':spec['steps'],'new_gru_calls':0}
    def reset(self,episode_id):
        require(self.next_step is None and episode_id==self.episode,'One reset for exact actor proof episode required');self.next_step=0
    def act(self,observation,*,episode_id,step):
        require(episode_id==self.episode and step==self.next_step and step<len(self.values),'Actor proof out-of-order access')
        x=prior.prior.frozen.policy.features(observation);features,prediction=self.values[step]
        require(prior.prior.frozen.bits_equal(x,features),'Actor proof features differ from actual causal state')
        self.next_step+=1;return prediction.copy()

ABORT_BEFORE="""                audit=decision['original_controller_audit']
                require(audit['query_index']==query_count and audit['failure_reason']==record['reason']==result['failure'],'Abort query/reason mismatch')
                require(audit['oracle_env_inputs']['episode_length']==len(xs),'Abort is not on actual final state')
                _sync(audit,names,old_post['applied_target'])
                if old_teacher is not None:require(core.canonical(audit['teacher_state_before_sync'])==core.canonical(old_teacher),'Teacher state discontinuity at abort')
                query_count+=1;abort_seen=True;ended=True;terminal_reason=record['reason']
                excluded.append({'step':record['step'],'reason':record['reason'],'type':'unexecuted_abort_no_label'});continue"""
ABORT_AFTER="""                excluded.append(check_unreturned_abort(record,len(xs),result))
                require(query_count==len(xs),'Unreturned query cannot increment returned-query count')
                abort_seen=True;ended=True;terminal_reason=record['reason'];continue"""
ACTOR_BEFORE="""    behavior_model=restore_recorded_model(run,identity,saved)
    if successful_v4:
        import state_residual_eval_runtime_v5 as old_eval
        require(sha(old_eval.__file__)=='90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64','V4 behavior actor source changed')
        behavior=old_eval.V4EpisodeActor(behavior_model,left)
    else:behavior=policy.EpisodeActor(behavior_model,left)
    behavior.reset(identity['episode_name'])"""
ACTOR_AFTER="""    require(not successful_v4,'Exception candidate accepts only fixed V8 failures')
    behavior=make_verified_behavior(run,identity,saved,evidence,policy.EpisodeActor)
    behavior.reset(identity['episode_name'])"""
HOOKS=((ABORT_BEFORE,ABORT_AFTER),(ACTOR_BEFORE,ACTOR_AFTER))

def engine_source():
    source=prior.engine_source();changed=source
    for before,after in HOOKS:
        require(changed.count(before)==1,'Original exact exception/actor hook changed');changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(HOOKS):
        require(restored.count(after)==1,'Private inverse hook ambiguous');restored=restored.replace(after,before)
    require(restored==source,'Exception candidate inverse changed frozen predicates')
    return changed

def load_candidate(directory,baseline_saved,baseline_sha,expected_family=None):
    run,spec=trust_run(directory);abort=scan_exact_boundary(run,spec['steps']);trust=private_trust_registry()
    module=ModuleType('_exception_prefix_private_predicate_engine')
    module.__dict__.update(__file__=prior.prior.frozen.__file__,REGISTRY=MappingProxyType({'v8':'exact_failed_exception_candidate'}),
        admitted_successful_wrist=lambda identity,result:False,check_unreturned_abort=check_unreturned_abort)
    source=engine_source();exec(compile(source,__file__+'::private_frozen_predicates','exec'),module.__dict__)
    module.SCHEMA=SCHEMA;module.verify_evaluation_semantics=trust.verify_evaluation_semantics
    module.restore_recorded_model=trust.restore_recorded_model
    behavior=[]
    def make_behavior(actual,identity,saved,evidence,actor_type):
        require(saved['metadata']['hidden_size']==spec['hidden_size'],'Wrong pinned model width')
        if 'actor_report'in spec:actor=RecordedPredictionProof(actual,spec,evidence)
        else:
            actor=actor_type(trust.restore_recorded_model(actual,identity,saved),saved['metadata']['output_indices'])
            actor.proof={'mode':'fresh_complete_single_actual_history_replay','prior_completed_proof_available':False,
                         'new_gru_calls':spec['steps']}
        behavior.append(actor);return actor
    module.make_verified_behavior=make_behavior
    data=module.load_run(run,baseline_saved,baseline_sha,expected_family)
    require(len(behavior)==1 and behavior[0].next_step==spec['steps'],'Actor sequence did not cover every completed transition')
    require(len(data['x'])==spec['steps'] and bool(data['mask'].all()),'Exact two-run candidate contains unsafe/incomplete physical records')
    require(data['excluded_labels']==[abort],'Only the original unreturned query is excluded')
    for name,h in spec['hashes'].items():require(sha(run/name)==h,'Evidence changed during candidate audit')
    for field in ('files_sha256','source_snapshot_sha256'):
        for name,h in data['evidence'][field].items():
            require(sha(prior.prior.frozen.inside(run,name))==h,'Complete evidence changed during candidate audit: '+name)
    actual_plans={p.name:hashlib.sha256(prior.prior.frozen.core.canonical(read(p)).encode()).hexdigest()
                  for p in sorted((run/'plans').glob('*.json'))}
    require(actual_plans==data['evidence']['plan_sha256'],'Teacher plan inventory changed during candidate audit')
    trust_run(run)
    evidence=data['evidence'];evidence.update(exception_prefix_contract_sha256=sha(__file__),
        original_registry2_sha256=PRIOR_SHA,private_engine_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_frozen_engine_sha256=prior.ENGINE_SHA,exact_run_whitelist=run.name,
        actor_proof=behavior[0].proof,original_assets_sha256={str(ROOT/k):v for k,v in ASSETS.items()})
    data['audit']['behavior_predictions_replayed_bitwise_this_pass']='actor_report'not in spec
    data['audit']['behavior_predictions_replayed_bitwise']='actor_report'not in spec
    data['audit']['actual_predictions_verified_by_current_or_pinned_complete_prior_replay']=True
    data['audit']['actor_proof']=behavior[0].proof
    data.update(training_qualification_granted=False,eligible_for_rl_episode_training=False,
        current_V9_training_dataset_admission=False,requires_separate_root_review_and_qualification=True,
        exception_query_after_state_known=False,exception_query_label_fabricated=False,
        scope='Candidate BC transitions from original complete audited physical prefix; no current trainer admission or episode success')
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',choices=tuple(RUNS),required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(),'Preserve prior candidate report');torch.set_num_threads(1)
    baseline,base_sha=prior.trainer.prior.frozen.load_baseline(ROOT/'revotron_cube_rl/runs'/a.run/'checkpoints/frozen_v4_baseline.pt')
    data=load_candidate(ROOT/'revotron_cube_rl/runs'/a.run,baseline,base_sha,prior.trainer.FAMILY_SHA)
    report={k:v for k,v in data.items()if k not in ('x','previous_left','labels','mask','projector','executed_sources')}
    report.update(selected_completed_candidate_transitions=len(data['x']),all_original_expert_labels_retained=True,
        arrays_sha256={k:hashlib.sha256(np.asarray(data[k]).tobytes()).hexdigest()for k in ('x','previous_left','labels','mask')},
        actor_feature_count=284,right28_learned=False,training_started=False,isaac_started=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'report':str(a.output),'sha256':sha(a.output),'candidate_transitions':len(data['x']),
        'actor_proof':data['audit']['actor_proof'],'training_qualification_granted':False},indent=2))

if __name__=='__main__':main()
