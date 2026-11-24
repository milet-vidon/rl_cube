"""Freeze V8 evidence inventory without treating a running model as final."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103';CODE=ROOT/'revotron_cube_rl'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=(20261104,20261105),required=True)
    a=parser.parse_args();seed=a.seed
    run=CODE/'runs'/f'teacher_hold_increment_v8_seed{seed}_20261104'
    authpath=WORK/f'v8_seed{seed}_root_training_authorization_20261104.json';auth=read(authpath)
    reviewpath=WORK/'v8_root_two_training_review_20261104.json';review=read(reviewpath)
    assert any(job['seed']==seed and job['authorization_sha256']==sha(authpath)and (CODE/job['run']).resolve()==run for job in review['jobs'])
    assert auth['training_config']['seed']==seed
    output=WORK/f'v8_seed{seed}_training_archive_inventory_20261104.json'
    assert not output.exists(),'Preserve prior inventory'
    fixed={}
    def add(path,category,digest=None):
        path=Path(path);assert path.is_file()and not path.is_symlink(),path
        actual=sha(path);assert digest is None or actual==digest,path
        relative=str(path.resolve().relative_to(ROOT));assert relative not in fixed,'Duplicate fixed file: '+relative
        fixed[relative]={'sha256':actual,'bytes':path.stat().st_size,'category':category}
    for name,digest in auth['source_sha256'].items():
        path=WORK/name
        if not path.is_file():path=CODE/name
        add(path,'frozen_training_source',digest)
    files={
      'host_contract_tests':('test_hold_increment_policy_v8.py','test_train_hold_increment_v8.py',
         'test_train_hold_increment_v8_peer.py','test_trusted_eval_dataset_registry_v1.py',
         'test_completed_hold_increment_v8_audit.py','test_completed_hold_increment_journal_v8.py'),
      'independent_and_root_review':('v8_trainer_independent_review_20261104.json','v8_trainer_independent_host_tests_20261104.log',
         'build_v8_trainer_independent_review_20261104.py','trusted_registry_v1_independent_review_20261104.json',
         'v8_root_two_training_review_20261104.json',authpath.name,'launch_hold_increment_v8_training_20261104.sh'),
      'independent_coverage_and_parameterization_diagnostics':('v8_eight_history_branch_coverage_20261104_01.json',
         'v8_finite_tanh_witness_diagnostic_20261104.json','diagnose_tanh_update_witness_v8.py'),
      'frozen_evaluation_protocol_and_contracts':('state_hold_increment_eval_runtime_v8.py','evaluate_state_hold_increment_v8.py',
         'test_state_hold_increment_eval_v8.py','test_state_hold_increment_eval_v8_delta.py','test_state_hold_increment_eval_v8_prepare.py',
         'test_v8_eval_prepare_independent.py','v8_eval_entry_independent_review_20261104.json',
         'v8_eval_independent_host_tests_v2.log','v8_eval_independent_host_tests.log','build_v8_eval_independent_review.py',
         'launch_hold_increment_v8_eval_20261104.sh'),
      'completed_review_tool_sources':('audit_completed_hold_increment_v8.py','audit_completed_hold_increment_journal_v8.py'),
    }
    for category,names in files.items():
        for name in names:add(WORK/name,category)
    preflight=WORK/'v8_trainer_preflight_20261104_01'
    for path in sorted(preflight.rglob('*')):
        if path.is_file():add(path,'reviewed_full_data_device_preflight')
    qpath=preflight/'qualification_candidate.json';q=read(qpath)
    assert sha(qpath)==auth['qualification_sha256']=='810b793afb4b827bfcea72778c66f55883552f143b0fda4bd2245eb5aaa2d1b9'
    assert q['training_authorized']is False and q['total_eligible_frames']==7681
    assert q['source_sha256']==auth['source_sha256']
    refs=[{'run':str(Path(d['evidence']['directory']).relative_to(ROOT)),'evidence':d['evidence'],
        'original_result_flags':d['original_result_flags'],'raw_history_copy_policy':'referenced by immutable evidence, not duplicated'}for d in q['datasets']]
    value={'schema':'v8_frozen_evidence_and_pending_training_inventory_v1','training_run':str(run.relative_to(ROOT)),
        'training_seed':seed,'physical_history_seed':9501,'root_authorization_sha256':sha(authpath),
        'root_authorized_configuration':auth['training_config'],'qualification_sha256':sha(qpath),
        'training_run_complete_at_inventory_time':(run/'result.json').exists(),'final_model_archived':False,
        'frozen_files':fixed,'source_dataset_references':refs,
        'pending_completion_requirements':[
          'All authorized epochs and eight unique histories per epoch; only declared candidate epochs selected with exact CPU ordering.',
          'Final model.pt and separate last_state.pt hashes match result; full final CPU model replay agrees with saved metrics.',
          'last_state.pt contains final AdamW moment/step tensors and Python/NumPy/Torch CPU/CUDA RNG bytes; no exact-resume claim.',
          'Every regular file in completed run, training log and independent completion audits preserved.',
          'Qualified model requires actual frozen evaluation checkpoint preflight; unqualified result remains an explicit failure.',
          'Raw/archive/decompressed bytes all hash-verified before review.'],
        'training_log':str((WORK/f'v8_seed{seed}_gpu_train.log').relative_to(ROOT)),
        'original_result_files_modified':False,'training_resume_performed':False,'exact_resume_demonstrated':False,
        'closed_loop_success_claim':False,'independent_generalization_claim':False,'eligible_for_rl_episode_training':False}
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'inventory':str(output),'sha256':sha(output),'frozen_files':len(fixed),
        'frozen_bytes':sum(r['bytes']for r in fixed.values()),'training_complete':value['training_run_complete_at_inventory_time']}))


if __name__=='__main__':main()
