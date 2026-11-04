"""Freeze completed V9 evidence inventory; trainer-author preparation, no Git."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103';CODE=ROOT/'revotron_cube_rl'
Q_SHA='6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7'
CANDIDATE_SHA='09701aa0fe403ccef7a5e7f39e0860d5900333e7fbe2d914f1f879c81c88ed47'
AUTH={64:'635d6f737a6b174d6e8c73d56a6a0f1708c618643f5e1cbc12a316f95ee5e186',128:'af0f5522650231ad25d40aa5edaf4a02b34eb74617a263e3fed9645721c4561a'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--hidden-size',type=int,choices=(64,128),required=True);a=p.parse_args();width=a.hidden_size
 prefix=f'v9_hidden{width}_seed20261104';run=CODE/'runs'/f'teacher_arm_hold_increment_v9_hidden{width}_seed20261104_20261104'
 authpath=WORK/(prefix+'_root_training_authorization_20261104.json');assert sha(authpath)==AUTH[width];auth=read(authpath)
 assert auth['training_config']['hidden_size']==width and auth['training_config']['seed']==20261104 and auth['training_config']['epochs']==5000
 result=read(run/'result.json');assert result['checkpoint_sha256']==sha(run/'model.pt')and result['last_state_sha256']==sha(run/'last_state.pt')
 qpath=run/'qualification_candidate.json';q=read(qpath);assert sha(qpath)==Q_SHA==auth['qualification_sha256']
 assert q['training_authorized']is False and q['total_eligible_frames']==9523 and len(q['datasets'])==9 and q['source_sha256']==auth['source_sha256']
 candidate=ROOT/'docs/evidence/arm_hold_increment_v9_candidate_20261104/archive_manifest.json';assert sha(candidate)==CANDIDATE_SHA
 c=read(candidate);assert len(c['entries'])==75
 fixed={}
 def add(path,category,expected=None):
  path=Path(path);assert path.is_file()and not path.is_symlink();relative=str(path.resolve().relative_to(ROOT));h=sha(path)
  assert expected is None or h==expected,path
  assert relative not in fixed,'Duplicate inventory source'
  fixed[relative]={'sha256':h,'bytes':path.stat().st_size,'category':category}
 for entry in c['entries']:add(ROOT/entry['source'],'shared_frozen_v9_candidate_evidence',entry['source_sha256'])
 add(candidate,'previously_cross_reviewed_frozen_candidate_manifest',CANDIDATE_SHA)
 add(WORK/'arm_hold_increment_v9_candidate_20261104_archive_cross_agent_review.json','candidate_cross_agent_review','9cbcae313fb1b37968085a6aa67d974c66c9f22f0be9444fa439285db35449b2')
 for name in ('audit_completed_arm_hold_increment_journal_v9.py','test_completed_arm_hold_increment_journal_v9.py','build_v9_completed_journal_audit_20261104.py'):
  add(WORK/name,'trainer_author_completion_inventory_tool')
 for name,h in auth['source_sha256'].items():assert sha(CODE/name)==h
 refs=[{'run':str(Path(d['evidence']['directory']).relative_to(ROOT)),'evidence':d['evidence'],'original_result_flags':d['original_result_flags'],'raw_history_copy_policy':'immutable references to already archived actual histories; not duplicated here'}for d in q['datasets']]
 output=WORK/(prefix+'_training_archive_inventory_20261104.json');assert not output.exists()
 value={'schema':'v9_completed_training_frozen_evidence_inventory_author_preparation_v1','training_run':str(run.relative_to(ROOT)),
  'training_seed':20261104,'hidden_size':width,'physical_history_seed':9501,'root_authorization_sha256':sha(authpath),'root_authorized_configuration':auth['training_config'],
  'qualification_sha256':Q_SHA,'source_dataset_references':refs,'frozen_files':fixed,'original_candidate_archive_manifest_sha256':CANDIDATE_SHA,
  'training_run_complete_at_inventory_time':True,'final_model_archived':False,'reviewer_role':'trainer_author_selfcheck_pending_cross_agent_review',
  'training_log':str((WORK/(prefix+'_training_20261104.log')).relative_to(ROOT)),
  'training_resume_performed':False,'exact_resume_demonstrated':False,'closed_loop_success_claim':False,'independent_generalization_claim':False,'eligible_for_rl_episode_training':False}
 output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'inventory':str(output),'sha256':sha(output),'frozen_files':len(fixed)}))
if __name__=='__main__':main()
