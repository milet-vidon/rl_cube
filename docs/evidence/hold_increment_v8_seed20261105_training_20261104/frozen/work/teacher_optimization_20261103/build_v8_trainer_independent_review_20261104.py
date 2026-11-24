"""Bind the independent CPU trainer review to frozen source and retained log."""
from pathlib import Path
import hashlib,json
import train_hold_increment_v8 as trainer

WORK=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected='055f8f0d6ed724c2932a2effa59d402c2e22472491519f9a1b09fb553e5a42cb'
assert sha(trainer.__file__)==expected,'Trainer changed after independent review'
log=WORK/'v8_trainer_independent_host_tests_20261104.log'
text=log.read_text();assert 'Ran 16 tests' in text and text.rstrip().endswith('OK')
review={'schema':'v8_trainer_independent_host_contract_review_v1','status':'PASS',
 'trainer_sha256':expected,'trainer_source':str(Path(trainer.__file__).resolve()),
 'source_sha256':trainer.source_hashes(),
 'test_source_sha256':sha(WORK/'test_train_hold_increment_v8_peer.py'),
 'host_test_log':str(log),'host_test_log_sha256':sha(log),'independent_host_tests_passed':16,
 'builder_sha256':sha(__file__),
 'command':'python -m unittest test_train_hold_increment_v8_peer -v',
 'resolved_findings':[
  'Checkpoint verification now dispatches the three new V7 histories to the trusted registry, not the legacy V4/V5/V6-only reader.',
  'Reviewed step545 HOLD regression requires an eligible frame, not only an eligible step499.',
  'Checkpoint hidden width, selected epoch and optimizer-step count match the separately authorized training definition.'],
 'verified_contracts':[
  'Eight ordered histories; seven actual-history step499 exact BASE regressions; failed V5 recovery labels remain nonzero.',
  'CPU sequential hidden state resets separately per history and advances through every frame to step545.',
  'Two reviewed step545 projected HOLD regressions, including targets outside moving-base residual bounds.',
  'Three-class confusion counts retain real UPDATE labels; checkpoint selection uses actual decoded errors after explicit development regressions.',
  'Actor receives actual state and history only; labels are confined to training and counterfactual assessment.',
  'Qualification, authorization, source graph, physical family, joint ordering and named UPDATE anchors remain explicit and hash-bound.',
  'Checkpoint-byte SHA precedes deserialization; external qualification/authorization bytes must match embedded evidence.',
  'All nontrainable checkpoint tensors are bitwise checked, including signed zero in the frozen baseline and anchor masks.',
  'Untrained or nonsequential candidates cannot be saved as qualified deployment candidates.',
  'Device-definition verification creates no optimizer and leaves every model tensor unchanged.'],
 'limitations':[
  'These are synthetic host contracts and static code review; they do not prove simulator behavior.',
  'Full eight-history input/device preflight and explicit root authorization are required before optimization.',
  'A trained candidate still requires independent full CPU replay and real checkpoint preflight before its separately scheduled Isaac evaluation.'],
 'isaac_started':False,'optimization_started_by_reviewer':False,'original_results_or_frozen_sources_modified':False,
 'physical_success_claim':False,'whole_episode_rl_qualification':False,'independent_generalization_claim':False}
path=WORK/'v8_trainer_independent_review_20261104.json'
assert not path.exists(),'Preserve existing review'
path.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'review':str(path),'sha256':sha(path),'host_test_log_sha256':sha(log),
                  'test_source_sha256':review['test_source_sha256']}))
