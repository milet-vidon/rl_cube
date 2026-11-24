"""Independent negative cases for the two-run exception consumer (CPU only)."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from unittest.mock import patch
import hashlib,json,unittest
import numpy as np
import exception_prefix_dataset_v1 as d
import train_arm_hold_increment_v9 as v9
import test_exception_prefix_dataset_v1 as author_tests

HERE=Path(__file__).resolve().parent

class IndependentExceptionPeer(unittest.TestCase):
 def test_private_function_globals_use_only_private_trust_and_current_training_is_unchanged(self):
  old=dict(d.prior.V8_TRUSTED);runs=tuple(v9.RUN_NAMES);sources=v9.source_hashes()
  private=d.private_trust_registry()
  for name in ('_v8_paths_and_trust','restore_recorded_model','verify_evaluation_semantics'):
   self.assertIs(getattr(private,name).__globals__,private.__dict__)
   self.assertIsNot(getattr(private,name).__globals__,d.prior.__dict__)
  self.assertEqual(dict(d.prior.V8_TRUSTED),old)
  self.assertEqual(tuple(v9.RUN_NAMES),runs);self.assertEqual(v9.source_hashes(),sources)
  self.assertEqual(len(runs),9);self.assertNotIn('exception_prefix_dataset_v1.py',sources)
  self.assertTrue(all(name not in runs for name in d.RUNS))

 def test_resealed_behavior_tuple_unknown_run_and_changed_asset_reject_before_loading_model(self):
  with TemporaryDirectory(dir=HERE)as tmp:
   root=Path(tmp);name=next(iter(d.RUNS));run=root/'revotron_cube_rl/runs'/name;run.mkdir(parents=True)
   asset=root/'asset.usda';asset.write_bytes(b'pinned fixture asset')
   identity={'episode_name':name,'policy':'v8','evaluated_policy_checkpoint_sha256':'checkpoint','qualification_sha256':d.QSHA,'training_authorization_sha256':'authorization'}
   physics={'assets_sha256':{str(asset):d.sha(asset)}}
   ip=run/'evaluation_protocol.json';pp=run/'physics_identity.json'
   def prepare(value):
    ip.write_text(json.dumps(value));pp.write_text(json.dumps(physics))
    return {'checkpoint':'checkpoint','authorization':'authorization','hashes':{p.name:d.sha(p)for p in(ip,pp)}}
   spec=prepare(identity)
   with patch.object(d,'ROOT',root),patch.object(d,'ASSETS',{'asset.usda':d.sha(asset)}),patch.object(d,'RUNS',MappingProxyType({name:spec})):
    self.assertEqual(d.trust_run(run)[0],run)
    for key,value in [('policy','v4'),('episode_name','other'),('evaluated_policy_checkpoint_sha256','other'),('qualification_sha256','other'),('training_authorization_sha256','other')]:
     changed=deepcopy(identity);changed[key]=value
     # Reseal generic hashes: independent fixed model/q/auth binding must reject.
     spec.update(prepare(changed))
     with self.subTest(key=key),self.assertRaises(ValueError):d.trust_run(run)
    spec.update(prepare(identity));asset.write_bytes(b'changed fixture asset')
    with self.assertRaisesRegex(ValueError,'asset bytes'):d.trust_run(run)
    with self.assertRaises(ValueError):d.trust_run(run.parent/'unknown_run')

 def test_resealed_actor_proof_wrong_counts_or_behavior_bindings_rejected(self):
  with TemporaryDirectory(dir=HERE)as tmp:
   w=Path(tmp);run,spec,evidence,_=author_tests.PriorActorProof().make(w);report=w/spec['actor_report'];old=json.loads(report.read_text())
   for key,value in [('actor_replay_calls',3),('returned_teacher_queries',3),('actor_hidden_resets',2),('extra_recurrent_forward_calls_for_diagnostics',1),('checkpoint_sha256','wrong'),('qualification_sha256','wrong'),('authorization_sha256','wrong'),('all_actual_features_predictions_native_actions_projected_targets_bitwise_equal',False)]:
    changed=deepcopy(old);changed[key]=value;report.write_text(json.dumps(changed));sealed=deepcopy(spec);sealed['actor_report_sha256']=d.sha(report)
    with patch.object(d,'WORK',w),self.subTest(key=key),self.assertRaises(ValueError):d.RecordedPredictionProof(run,sealed,evidence)

 def test_report_reseal_cannot_substitute_another_manifest_or_feature_journal(self):
  with TemporaryDirectory(dir=HERE)as tmp:
   w=Path(tmp);run,spec,evidence,_=author_tests.PriorActorProof().make(w);report=w/spec['actor_report'];original=json.loads(report.read_text())
   for key,value in [('actual_artifact_manifest_sha256','different'),('actual_files_sha256',{'evaluation_transitions.jsonl':'bad'}),('actual_source_snapshots_sha256',{'different':'source'})]:
    altered=deepcopy(original);altered[key]=value;report.write_text(json.dumps(altered));sealed=deepcopy(spec);sealed['actor_report_sha256']=d.sha(report)
    with patch.object(d,'WORK',w),self.subTest(key=key),self.assertRaises(ValueError):d.RecordedPredictionProof(run,sealed,evidence)

 def test_cached_proof_output_cannot_mutate_next_record_and_rejects_wrong_episode(self):
  with TemporaryDirectory(dir=HERE)as tmp:
   w=Path(tmp);run,spec,evidence,_=author_tests.PriorActorProof().make(w)
   with patch.object(d,'WORK',w),patch.object(d.prior.prior.frozen.policy,'features',side_effect=lambda value:np.asarray(value['features'],np.float32)):
    actor=d.RecordedPredictionProof(run,spec,evidence);actor.reset('run')
    first=actor.act({'features':[0]*284,'expert_label':'ignored by measured feature extraction'},episode_id='run',step=0)
    first[:]=123.;np.testing.assert_array_equal(actor.values[0][1],np.zeros(28,np.float32))
    with self.assertRaises(ValueError):actor.act({'features':[1]*284},episode_id='other',step=1)
    second=actor.act({'features':[1]*284},episode_id='run',step=1);np.testing.assert_array_equal(second,np.ones(28,np.float32))
    self.assertEqual(actor.proof['new_gru_calls'],0);self.assertEqual(actor.next_step,2)

if __name__=='__main__':unittest.main()
