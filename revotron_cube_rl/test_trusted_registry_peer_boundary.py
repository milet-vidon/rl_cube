import ast,hashlib,unittest
from pathlib import Path
import trusted_eval_dataset_registry_v1 as registry

class RegistryBoundaryTests(unittest.TestCase):
 def identity(self):
  checkpoint=next(iter(registry.V7_TRUSTED))
  return {'policy':'v7','evaluated_policy_checkpoint_sha256':checkpoint,
      'qualification_sha256':registry.V7_QUALIFICATION_SHA,'training_authorization_sha256':registry.V7_TRUSTED[checkpoint]}
 def test_unknown_resealed_checkpoint_is_rejected_before_files(self):
  identity=self.identity();identity['evaluated_policy_checkpoint_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'outside.*registry'):registry._v7_paths_and_trust(Path('/not_read'),identity)
 def test_cross_checkpoint_authorization_is_rejected_before_files(self):
  identity=self.identity();identity['training_authorization_sha256']=list(registry.V7_TRUSTED.values())[1]
  with self.assertRaisesRegex(ValueError,'Wrong reviewed training authorization'):registry._v7_paths_and_trust(Path('/not_read'),identity)
 def test_different_qualification_rejected_before_files(self):
  identity=self.identity();identity['qualification_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'Unreviewed V7 qualification'):registry._v7_paths_and_trust(Path('/not_read'),identity)
 def test_same_actual_episode_actor_class_is_used_by_private_engine(self):
  engine=registry._engine()
  self.assertIs(engine.policy.EpisodeActor,registry.policy.EpisodeActor)
  self.assertIs(engine.policy.features,registry.policy.features)
  self.assertIs(engine.restore_recorded_model,registry.restore_recorded_model)
 def test_old_frozen_predicate_functions_except_admission_are_identical_ast(self):
  raw=Path(registry.frozen.__file__).read_text();changed=registry.engine_source()
  def funcs(text):return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  a=funcs(raw);b=funcs(changed)
  self.assertEqual({k:v for k,v in a.items()if k!='verify_evidence'},{k:v for k,v in b.items()if k!='verify_evidence'})
  self.assertEqual(hashlib.sha256(changed.replace("identity['policy']in REGISTRY","identity['policy']in ('v4','v5','v6')").encode()).hexdigest(),registry.ENGINE_SHA)

if __name__=='__main__':unittest.main()
