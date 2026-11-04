import ast
from pathlib import Path
import tempfile
import unittest
import trusted_eval_dataset_registry_v2 as m


class RegistryV2Independent(unittest.TestCase):
    def test_v8_actual_actor_and_features_are_same_recurrent_284_route(self):
        self.assertIs(m.policy.EpisodeActor,m.prior.frozen.policy.EpisodeActor)
        self.assertIs(m.policy.features,m.prior.frozen.policy.features)
        tree=ast.parse(m.engine_source())
        actor_branches=[n for n in ast.walk(tree)if isinstance(n,ast.If)
                        and isinstance(n.test,ast.Name)and n.test.id=='successful_v4']
        self.assertEqual(len(actor_branches),2)
        self.assertFalse(m.admitted_successful_wrist({'policy':'v8'},{'original_wrist_gate_qualified':1}))
        self.assertTrue(m.admitted_successful_wrist({'policy':'v8'},{'original_wrist_gate_qualified':True}))
        self.assertFalse(m.admitted_successful_wrist({'policy':'v7'},{'original_wrist_gate_qualified':True}))

    def test_five_exact_hooks_inverse_and_original_full_success_auditor_unchanged(self):
        source=Path(m.prior.frozen.__file__).read_bytes();changed=m.engine_source()
        self.assertEqual(len(m.SOURCE_HOOKS),5)
        for old,new in reversed(m.SOURCE_HOOKS):
            self.assertEqual(changed.count(new),1);changed=changed.replace(new,old)
        self.assertEqual(changed.encode(),source)
        def function(text,name):
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
        self.assertEqual(function(source,'verify_successful_v4_wrist'),function(m.engine_source(),'verify_successful_v4_wrist'))

    def test_unknown_checkpoint_and_cross_authorization_rejected_before_external_load(self):
        keys=list(m.V8_TRUSTED)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            for identity in (
                {'policy':'v8','evaluated_policy_checkpoint_sha256':'0'*64},
                {'policy':'v8','evaluated_policy_checkpoint_sha256':keys[0],
                 'qualification_sha256':m.V8_QUALIFICATION_SHA,'training_authorization_sha256':m.V8_TRUSTED[keys[1]]},
                {'policy':'v8','evaluated_policy_checkpoint_sha256':keys[0],
                 'qualification_sha256':'0'*64,'training_authorization_sha256':m.V8_TRUSTED[keys[0]]},
            ):
                with self.subTest(identity=identity),self.assertRaises(ValueError):m._v8_paths_and_trust(directory,identity)

    def test_private_engine_does_not_mutate_frozen_module_or_actor_branch(self):
        old=m.prior.frozen
        verify=old.verify_evaluation_semantics;restore=old.restore_recorded_model
        engine=m._engine()
        self.assertIs(old.verify_evaluation_semantics,verify);self.assertIs(old.restore_recorded_model,restore)
        self.assertIsNot(engine,old);self.assertIs(engine.verify_evaluation_semantics,m.verify_evaluation_semantics)
        self.assertIs(engine.restore_recorded_model,m.restore_recorded_model)
        self.assertEqual(engine.SCHEMA,m.SCHEMA)


if __name__=='__main__':unittest.main()
