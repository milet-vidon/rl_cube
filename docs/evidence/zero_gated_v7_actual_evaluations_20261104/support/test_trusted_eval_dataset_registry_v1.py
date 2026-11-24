"""Host integrity tests on an isolated copy of completed real evaluation data.

Set REGISTRY_EVAL_FIXTURE to the recorded V6 diagnostic run. Tests never edit the
source evidence, import an AppLauncher, or grant training qualification.
"""
from copy import deepcopy
import io,json,os,shutil,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import trusted_eval_dataset_registry_v1 as loader

HERE=Path(__file__).resolve().parent
FIXTURE=Path(os.environ.get('REGISTRY_EVAL_FIXTURE','/home/brainco/rl_cube/github_project/revotron_cube_rl/runs/left_zero_gated_residual_v7_eval_9501_20261104'))

class PrefixDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE/'artifact_manifest.json').exists():raise unittest.SkipTest('Explicit recorded V6 eval fixture is unavailable')
        cls.temporary=tempfile.TemporaryDirectory(dir=HERE,prefix='v7_prefix_host_test_')
        cls.dataset_dir=Path(cls.temporary.name).resolve();assert cls.dataset_dir.is_relative_to(HERE.resolve())
        manifest=loader.read_json(FIXTURE/'artifact_manifest.json')
        for name in ['artifact_manifest.json',*manifest['sha256']]:
            target=cls.dataset_dir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(FIXTURE/name,target)
        for folder in ('source_snapshot','evaluation_source_snapshot','checkpoints','plans'):
            shutil.copytree(FIXTURE/folder,cls.dataset_dir/folder)
        cls.baseline=torch.load(cls.dataset_dir/'checkpoints/frozen_v4_baseline.pt',map_location='cpu',weights_only=True)
        cls.baseline_sha=loader.sha(cls.dataset_dir/'checkpoints/frozen_v4_baseline.pt')
        cls.family=loader.read_json(cls.dataset_dir/'evaluation_protocol.json')['actual_physics_family_sha256']
        cls.lines=(cls.dataset_dir/'evaluation_transitions.jsonl').read_bytes().splitlines(keepends=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):self.saved={}
    def tearDown(self):
        for name,blob in self.saved.items():(self.dataset_dir/name).write_bytes(blob)
    def write(self,name,blob):
        path=self.dataset_dir/name
        if name not in self.saved:self.saved[name]=path.read_bytes()
        path.write_bytes(blob)
    def reseal(self,*names):
        manifest=loader.read_json(self.dataset_dir/'artifact_manifest.json')
        for name in names:manifest['sha256'][name]=loader.sha(self.dataset_dir/name)
        self.write('artifact_manifest.json',(json.dumps(manifest)+'\n').encode())
    def change_record(self,index,edit):
        records=list(self.lines);record=json.loads(records[index]);edit(record)
        records[index]=(json.dumps(record,allow_nan=False)+'\n').encode()
        self.write('evaluation_transitions.jsonl',b''.join(records));self.reseal('evaluation_transitions.jsonl')
    def load(self):return loader.load_run(self.dataset_dir,self.baseline,self.baseline_sha,self.family)

    def test_real_complete_prefix_has_only_752_candidates_and_preserves_false_flags(self):
        episode=self.load();self.assertEqual(episode['x'].shape,(752,284));self.assertEqual(int(episode['mask'].sum()),752)
        self.assertEqual(episode['audit']['oracle_queries'],753)
        self.assertEqual(episode['excluded_labels'],[{'step':752,'reason':'contact_topology_or_settle_timeout','type':'unexecuted_abort_no_label'}])
        for key in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim'):
            self.assertIs(episode['result'][key],False)
        self.assertEqual(episode['labels'].dtype,np.float32)
        loader.verify_saved_datasets({'dataset_evidence':[episode['evidence']],'physics_family_sha256':self.family},[self.dataset_dir])

    def test_runtime_semantics_normalizes_to_exact_frozen_v5_source(self):
        proof=self.load()['evidence']['evaluation_semantics_proof']
        self.assertEqual(proof['policy_version'],'v7')
        self.assertEqual(proof['runtime_after_only_policy_routing_rename_sha256'],'90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64')
        self.assertTrue(proof['same_original_physics_step_projection_right_action_query_order_gates_termination_and_audit'])

    def test_resealed_checkpoint_gate_payload_is_not_a_trusted_behavior(self):
        name='checkpoints/evaluated_policy.pt';saved=torch.load(self.dataset_dir/name,map_location='cpu',weights_only=True)
        saved['model']['squeeze_active_progress'].zero_()
        buffer=io.BytesIO();torch.save(saved,buffer);self.write(name,buffer.getvalue())
        identity=loader.read_json(self.dataset_dir/'evaluation_protocol.json')
        identity['evaluated_policy_checkpoint_sha256']=loader.sha(self.dataset_dir/name)
        result=loader.read_json(self.dataset_dir/'result.json');result['evaluation_identity']=identity
        result['evaluated_policy_checkpoint_sha256']=identity['evaluated_policy_checkpoint_sha256']
        self.write('evaluation_protocol.json',json.dumps(identity).encode());self.write('result.json',json.dumps(result).encode())
        self.reseal('evaluation_protocol.json','result.json')
        with self.assertRaisesRegex(ValueError,'reviewed registry'):self.load()

    def test_canonical_control_buffer_check_rejects_changed_gate_in_memory(self):
        saved=torch.load(self.dataset_dir/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
        saved['model']['squeeze_active_progress'].zero_()
        with self.assertRaisesRegex(ValueError,'Supplied behavior tensor'):
            loader.restore_recorded_model(self.dataset_dir,loader.read_json(self.dataset_dir/'evaluation_protocol.json'),saved)

    def test_gru_history_replay_rejects_a_different_behavior_model(self):
        original=loader.restore_recorded_model
        def changed(*args):
            model=original(*args)
            with torch.no_grad():model.gru.weight_ih_l0.add_(.1)
            return model
        with patch.object(loader,'restore_recorded_model',changed):
            with self.assertRaisesRegex(ValueError,'complete-history GRU replay'):self.load()

    def test_resealed_gate_contract_is_rejected(self):
        identity=loader.read_json(self.dataset_dir/'evaluation_protocol.json')
        identity['public_residual_gate_contract']['squeeze_active_progress_numerator']=73
        result=loader.read_json(self.dataset_dir/'result.json');result['evaluation_identity']=identity
        self.write('evaluation_protocol.json',json.dumps(identity).encode());self.write('result.json',json.dumps(result).encode())
        self.reseal('evaluation_protocol.json','result.json')
        with self.assertRaisesRegex(ValueError,'Public residual gate'):self.load()

    def test_resealed_runtime_source_still_requires_reviewed_semantics(self):
        name='evaluation_source_snapshot/state_zero_gated_eval_runtime_v7.py'
        self.write(name,(self.dataset_dir/name).read_bytes()+b'\n# semantic source changed\n')
        identity=loader.read_json(self.dataset_dir/'evaluation_protocol.json')
        identity['source_sha256']['state_zero_gated_eval_runtime_v7.py']=loader.sha(self.dataset_dir/name)
        result=loader.read_json(self.dataset_dir/'result.json');result['evaluation_identity']=identity
        self.write('evaluation_protocol.json',json.dumps(identity).encode());self.write('result.json',json.dumps(result).encode())
        self.reseal('evaluation_protocol.json','result.json')
        with self.assertRaisesRegex(ValueError,'Unreviewed V7 runtime'):self.load()

    def test_manifest_tamper_rejected(self):
        manifest=loader.read_json(self.dataset_dir/'artifact_manifest.json');manifest['sha256']['result.json']='0'*64
        self.write('artifact_manifest.json',json.dumps(manifest).encode())
        with self.assertRaisesRegex(ValueError,'manifest mismatch'):self.load()

    def test_frozen_source_snapshot_tamper_rejected_independently(self):
        name='source_snapshot/left_dagger_baseline_runtime_v3.py';self.write(name,(self.dataset_dir/name).read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'Source snapshot mismatch'):self.load()

    def test_original_diagnostic_false_cannot_be_promoted(self):
        result=loader.read_json(self.dataset_dir/'result.json');result['eligible_training_episode']=True
        self.write('result.json',(json.dumps(result)+'\n').encode());self.reseal('result.json')
        with self.assertRaisesRegex(ValueError,'False flag'):self.load()

    def test_resealed_expert_label_tamper_rejected(self):
        self.change_record(0,lambda r:r['counterfactual_expert_projection_diagnostic_only']['projected_target'].__setitem__(0,99.))
        with self.assertRaisesRegex(ValueError,'projected label'):self.load()

    def test_resealed_right_native_action_tamper_rejected(self):
        self.change_record(0,lambda r:r['execution']['native_action'].__setitem__(28,.001))
        with self.assertRaisesRegex(ValueError,'right native'):self.load()

    def test_resealed_memory_sync_tamper_rejected(self):
        index=next(i for i,line in enumerate(self.lines)if json.loads(line).get('original_controller_audit',{}).get('execution_memory_changes'))
        self.change_record(index,lambda r:r['original_controller_audit']['execution_memory_changes'][0].__setitem__('after',99.))
        with self.assertRaisesRegex(ValueError,'Memory sync'):self.load()

    def test_resealed_hidden_state_discontinuity_rejected(self):
        self.change_record(1,lambda r:r['original_controller_audit']['teacher_state_before_sync'].__setitem__('tick',999))
        with self.assertRaisesRegex(ValueError,'hidden-state discontinuity'):self.load()

    def test_resealed_public_phase_clock_tamper_rejected(self):
        def change(record):
            record['actor_observation_pre']['phase_progress']=.5
            record['original_controller_audit']['public_same_action_schedule']['phase_progress']=.5
        self.change_record(0,change)
        with self.assertRaisesRegex(ValueError,'progress/clock'):self.load()

    def test_resealed_actual_oracle_step_tamper_rejected(self):
        self.change_record(0,lambda r:r['original_controller_audit']['oracle_env_inputs'].__setitem__('episode_length',1))
        with self.assertRaisesRegex(ValueError,'counter/reset'):self.load()

    def test_resealed_post_gate_tamper_rejected(self):
        self.change_record(0,lambda r:r['original_stage_post_gates'].__setitem__('robot_safety',False))
        with self.assertRaisesRegex(ValueError,'post-gate'):self.load()

    def test_resealed_raw_done_tamper_rejected(self):
        self.change_record(0,lambda r:r['raw_environment_termination'].__setitem__('raw_done',True))
        with self.assertRaisesRegex(ValueError,'termination streams'):self.load()

    def test_abort_cannot_hide_a_completed_physics_step(self):
        self.change_record(-2,lambda r:r.__setitem__('physics_step_completed_for_unfinished_decision',True))
        with self.assertRaisesRegex(ValueError,'Unrecorded physical step'):self.load()

    def test_aborted_query_is_not_an_extra_training_transition(self):
        record=json.loads(self.lines[0]);record['step']=752
        altered=b''.join(self.lines[:-1])+json.dumps(record).encode()+b'\n'+self.lines[-1]
        self.write('evaluation_transitions.jsonl',altered);self.reseal('evaluation_transitions.jsonl')
        with self.assertRaisesRegex(ValueError,'after terminal'):self.load()

    def test_final_reason_must_match_the_unexecuted_failure_query(self):
        self.change_record(-1,lambda r:r['summary'].__setitem__('failure','different_failure'))
        with self.assertRaisesRegex(ValueError,'Final diagnostic scope/reason'):self.load()

    def test_plan_content_cannot_disagree_with_its_hashed_name(self):
        path=next((self.dataset_dir/'plans').glob('*.json'));name=path.relative_to(self.dataset_dir).as_posix()
        value=loader.read_json(path);value.append('changed')
        self.write(name,(json.dumps(value)+'\n').encode())
        with self.assertRaisesRegex(ValueError,'Plan content'):self.load()

class V4SuccessfulHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        success=FIXTURE.parent/'left_phase_v4_matched_eval_9501_20261104_v2'
        with patch.dict(PrefixDatasetTests.setUpClass.__func__.__globals__,{'FIXTURE':success}):
            PrefixDatasetTests.setUpClass.__func__(cls)
    tearDownClass=classmethod(PrefixDatasetTests.tearDownClass.__func__)
    setUp=PrefixDatasetTests.setUp
    tearDown=PrefixDatasetTests.tearDown
    write=PrefixDatasetTests.write
    reseal=PrefixDatasetTests.reseal
    change_record=PrefixDatasetTests.change_record
    load=PrefixDatasetTests.load

    def test_actual_success_uses_198_behavior_replay_and_284_supervision(self):
        data=self.load();self.assertEqual(data['x'].shape,(1654,284));self.assertEqual(int(data['mask'].sum()),1654)
        self.assertEqual(data['audit']['recorded_actor_feature_count'],198)
        self.assertTrue(data['audit']['behavior_predictions_replayed_bitwise'])
        self.assertTrue(data['audit']['original_successful_wrist_audit']['qualified'])
        self.assertEqual(data['excluded_labels'],[]);self.assertIsNone(data['result']['failure'])
        self.assertFalse(data['result']['eligible_training_episode']);self.assertFalse(data['result']['whole_robot_pure_neural_episode'])

    def test_resealed_loss_of_final_assembled_hold_is_not_success(self):
        rows=(self.dataset_dir/'trajectory.jsonl').read_bytes().splitlines(keepends=True)
        last=json.loads(rows[-1]);last['layer_measurement']['assembled_for_grasp']=False
        rows[-1]=(json.dumps(last)+'\n').encode();self.write('trajectory.jsonl',b''.join(rows));self.reseal('trajectory.jsonl')
        with self.assertRaisesRegex(ValueError,'wrist/grasp auditors'):self.load()


class PrivatePredicateEngineTests(unittest.TestCase):
    def test_admission_hook_is_only_source_change(self):
        restored=loader.engine_source().replace("identity['policy']in REGISTRY","identity['policy']in ('v4','v5','v6')")
        import hashlib
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),loader.ENGINE_SHA)
    def test_original_module_functions_and_bytes_are_not_mutated(self):
        old=loader.frozen.load_run;old_verify=loader.frozen.verify_evaluation_semantics
        a=loader._engine();b=loader._engine()
        self.assertIsNot(a,b)
        self.assertIs(loader.frozen.load_run,old)
        self.assertIs(loader.frozen.verify_evaluation_semantics,old_verify)
        self.assertEqual(a.load_run.__code__.co_code,old.__code__.co_code)
        self.assertEqual(loader.sha(loader.frozen.__file__),loader.ENGINE_SHA)
        with self.assertRaises(TypeError):loader.REGISTRY['unknown']='not allowed'

if __name__=='__main__':unittest.main()
