"""Registry routing and isolated actual-artifact tamper checks; no simulator."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import ast,hashlib,io,json,os,shutil,tempfile,unittest
import torch
import trusted_eval_dataset_registry_v2 as r

HERE=Path(__file__).resolve().parent


class RegistryV2HostContracts(unittest.TestCase):
    def test_counted_hooks_inverse_restores_entire_frozen_engine(self):
        original=Path(r.prior.frozen.__file__).read_text();changed=r.engine_source()
        for before,after in reversed(r.SOURCE_HOOKS):
            self.assertEqual(changed.count(after),1);changed=changed.replace(after,before)
        self.assertEqual(changed,original);self.assertEqual(len(r.SOURCE_HOOKS),5)
        self.assertEqual(r.sha(r.prior.__file__),r.V1_SHA)

    def test_nonrouting_functions_and_success_auditor_ast_unchanged(self):
        functions=lambda text:{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        old=functions(Path(r.prior.frozen.__file__).read_text());new=functions(r.engine_source())
        self.assertEqual(set(old),set(new))
        for name in old:
            if name not in ('verify_evidence','load_run'):self.assertEqual(old[name],new[name],name)
        self.assertEqual(old['verify_successful_v4_wrist'],new['verify_successful_v4_wrist'])
        source=r.engine_source()
        self.assertEqual(source.count('if successful_v4:'),2) # V4 actor + V4 recorded features only.

    def test_success_outcome_is_not_v4_actor_or_unreviewed_policy_admission(self):
        for version,flag,expected in [('v4',True,True),('v8',True,True),('v8',False,False),
            ('v8',1,False),('v5',True,False),('v6',True,False),('v7',True,False),('unknown',True,False)]:
            self.assertIs(r.admitted_successful_wrist({'policy':version},{'original_wrist_gate_qualified':flag}),expected)
        self.assertNotIn('unknown',r.REGISTRY)
        self.assertEqual(len(r.V8_TRUSTED),2)

    def test_private_instances_do_not_mutate_frozen_old_module(self):
        old=r.prior.frozen.verify_evaluation_semantics
        a=r._engine();b=r._engine();self.assertIsNot(a,b)
        self.assertIs(r.prior.frozen.verify_evaluation_semantics,old)
        self.assertIs(a.policy.EpisodeActor,r.policy.EpisodeActor)
        self.assertIs(a.policy.features,r.policy.features)

    def test_v8_runtime_routing_reconstructs_exact_legacy_control_source(self):
        raw=(HERE/'state_hold_increment_eval_runtime_v8.py').read_bytes()
        self.assertEqual(hashlib.sha256(r.normalize_v8_runtime(raw).encode()).hexdigest(),r.prior.LEGACY_RUNTIME_SHA)
        with self.assertRaisesRegex(ValueError,'runtime bytes'):r.normalize_v8_runtime(raw+b'\n')

    def test_v8_trust_rejects_unknown_checkpoint_cross_authorization_and_qualification(self):
        checkpoint=next(iter(r.V8_TRUSTED));identity={'policy':'v8','evaluated_policy_checkpoint_sha256':checkpoint,
            'qualification_sha256':r.V8_QUALIFICATION_SHA,'training_authorization_sha256':r.V8_TRUSTED[checkpoint]}
        for field,value in [('evaluated_policy_checkpoint_sha256','0'*64),('qualification_sha256','0'*64),
                ('training_authorization_sha256',list(r.V8_TRUSTED.values())[1])]:
            altered=identity|{field:value}
            with self.assertRaises(ValueError):r._v8_paths_and_trust(HERE,altered)


FIXTURE=Path(os.environ.get('V8_REGISTRY_FIXTURE','/home/brainco/rl_cube/github_project/revotron_cube_rl/runs/left_hold_increment_v8_seed20261104_eval_9501_20261104'))


class RegistryV2ActualArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not(FIXTURE/'artifact_manifest.json').exists():raise unittest.SkipTest('Completed actual V8 fixture unavailable; host checks are not physical proof')
        cls.tmp=tempfile.TemporaryDirectory(dir=HERE,prefix='v8_registry_isolated_');cls.fixture_run=Path(cls.tmp.name).resolve()
        manifest=r.read_json(FIXTURE/'artifact_manifest.json')
        for name in ['artifact_manifest.json',*manifest['sha256']]:
            target=cls.fixture_run/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(FIXTURE/name,target)
        for folder in ('source_snapshot','evaluation_source_snapshot','checkpoints','plans'):
            shutil.copytree(FIXTURE/folder,cls.fixture_run/folder,dirs_exist_ok=True)
        cls.baseline=torch.load(cls.fixture_run/'checkpoints/frozen_v4_baseline.pt',map_location='cpu',weights_only=True)
        cls.bsha=r.sha(cls.fixture_run/'checkpoints/frozen_v4_baseline.pt')
        cls.family=r.read_json(cls.fixture_run/'evaluation_protocol.json')['actual_physics_family_sha256']
        cls.lines=(cls.fixture_run/'evaluation_transitions.jsonl').read_bytes().splitlines(keepends=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):self.saved={}
    def tearDown(self):
        for name,data in self.saved.items():(self.fixture_run/name).write_bytes(data)
    def write(self,name,data):
        if name not in self.saved:self.saved[name]=(self.fixture_run/name).read_bytes()
        (self.fixture_run/name).write_bytes(data)
    def reseal(self,*names):
        m=r.read_json(self.fixture_run/'artifact_manifest.json')
        for name in names:m['sha256'][name]=r.sha(self.fixture_run/name)
        self.write('artifact_manifest.json',(json.dumps(m)+'\n').encode())
    def change_record(self,index,edit):
        lines=list(self.lines);record=json.loads(lines[index]);edit(record);lines[index]=(json.dumps(record)+'\n').encode()
        self.write('evaluation_transitions.jsonl',b''.join(lines));self.reseal('evaluation_transitions.jsonl')
    def load(self):return r.load_run(self.fixture_run,self.baseline,self.bsha,self.family)

    def test_completed_original_outcome_flags_and_actual_actor_replay(self):
        data=self.load();original=r.read_json(FIXTURE/'result.json')
        self.assertEqual(len(data['x']),original['steps']);self.assertEqual(data['x'].shape[1],284)
        self.assertTrue(data['audit']['behavior_predictions_replayed_bitwise'])
        self.assertFalse(data['training_qualification_granted'])
        for key in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim'):
            self.assertIs(data['result'][key],False)
        self.assertEqual(data['result']['original_wrist_gate_qualified'],original['original_wrist_gate_qualified'])
        if original['original_wrist_gate_qualified']:
            self.assertEqual(data['audit']['original_successful_wrist_audit'],original['original_grasp_gate'])
        proof=data['evidence']['evaluation_semantics_proof']
        self.assertEqual(len(proof['source_snapshot_resolutions']),18)
        self.assertTrue(all(v['scope']=='actual_evaluation_direct_snapshot'for v in proof['source_snapshot_resolutions'].values()))

    def test_resealed_fabricated_success_or_failure_is_rejected(self):
        result=r.read_json(self.fixture_run/'result.json');was_success=result['original_wrist_gate_qualified']
        result['original_wrist_gate_qualified']=not was_success
        result['failure']='fabricated_failure'if was_success else None
        self.write('result.json',json.dumps(result).encode());self.reseal('result.json')
        with self.assertRaises(ValueError):self.load()

    def test_resealed_expert_label_is_rejected(self):
        self.change_record(499,lambda row:row['counterfactual_expert_projection_diagnostic_only']['projected_target'].__setitem__(0,123.))
        with self.assertRaisesRegex(ValueError,'projected label mismatch'):self.load()

    def test_resealed_original_false_flag_is_rejected(self):
        result=r.read_json(self.fixture_run/'result.json');result['corrective_labels_qualified_for_training']=True
        self.write('result.json',json.dumps(result).encode());self.reseal('result.json')
        with self.assertRaisesRegex(ValueError,'False flag'):self.load()

    def test_in_memory_anchor_tensor_tamper_is_rejected(self):
        saved=torch.load(self.fixture_run/'checkpoints/evaluated_policy.pt',map_location='cpu',weights_only=True)
        saved['model']['update_anchor_is_previous'][0]=True
        with self.assertRaisesRegex(ValueError,'Supplied V8 tensor'):
            r.restore_recorded_model(self.fixture_run,r.read_json(self.fixture_run/'evaluation_protocol.json'),saved)

    def test_actual_gru_behavior_cannot_be_replaced_after_load(self):
        original=r.restore_recorded_model
        def altered(*args):
            model=original(*args)
            with torch.no_grad():
                model.mode_head.weight.zero_();model.mode_head.bias.reshape(28,3).fill_(-100.)
                model.mode_head.bias.reshape(28,3)[:,2]=100.;model.head.bias.fill_(1.)
            return model
        with patch.object(r,'restore_recorded_model',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'complete-history GRU replay'):self.load()

    def test_resealed_predicted_target_disagrees_with_execution(self):
        self.change_record(499,lambda row:row['predicted_left_target'].__setitem__(0,row['predicted_left_target'][0]+.01))
        with self.assertRaisesRegex(ValueError,'composition/right'):self.load()

    def test_resealed_new_anchor_metadata_is_rejected(self):
        identity=r.read_json(self.fixture_run/'evaluation_protocol.json');identity['update_anchor_is_previous'][0]=True
        result=r.read_json(self.fixture_run/'result.json');result['evaluation_identity']=identity
        self.write('evaluation_protocol.json',json.dumps(identity).encode());self.write('result.json',json.dumps(result).encode())
        self.reseal('evaluation_protocol.json','result.json')
        with self.assertRaisesRegex(ValueError,'anchor mask differs'):self.load()


if __name__=='__main__':unittest.main()
