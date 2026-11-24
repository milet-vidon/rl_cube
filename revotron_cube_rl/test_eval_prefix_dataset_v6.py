"""Host integrity tests on an isolated copy of completed real evaluation data.

Set V6_EVAL_FIXTURE to the recorded V5 diagnostic run. Tests never edit the
source evidence, import an AppLauncher, or grant training qualification.
"""
from copy import deepcopy
import json,os,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import eval_prefix_dataset_v6 as loader

HERE=Path(__file__).resolve().parent
FIXTURE=Path(os.environ.get('V6_EVAL_FIXTURE','/home/brainco/rl_cube/github_project/revotron_cube_rl/runs/left_state_residual_v5_long_eval_9501_20261104_v2'))

class PrefixDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE/'artifact_manifest.json').exists():raise unittest.SkipTest('Explicit recorded V5 eval fixture is unavailable')
        cls.temporary=tempfile.TemporaryDirectory(dir=HERE,prefix='v6_prefix_host_test_')
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

    def test_real_complete_prefix_has_only_619_candidates_and_preserves_false_flags(self):
        episode=self.load();self.assertEqual(episode['x'].shape,(619,284));self.assertEqual(int(episode['mask'].sum()),619)
        self.assertEqual(episode['audit']['oracle_queries'],620)
        self.assertEqual(episode['excluded_labels'],[{'step':619,'reason':'contact_topology_or_settle_timeout','type':'unexecuted_abort_no_label'}])
        for key in ('eligible_training_episode','corrective_labels_qualified_for_training','runtime_bridge_validated','pure_student_episode_claim'):
            self.assertIs(episode['result'][key],False)
        self.assertEqual(episode['labels'].dtype,np.float32)
        loader.verify_saved_datasets({'dataset_evidence':[episode['evidence']],'physics_family_sha256':self.family},[self.dataset_dir])

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
        self.change_record(500,lambda r:r['original_controller_audit']['execution_memory_changes'][0].__setitem__('after',99.))
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
        record=json.loads(self.lines[0]);record['step']=619
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

if __name__=='__main__':unittest.main()
