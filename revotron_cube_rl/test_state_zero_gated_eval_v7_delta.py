"""CPU contracts: V7 routing preserves frozen control and actual GRU history."""
import ast
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
import evaluate_state_zero_gated_v7 as entry
from zero_gated_residual_policy_v7 import EpisodeActor,features
from test_zero_gated_residual_policy_v7 import policy
from test_state_zero_gated_eval_v7 import FakeRuntime,joint_safety
from state_zero_gated_eval_runtime_v7 import evaluate

ROOT=Path(__file__).resolve().parent

class V7RoutingTests(unittest.TestCase):
    def test_entire_runtime_is_only_version_route_change(self):
        original=(ROOT/'state_residual_eval_runtime_v6.py').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),'06d3ed5aaf99bea9e29898c1bc4d2e50b874533386aa0725103af77891b08094')
        changed=(ROOT/'state_zero_gated_eval_runtime_v7.py').read_text()
        changed=changed.replace('from zero_gated_residual_policy_v7 import EpisodeActor, features as v7_features',
                                'from phase_gated_residual_policy_v6 import EpisodeActor, features as v6_features')
        changed=changed.replace("== 'v7'","== 'v6'").replace("=='v7'","=='v6'").replace('v7_features(', 'v6_features(')
        self.assertEqual(changed,original.decode())

    def test_main_execution_and_cleanup_ast_unchanged_after_cli_additions(self):
        def main(path,new):
            text=path.read_text()
            if new:
                text=text.replace("choices=('v7',),default='v7'","choices=('v6','v4'),default='v6'")
                text='\n'.join(line for line in text.splitlines()if "parser.add_argument('--authorization"not in line)
            tree=ast.parse(text)
            return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='main'),include_attributes=False)
        self.assertEqual(main(ROOT/'evaluate_state_zero_gated_v7.py',True),main(ROOT/'evaluate_state_residual_v6.py',False))

    def test_real_learned_gate_routes_from_reset_and_replays_actual_history(self):
        torch.manual_seed(16);model=policy();model.head.bias.data.fill_(.5)
        model.correction_gate.bias.data.fill_(-1.);model.correction_gate.bias.data[[2,17]]=1.
        runtime=FakeRuntime(success_step=3);runtime.teacher.name='squeeze'
        runtime.actor=EpisodeActor(model,list(range(28)));runtime.actor.reset(runtime.args.name)
        def schedule(obs,before,after):
            runtime.current_schedule={'phase':after['name'],'phase_progress':(71/84,72/84,1.)[len(runtime.rows)]}
            return dict(runtime.current_schedule)
        runtime.schedule_resolver=schedule;journal=io.StringIO()
        with patch.dict('sys.modules',{'joint_safety':joint_safety}):summary=evaluate(runtime,journal,max_steps=10)
        rows=[json.loads(line)for line in journal.getvalue().splitlines()if json.loads(line)['record_type']=='executed_transition']
        self.assertEqual(len(rows),3);self.assertTrue(summary['original_wrist_gate_qualified'])
        self.assertEqual(summary['left_teacher_position_action_steps'],0)
        replay=EpisodeActor(model,list(range(28)));replay.reset(runtime.args.name)
        for step,row in enumerate(rows):
            obs=row['actor_observation_pre']|{'previous_executed_source':'student'}
            result=replay.act(obs,episode_id=runtime.args.name,step=step)
            np.testing.assert_array_equal(np.asarray(result,dtype=np.float32).view(np.int32),np.asarray(row['predicted_left_target'],dtype=np.float32).view(np.int32))
            if step:self.assertEqual(obs['applied_target'],rows[step-1]['actor_observation_post']['applied_target'])
            x=torch.from_numpy(features(obs)).reshape(1,1,-1)
            base=model.baseline_targets(x,torch.tensor(obs['applied_target'][:28]).reshape(1,1,28)).numpy().reshape(28)
            different=np.asarray(result,dtype=np.float32).view(np.int32)!=base.view(np.int32)
            self.assertEqual(np.flatnonzero(different).tolist(),[]if step==0 else[2,17])
        self.assertEqual(replay.next_step,3)


class V7QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(os.environ.get('V7_QUALIFICATION_FIXTURE',ROOT/'v7_five_history_qualification_03.json'))
        if not path.exists():path=ROOT/'v7_five_history_inspection_20261104_03/qualification_candidate.json'
        if not path.exists():raise unittest.SkipTest('Frozen real v7 qualification fixture unavailable')
        cls.qualification=json.loads(path.read_text())
        cls.paths=[Path('/reviewed')/name for name in entry.inspect_residual_dataset_v7.RUN_NAMES]
    def validate(self,q=None,paths=None):
        with patch.object(entry.inspect_residual_dataset_v7,'source_hashes',return_value=self.qualification['source_sha256']):
            entry.validate_qualification(q or self.qualification,paths or self.paths)
    def test_frozen_five_history_candidate_with_original_false_flags_is_accepted(self):self.validate()
    def test_changed_candidate_flags_sources_gate_classes_and_reachability_rejected(self):
        cases=[('training_authorized',True),('pure_student_episode_success_claim',True),
               ('eligible_under_declared_v7_candidate_bounds',False),('source_sha256',{}),
               ('current_candidate_gate',{}),('classification_contract',{}),('active_max_unreachable_distance_rad',1e-10)]
        for key,value in cases:
            with self.subTest(key=key):
                q=deepcopy(self.qualification);q[key]=value
                with self.assertRaises(ValueError):self.validate(q)
    def test_reordered_missing_or_duplicate_histories_rejected(self):
        for paths in (self.paths[::-1],self.paths[:-1],self.paths[:-1]+[self.paths[0]]):
            with self.assertRaises(ValueError):self.validate(paths=paths)


class V7CandidateMetadataTests(unittest.TestCase):
    def metadata(self):
        return {'offline_candidate_qualified':True,'best_epoch':1,'optimizer_steps':5,
            'final_cpu_sequential_metrics':{'step499_reviewed_zero_histories_bitwise_baseline':True,'inference':'sequential_one_frame_hidden_state'},
            'eligible_for_rl_episode_training':False,'independent_generalization_demonstrated':False,
            'closed_loop_success_demonstrated':False,'right28_learned':False}
    def test_trained_offline_candidate_with_only_declared_scope(self):entry.validate_candidate_metadata(self.metadata())
    def test_missing_cpu_boundary_training_or_changed_scope_is_rejected(self):
        for key,value in [('best_epoch',0),('optimizer_steps',0),('offline_candidate_qualified',False),
                          ('final_cpu_sequential_metrics',{}),('closed_loop_success_demonstrated',True),('right28_learned',True)]:
            with self.subTest(key=key):
                m=self.metadata();m[key]=value
                with self.assertRaises(ValueError):entry.validate_candidate_metadata(m)

if __name__=='__main__':unittest.main()
