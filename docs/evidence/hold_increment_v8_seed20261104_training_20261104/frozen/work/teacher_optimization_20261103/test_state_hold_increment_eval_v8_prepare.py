"""V8 prepare boundary contracts; the qualified checkpoint loader is mocked.

These tests do not constitute a real trained-model preflight or Isaac evidence.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import evaluate_state_hold_increment_v8 as entry


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class V8PrepareContracts(unittest.TestCase):
    def invoke(self,change=None):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as temporary:
            root=Path(temporary);candidate=root/'candidate.pt';baseline=root/'baseline.pt'
            qualification=root/'q.json';authorization=root/'auth.json'
            candidate.write_bytes(b'mocked loader boundary; not a trained checkpoint')
            baseline.write_bytes(b'mocked baseline');authorization.write_text('{}')
            p=entry.hold_increment_policy_v8;t=entry.train_hold_increment_v8
            left=list(p.EXPECTED_LEFT_NAMES)
            names=left+[n.replace('_L','_R').replace('left_','right_')for n in left]
            bounds=torch.tensor(p.candidate_bounds(left),dtype=torch.float32)
            anchors=torch.tensor([n in p.RECOVERY_JOINTS for n in left],dtype=torch.bool)
            common={'left_joint_names':left,'update_bound_rad':bounds.tolist(),
                    'update_anchor_is_previous':anchors.tolist(),
                    'action_contract':deepcopy(p.ACTION_CONTRACT),
                    'classification_contract':deepcopy(p.CLASSIFICATION_CONTRACT)}
            q=deepcopy(common)|{'schema':t.QUAL_SCHEMA,'training_authorized':False,
                'requires_root_review_before_training':True,'eligible_for_development_bc_transition_supervision':True,
                'all_original_continuous_labels_preserved':True,'eligible_for_rl_episode_training':False,
                'pure_student_episode_success_claim':False,'independent_generalization_claim':False,
                'original_result_files_modified':False,'source_sha256':t.source_hashes(),
                'model_schema':p.SCHEMA,'feature_schema':p.FEATURE_SCHEMA,
                'public_gate_contract':deepcopy(p.PUBLIC_GATE_CONTRACT),
                'selection_contract':deepcopy(t.SELECTION_CONTRACT),
                'branch_coverage':[{'active_max_union_gap_rad':0.,
                    'active_no_float32_branch_witness_elements':0,'masked_corrections_gt_1e_6':0}for _ in range(8)],
                'physics_family_sha256':'mock physics family','frozen_baseline_checkpoint_sha256':sha(baseline)}
            base_metadata={'joint_names':names,'output_indices':list(range(28))}
            metadata=deepcopy(common)|deepcopy(base_metadata)|{'feature_schema':p.FEATURE_SCHEMA,
                'source_sha256':t.source_hashes(),
                'gate_contract':deepcopy(p.PUBLIC_GATE_CONTRACT),'offline_candidate_qualified':True,
                'best_epoch':1,'optimizer_steps':8,
                'final_cpu_sequential_metrics':{'step499_reviewed_zero_histories_bitwise_baseline':True,
                    'step545_reviewed_hold_histories_exact_projected_hold':True,
                    'inference':'sequential_one_frame_hidden_state'},
                'eligible_for_rl_episode_training':False,'independent_generalization_demonstrated':False,
                'closed_loop_success_demonstrated':False,'right28_learned':False}
            model=SimpleNamespace(mean=torch.zeros(284),residual_bound=bounds.clone(),
                left_joint_names=tuple(left),update_anchor_is_previous=anchors.clone(),
                action_contract=deepcopy(p.ACTION_CONTRACT))
            args=SimpleNamespace(source=root/'source',output=root/'output',checkpoint=candidate,
                expected_checkpoint_sha256=sha(candidate),baseline=baseline,qualification=qualification,
                authorization=authorization,authorization_sha256=sha(authorization),
                dataset=[root/n for n in t.RUN_NAMES],seed=9501,name='host_prepare',max_steps=2000)
            if change:change(q,metadata,args,model)
            qualification.write_text(json.dumps(q));args.qualification_sha256=sha(qualification)
            metadata.setdefault('qualification_sha256',args.qualification_sha256)
            metadata.setdefault('root_authorization_sha256',args.authorization_sha256)
            with ExitStack()as stack:
                stack.enter_context(patch.object(entry,'source_preflight',return_value={'names':names,'left':list(range(28))}))
                stack.enter_context(patch.object(entry,'load_baseline',return_value=({'metadata':base_metadata},sha(baseline))))
                stack.enter_context(patch.object(entry,'checkpoint_preflight',return_value={'mock':True}))
                loader=stack.enter_context(patch.object(entry,'load_checkpoint',return_value=(model,metadata)))
                _,_,_,_,identity=entry.prepare(args)
                positional=loader.call_args.args;kwargs=loader.call_args.kwargs
                self.assertEqual(positional[:2],(args.output/'checkpoints/evaluated_policy.pt',args.output/'checkpoints/frozen_v4_baseline.pt'))
                self.assertEqual(kwargs['qualification_path'],args.output/'reviewed_transition_qualification.json')
                self.assertEqual(kwargs['authorization_path'],args.output/'reviewed_training_authorization.json')
                self.assertEqual(kwargs['expected_checkpoint_sha256'],sha(candidate))
                self.assertEqual(kwargs['expected_qualification_sha256'],sha(qualification))
                self.assertEqual(kwargs['expected_authorization_sha256'],sha(authorization))
                self.assertEqual(kwargs['expected_physics_family'],q['physics_family_sha256'])
                for name,digest in identity['source_sha256'].items():
                    self.assertEqual(sha(args.output/'evaluation_source_snapshot'/name),digest)
                self.assertEqual(identity['update_anchor_is_previous'],anchors.tolist())
                self.assertEqual(identity['update_bound_rad'],bounds.tolist())
                self.assertEqual(identity['left_teacher_position_action_steps'],0)
                self.assertFalse(identity['expert_hidden_or_labels_in_actor'])
                self.assertTrue(identity['all_checkpoint_source_dependencies_snapshotted_exactly'])
                self.assertEqual(identity['checkpoint_source_dependency_count'],len(t.source_hashes()))
                return identity

    def test_actual_snapshots_eight_history_scope_and_named_action_contract(self):
        identity=self.invoke()
        self.assertEqual(identity['policy'],'v8')
        self.assertEqual(identity['development_histories'],8)
        self.assertIn('HOLD is previous actual',identity['bound_scope'])
        self.assertEqual(identity['action_contract'],entry.hold_increment_policy_v8.ACTION_CONTRACT)

    def test_changed_policy_or_reviewed_hash_order_and_scope_is_rejected(self):
        edits=[lambda q,m,a,p:setattr(a,'expected_checkpoint_sha256','wrong'),
            lambda q,m,a,p:setattr(a,'authorization_sha256','wrong'),
            lambda q,m,a,p:setattr(a,'dataset',a.dataset[::-1]),
            lambda q,m,a,p:setattr(a,'dataset',a.dataset[:-1]+[a.dataset[0]]),
            lambda q,m,a,p:m.update(qualification_sha256='wrong'),
            lambda q,m,a,p:m.update(root_authorization_sha256='wrong'),
            lambda q,m,a,p:m.update(offline_candidate_qualified=False),
            lambda q,m,a,p:m.update(best_epoch=0),
            lambda q,m,a,p:m['final_cpu_sequential_metrics'].update(step545_reviewed_hold_histories_exact_projected_hold=False),
            lambda q,m,a,p:m.update(closed_loop_success_demonstrated=True)]
        for index,change in enumerate(edits):
            with self.subTest(index=index),self.assertRaises(ValueError):self.invoke(change)

    def test_named_anchor_bounds_actual_tensors_and_feature_order_are_rejected_on_change(self):
        edits=[lambda q,m,a,p:m.update(output_indices=[1,0]+list(range(2,28))),
            lambda q,m,a,p:m.update(feature_schema='wrong'),
            lambda q,m,a,p:setattr(p,'mean',torch.zeros(283)),
            lambda q,m,a,p:setattr(p,'residual_bound',p.residual_bound+1e-6),
            lambda q,m,a,p:setattr(p,'left_joint_names',p.left_joint_names[::-1]),
            lambda q,m,a,p:p.update_anchor_is_previous.fill_(False),
            lambda q,m,a,p:m.update(update_anchor_is_previous=[1]*28),
            lambda q,m,a,p:q.update(update_bound_rad=[.006]*28),
            lambda q,m,a,p:m.update(action_contract={}),
            lambda q,m,a,p:setattr(p,'action_contract',{})]
        for index,change in enumerate(edits):
            with self.subTest(index=index),self.assertRaises(ValueError):self.invoke(change)

    def test_qualification_keeps_failures_false_and_requires_exact_branch_witnesses(self):
        edits=[lambda q,m,a,p:q.update(training_authorized=True),
            lambda q,m,a,p:q.update(source_sha256={}),
            lambda q,m,a,p:q.update(public_gate_contract={}),
            lambda q,m,a,p:q.update(action_contract={}),
            lambda q,m,a,p:q.update(classification_contract={}),
            lambda q,m,a,p:q.update(selection_contract={}),
            lambda q,m,a,p:q.update(original_result_files_modified=True),
            lambda q,m,a,p:q.update(frozen_baseline_checkpoint_sha256='wrong'),
            lambda q,m,a,p:q['branch_coverage'][0].update(active_max_union_gap_rad=1e-10),
            lambda q,m,a,p:q['branch_coverage'][0].update(active_no_float32_branch_witness_elements=1),
            lambda q,m,a,p:q['branch_coverage'][0].update(masked_corrections_gt_1e_6=1)]
        for index,change in enumerate(edits):
            with self.subTest(index=index),self.assertRaises(ValueError):self.invoke(change)


if __name__=='__main__':unittest.main()
