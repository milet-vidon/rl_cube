"""Independent prepare-boundary tests with a mocked qualified model loader."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import evaluate_state_zero_gated_v7 as entry

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class PrepareBoundaryReview(unittest.TestCase):
 def invoke(self,change=None):
  with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
   root=Path(temporary);candidate=root/'candidate.pt';baseline=root/'baseline.pt';qual=root/'q.json';auth=root/'auth.json'
   candidate.write_bytes(b'mocked candidate; no trained model claim');baseline.write_bytes(b'mocked baseline')
   auth.write_text('{"scope":"mocked loader boundary, not real authorization"}')
   left=list(entry.zero_gated_residual_policy_v7.EXPECTED_LEFT_NAMES)
   names=left+[n.replace('_L','_R').replace('left_','right_') for n in left]
   bounds=torch.tensor(entry.zero_gated_residual_policy_v7.candidate_bounds(left),dtype=torch.float32)
   q={'schema':'v7_five_history_bc_integrity_candidate_sidecar_v1','training_authorized':False,
    'candidate_bc_transitions_passed_integrity':True,'eligible_under_declared_v7_candidate_bounds':True,
    'zero_class_definition_approved':True,'eligible_for_rl_episode_training':False,'pure_student_episode_success_claim':False,
    'independent_generalization_claim':False,'original_result_files_modified':False,
    'source_sha256':entry.inspect_residual_dataset_v7.source_hashes(),
    'current_candidate_gate':deepcopy(entry.zero_gated_residual_policy_v7.PUBLIC_GATE_CONTRACT),
    'classification_contract':deepcopy(entry.zero_gated_residual_policy_v7.CLASSIFICATION_CONTRACT),
    'active_max_unreachable_distance_rad':0.,'frozen_baseline_checkpoint_sha256':sha(baseline),
    'physics_family_sha256':'mock family','current_candidate_bounds_by_name':dict(zip(left,bounds.tolist()))}
   base_meta={'joint_names':names,'output_indices':list(range(28))}
   meta=deepcopy(base_meta)|{'feature_schema':entry.zero_gated_residual_policy_v7.FEATURE_SCHEMA,
    'offline_candidate_qualified':True,'best_epoch':1,'optimizer_steps':5,
    'final_cpu_sequential_metrics':{'step499_reviewed_zero_histories_bitwise_baseline':True,'inference':'sequential_one_frame_hidden_state'},
    'eligible_for_rl_episode_training':False,'independent_generalization_demonstrated':False,'closed_loop_success_demonstrated':False,'right28_learned':False}
   model=SimpleNamespace(mean=torch.zeros(284),residual_bound=bounds.clone())
   args=SimpleNamespace(source=root/'source',output=root/'output',checkpoint=candidate,expected_checkpoint_sha256=sha(candidate),
    baseline=baseline,qualification=qual,authorization=auth,authorization_sha256=sha(auth),
    dataset=[root/n for n in entry.inspect_residual_dataset_v7.RUN_NAMES],seed=9501,name='host_boundary',max_steps=2000)
   if change:change(q,meta,args,model)
   qual.write_text(json.dumps(q));args.qualification_sha256=sha(qual)
   meta.setdefault('qualification_sha256',args.qualification_sha256);meta.setdefault('root_authorization_sha256',args.authorization_sha256)
   preflight={'names':names,'left':list(range(28))}
   with ExitStack() as stack:
    stack.enter_context(patch.object(entry,'source_preflight',return_value=preflight))
    stack.enter_context(patch.object(entry,'load_baseline',return_value=({'metadata':base_meta},sha(baseline))))
    stack.enter_context(patch.object(entry,'checkpoint_preflight',return_value={'mock':True}))
    loader=stack.enter_context(patch.object(entry,'load_checkpoint',return_value=(model,meta)))
    result=entry.prepare(args);identity=result[-1]
    positional=loader.call_args.args;kwargs=loader.call_args.kwargs
    self.assertEqual(positional[:2],(args.output/'checkpoints/evaluated_policy.pt',args.output/'checkpoints/frozen_v4_baseline.pt'))
    self.assertEqual(kwargs['qualification_path'],args.output/'reviewed_transition_qualification.json')
    self.assertEqual(kwargs['authorization_path'],args.output/'reviewed_training_authorization.json')
    self.assertEqual(kwargs['expected_checkpoint_sha256'],sha(candidate))
    self.assertEqual(kwargs['expected_qualification_sha256'],sha(qual));self.assertEqual(kwargs['expected_authorization_sha256'],sha(auth))
    self.assertEqual(kwargs['expected_physics_family'],q['physics_family_sha256'])
    for name,digest in identity['source_sha256'].items():self.assertEqual(sha(args.output/'evaluation_source_snapshot'/name),digest)
    self.assertEqual(identity['training_authorization_sha256'],sha(auth));self.assertEqual(identity['left_teacher_position_action_steps'],0)
    self.assertEqual(identity['residual_bound_rad'],bounds.tolist())
    return identity
 def test_success_uses_immutable_policy_baseline_qualification_authorization_snapshots(self):
  identity=self.invoke();self.assertEqual(identity['policy'],'v7');self.assertEqual(identity['development_histories'],5)
 def test_metadata_schema_joint_order_bounds_and_source_claims_rejected(self):
  edits=[lambda q,m,a,p:m.update(qualification_sha256='wrong'),lambda q,m,a,p:m.update(root_authorization_sha256='wrong'),
   lambda q,m,a,p:m.update(output_indices=[1,0]+list(range(2,28))),lambda q,m,a,p:m.update(feature_schema='wrong'),
   lambda q,m,a,p:setattr(p,'mean',torch.zeros(283)),lambda q,m,a,p:setattr(p,'residual_bound',p.residual_bound+1e-6),
   lambda q,m,a,p:m.update(offline_candidate_qualified=False),lambda q,m,a,p:m.update(final_cpu_sequential_metrics={})]
  for i,edit in enumerate(edits):
   with self.subTest(case=i),self.assertRaises(ValueError):self.invoke(edit)
 def test_snapshot_pins_and_exact_dataset_order_rejected(self):
  edits=[lambda q,m,a,p:setattr(a,'expected_checkpoint_sha256','wrong'),lambda q,m,a,p:setattr(a,'authorization_sha256','wrong'),
   lambda q,m,a,p:setattr(a,'dataset',a.dataset[::-1]),lambda q,m,a,p:setattr(a,'dataset',a.dataset[:-1]+[a.dataset[0]])]
  for i,edit in enumerate(edits):
   with self.subTest(case=i),self.assertRaises(ValueError):self.invoke(edit)
 def test_changed_qualification_semantics_rejected(self):
  edits=[lambda q,m,a,p:q.update(training_authorized=True),lambda q,m,a,p:q.update(source_sha256={}),
   lambda q,m,a,p:q.update(current_candidate_gate={}),lambda q,m,a,p:q.update(classification_contract={}),
   lambda q,m,a,p:q.update(frozen_baseline_checkpoint_sha256='wrong'),lambda q,m,a,p:q.update(active_max_unreachable_distance_rad=1e-10)]
  for i,edit in enumerate(edits):
   with self.subTest(case=i),self.assertRaises(ValueError):self.invoke(edit)

if __name__=='__main__':unittest.main()
