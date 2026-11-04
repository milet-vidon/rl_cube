from pathlib import Path
import hashlib,json

p=Path(__file__).resolve().parent
names=('state_hold_increment_eval_runtime_v8.py','evaluate_state_hold_increment_v8.py',
       'train_hold_increment_v8.py','test_v8_eval_prepare_independent.py',
       'v8_eval_independent_host_tests_v2.log')
report={'schema':'v8_eval_entry_independent_review_v1','blocking_findings':[],
 'source_sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest()for n in names},
 'checks':{'remote_host_tests_passed':24,'original_tests':20,'additional_independent_tests':4,
 'extra_resealed_metadata_negative_variants':8,'runtime_inverse_route_equals_frozen_v7':True,
 'main_execution_and_cleanup_ast_preserved':True,'eight_ordered_histories_required':True,
 'step499_exact_base_histories':7,'step545_projected_hold_histories':2,
 'step545_required_joints':['left_ring_MCP_joint','left_ring_PIP_joint','left_ring_DIP_joint'],
 'named_update_anchor_actual_tensor_and_metadata_bound':True,
 'hold_uses_previous_actual_without_moving_baseline_bound':True,
 'checkpoint_q_auth_snapshots_passed_to_verified_loader':True,
 'loader_requires_offline_qualified_by_default_not_disabled':True,
 'all_18_training_dependencies_have_exact_source_snapshot':True,
 'original_false_scope_flags_retained':True},
 'environment_note':'Initial host launch lacked CODE on PYTHONPATH and failed imports; preserved v8_eval_independent_host_tests.log. Re-run v2 log used activated project environment and CODE:WORK path, 24 PASS.',
 'limits':['prepare tests mock checkpoint loader; no real trained-checkpoint preflight claimed',
 'no Isaac or hardware executed','not a closed-loop success, RL or generalization qualification'],
 'reviewed_loader_behavior':['External checkpoint/q/auth SHA and exact embedded text checked',
 'Eight histories verified through original evidence consumers',
 'Canonical frozen tensors, gates, bounds, anchor mask and original V4 baseline compared bitwise',
 'Trainable tensors require exact keys/shapes/dtypes, finite values and strict state load',
 'Final CPU 499/545 regressions and improvement metrics required']}
out=p/'v8_eval_entry_independent_review_20261104.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(out.name,hashlib.sha256(out.read_bytes()).hexdigest())
