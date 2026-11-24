"""Fail-closed boundaries for a diagnostic which cannot admit training data."""
import copy,json,tempfile,unittest
from pathlib import Path
import replay_wide128_v8_actual_diagnostic_20261104 as d

def fixture(count=1):
 keys=('steps','physical_steps','environment_episode_counter','fully_audited_transition_steps','action_execution_dispatch_attempts','trajectory_rows_recorded','teacher_query_count')
 result={k:count for k in keys}|{'failure':d.FAILURE,'original_wrist_gate_qualified':False}
 abort={k:count for k in keys if k not in ('steps','teacher_query_count')}|{'record_type':'abort','step':count,'reason':d.FAILURE,
  'unfinished_decision':{'step':count,'record_type':'decision_attempt','left_teacher_position_action_executed':False},
  'physics_step_completed_for_unfinished_decision':False,'execution_dispatch_without_counter_advance':False,'invalid_evidence_fields':[]}
 return result,abort

class BoundaryTests(unittest.TestCase):
 def test_unreturned_query_has_no_label_and_no_actor(self):
  result,abort=fixture();checked=d.check_unreturned_abort(abort,1,result)
  for key in ('teacher_query_returned','expert_label_present','actor_replayed','physics_executed','training_qualification'):self.assertIs(checked[key],False)
 def test_resealed_synthetic_query_or_action_is_rejected(self):
  for field in ('original_controller_audit','counterfactual_expert_projection_diagnostic_only','predicted_left_target','execution','actual_actor_features'):
   result,abort=fixture();abort['unfinished_decision'][field]={}
   with self.subTest(field=field),self.assertRaises(ValueError):d.check_unreturned_abort(abort,1,result)
 def test_unrecorded_physics_or_counter_mismatch_rejected(self):
  for field,value in [('physics_step_completed_for_unfinished_decision',True),('execution_dispatch_without_counter_advance',True),('physical_steps',2)]:
   result,abort=fixture();abort[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):d.check_unreturned_abort(abort,1,result)
 def test_stream_rejects_any_execution_after_abort_or_summary(self):
  result,abort=fixture();row={'record_type':'executed_transition','step':0,'transition_finished':True,'terminal':False,'failure_reason':None,'original_wrist_gate_qualified':False,'invalid_evidence_fields':[]}
  final={'record_type':'evaluation_final','summary':result}
  with tempfile.TemporaryDirectory()as td:
   path=Path(td)/'evaluation_transitions.jsonl'
   path.write_text(''.join(json.dumps(v)+'\n'for v in(row,abort,final)))
   records,_,_=d.load_completed_records(Path(td),result);self.assertEqual(len(records),1)
   for stream in ((row,abort,row,final),(row,abort,final,final),(row,final)):
    path.write_text(''.join(json.dumps(v)+'\n'for v in stream))
    with self.assertRaises(ValueError):d.load_completed_records(Path(td),result)

if __name__=='__main__':unittest.main()
