from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,copy,hashlib,json,unittest
import numpy as np
import exception_prefix_dataset_v1 as d

HERE=Path(__file__).resolve().parent
COUNTS=('steps','physical_steps','environment_episode_counter','fully_audited_transition_steps',
        'action_execution_dispatch_attempts','trajectory_rows_recorded','teacher_query_count')

def fixture(count=2):
    result={k:count for k in COUNTS}|{'failure':d.FAILURE,'original_wrist_gate_qualified':False}
    transitions=[{'record_type':'executed_transition','step':i,'transition_finished':True,'terminal':False,
        'failure_reason':None,'original_wrist_gate_qualified':False,'invalid_evidence_fields':[]}for i in range(count)]
    abort={'record_type':'abort','step':count,'reason':d.FAILURE,
        'unfinished_decision':{'step':count,'record_type':'decision_attempt','left_teacher_position_action_executed':False},
        **{k:count for k in COUNTS if k not in ('steps','teacher_query_count')},
        'physics_step_completed_for_unfinished_decision':False,'execution_dispatch_without_counter_advance':False,'invalid_evidence_fields':[]}
    final={'record_type':'evaluation_final','summary':copy.deepcopy(result),'invalid_evidence_fields':[]}
    return result,transitions+[abort,final]

def write(run,result,rows):
    (run/'result.json').write_text(json.dumps(result))
    (run/'evaluation_transitions.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))

class EngineContracts(unittest.TestCase):
    def test_exact_inverse_preserves_every_frozen_engine_byte(self):
        source=d.engine_source()
        for before,after in reversed(d.HOOKS):
            self.assertEqual(source.count(after),1);source=source.replace(after,before)
        self.assertEqual(source,d.prior.engine_source())
        self.assertEqual(len(d.HOOKS),2)

    def test_all_non_loader_function_asts_and_physical_transition_predicates_unchanged(self):
        old=d.prior.engine_source();new=d.engine_source()
        funcs=lambda s:{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        a,b=funcs(old),funcs(new);self.assertEqual(set(a),set(b))
        for name in a:
            if name!='load_run':self.assertEqual(a[name],b[name],name)
        start="            require(kind=='executed_transition'"
        end="    require(any(masks),'No safe candidate transitions')"
        self.assertEqual(old[old.index(start):old.index(end)],new[new.index(start):new.index(end)])

    def test_private_trust_is_narrow_and_does_not_mutate_frozen_registry(self):
        before=dict(d.prior.V8_TRUSTED);old_restore=d.prior.restore_recorded_model
        a=d.private_trust_registry();b=d.private_trust_registry()
        self.assertIsNot(a,b);self.assertEqual(dict(d.prior.V8_TRUSTED),before)
        self.assertIs(d.prior.restore_recorded_model,old_restore)
        self.assertEqual(len(a.V8_TRUSTED),2)
        self.assertNotIn('6280460b32b1d87a7bf2512820ad1b23a83437261bf5881a6c64b441de3d4d5b',a.V8_TRUSTED)

    def test_no_existing_trainer_compatible_entry_or_global_registration(self):
        self.assertFalse(hasattr(d,'load_run'));self.assertTrue(callable(d.load_candidate))
        self.assertEqual(sorted(v['steps']for v in d.RUNS.values()),[1725,1803])
        self.assertEqual({v['hidden_size']for v in d.RUNS.values()},{64,128})
        with self.assertRaises(ValueError):d.trust_run(HERE)

class ExceptionBoundary(unittest.TestCase):
    def test_exact_returned_prefix_excludes_unknown_query_without_increment(self):
        with TemporaryDirectory(dir=HERE)as temp:
            run=Path(temp);result,rows=fixture();write(run,result,rows)
            proof=d.scan_exact_boundary(run,2)
            self.assertEqual(proof['step'],2);self.assertFalse(proof['teacher_query_returned'])
            self.assertFalse(proof['teacher_after_state_known']);self.assertFalse(proof['expert_label_present'])

    def test_invented_label_audit_feature_prediction_or_action_rejected(self):
        result,rows=fixture();original=rows[-2]
        for name in ('original_controller_audit','actor_observation_pre','actual_actor_features','predicted_left_target',
                     'counterfactual_expert_projection_diagnostic_only','execution','normalized_action'):
            with self.subTest(name=name):
                changed=copy.deepcopy(original);changed['unfinished_decision'][name]=None
                with self.assertRaisesRegex(ValueError,'exactly'):d.check_unreturned_abort(changed,2,result)

    def test_counter_done_dispatch_and_invalid_flags_reject(self):
        result,rows=fixture();original=rows[-2]
        mutations=[('step',1),('physical_steps',3),('environment_episode_counter',3),
            ('physics_step_completed_for_unfinished_decision',True),('execution_dispatch_without_counter_advance',True),
            ('invalid_evidence_fields',['bad']),('reason','other failure')]
        for key,value in mutations:
            with self.subTest(key=key):
                changed=copy.deepcopy(original);changed[key]=value
                with self.assertRaises(ValueError):d.check_unreturned_abort(changed,2,result)
        changed=copy.deepcopy(original);changed['unfinished_decision']['left_teacher_position_action_executed']=0
        with self.assertRaises(ValueError):d.check_unreturned_abort(changed,2,result)

    def test_ordering_missing_duplicate_final_and_post_abort_execution_rejected(self):
        result,rows=fixture()
        variants=[rows[:-1],rows+[copy.deepcopy(rows[-1])],rows+[copy.deepcopy(rows[0])],
            rows[:-1]+[copy.deepcopy(rows[0]),rows[-1]],rows[:-1]+[copy.deepcopy(rows[-2]),rows[-1]],
            [rows[1],rows[0],*rows[2:]],rows[1:],rows[:-2]+[rows[-1],rows[-2]]]
        with TemporaryDirectory(dir=HERE)as temp:
            for changed in variants:
                write(Path(temp),result,changed)
                with self.assertRaises(ValueError):d.scan_exact_boundary(Path(temp),2)

    def test_unreturned_query_cannot_be_counted_as_successful_return(self):
        result,rows=fixture();result['teacher_query_count']=3;rows[-1]['summary']['teacher_query_count']=3
        with TemporaryDirectory(dir=HERE)as temp:
            write(Path(temp),result,rows)
            with self.assertRaisesRegex(ValueError,'Final count'):d.scan_exact_boundary(Path(temp),2)

    def test_incomplete_or_earlier_terminal_physical_frame_rejected(self):
        result,rows=fixture()
        for key,value in (('transition_finished',False),('terminal',True),('failure_reason','drop'),
                          ('original_wrist_gate_qualified',True),('invalid_evidence_fields',['nan'])):
            changed=copy.deepcopy(rows);changed[0][key]=value
            with TemporaryDirectory(dir=HERE)as temp:
                write(Path(temp),result,changed)
                with self.assertRaises(ValueError):d.scan_exact_boundary(Path(temp),2)

class PriorActorProof(unittest.TestCase):
    def make(self,work):
        run=work/'run';run.mkdir();script=work/'replay_wide128_v8_actual_diagnostic_20261104.py';script.write_text('# fixed test audit\n')
        detail=work/'proof';detail.mkdir();branch=detail/'branch_replay.jsonl';branch.write_text('{}\n')
        records=[{'record_type':'executed_transition','actual_actor_features':[float(i)]*284,'predicted_left_target':[float(i)]*28}for i in range(2)]
        journal=run/'evaluation_transitions.jsonl';journal.write_text(''.join(json.dumps(r)+'\n'for r in records))
        spec={'steps':2,'checkpoint':'cp','authorization':'auth','actor_report':'proof/report.json',
            'actor_branch_sha256':d.sha(branch),'actor_script_sha256':d.sha(script)}
        evidence={'artifact_manifest_sha256':'manifest','files_sha256':{'evaluation_transitions.jsonl':d.sha(journal)},'source_snapshot_sha256':{'s':'h'}}
        report={'passed':True,'run':str(run),'branch_detail_sha256':d.sha(branch),'source_sha256':{script.name:d.sha(script)},
            'all_actual_features_predictions_native_actions_projected_targets_bitwise_equal':True,
            'actual_completed_steps':2,'returned_teacher_queries':2,'actor_replay_calls':2,'actor_hidden_resets':1,
            'extra_recurrent_forward_calls_for_diagnostics':0,'checkpoint_sha256':'cp','qualification_sha256':d.QSHA,
            'authorization_sha256':'auth','actual_artifact_manifest_sha256':'manifest','actual_files_sha256':evidence['files_sha256'],
            'actual_source_snapshots_sha256':evidence['source_snapshot_sha256']}
        p=detail/'report.json';p.write_text(json.dumps(report));spec['actor_report_sha256']=d.sha(p)
        return run,spec,evidence,branch

    def test_cached_actor_uses_same_features_once_in_order_no_label_input(self):
        with TemporaryDirectory(dir=HERE)as temp:
            w=Path(temp);run,spec,evidence,_=self.make(w)
            with patch.object(d,'WORK',w),patch.object(d.prior.prior.frozen.policy,'features',side_effect=lambda x:np.asarray(x['features'],np.float32)):
                actor=d.RecordedPredictionProof(run,spec,evidence);actor.reset('run')
                for i in range(2):np.testing.assert_array_equal(actor.act({'features':[i]*284},episode_id='run',step=i),[i]*28)
                self.assertEqual(actor.proof['new_gru_calls'],0)
                with self.assertRaises(ValueError):actor.reset('run')
                with self.assertRaises(ValueError):actor.act({'features':[0]*284},episode_id='run',step=2)

    def test_actor_cached_proof_rejects_branch_file_evidence_or_features_tamper(self):
        with TemporaryDirectory(dir=HERE)as temp:
            w=Path(temp);run,spec,evidence,branch=self.make(w)
            with patch.object(d,'WORK',w):
                for mode in ('branch','evidence','actual_bytes'):
                    old=branch.read_bytes();j=run/'evaluation_transitions.jsonl';oldj=j.read_bytes();ev=copy.deepcopy(evidence)
                    if mode=='branch':branch.write_bytes(old+b' ')
                    if mode=='evidence':ev['files_sha256']['evaluation_transitions.jsonl']='0'*64
                    if mode=='actual_bytes':j.write_bytes(oldj+b' ')
                    with self.assertRaises(ValueError):d.RecordedPredictionProof(run,spec,ev)
                    branch.write_bytes(old);j.write_bytes(oldj)
                with patch.object(d.prior.prior.frozen.policy,'features',return_value=np.ones(284,np.float32)):
                    actor=d.RecordedPredictionProof(run,spec,evidence);actor.reset('run')
                    with self.assertRaisesRegex(ValueError,'features'):actor.act({},episode_id='run',step=0)

if __name__=='__main__':unittest.main()
