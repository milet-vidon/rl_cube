"""Author checks for the V10 data-only extension; no real optimizer or simulator."""
from copy import deepcopy
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import ast,hashlib,json,unittest
import torch
import train_arm_hold_increment_v10 as t
import test_train_arm_hold_increment_v9_peer as old_tests

torch.set_num_threads(1)


def fixtures(length=561):
    saved,model,episodes=old_tests.fixtures(length)
    for name in t.RUN_NAMES[9:]:
        data=deepcopy(episodes[0]);data['episode_id']=name;data['evidence']={'directory':name}
        # New history labels remain supervision, not seven-history BASE regressions.
        data['labels'][499,16:24]+=.005
        episodes.append(data)
    return saved,model,episodes


def metadata_fixture(saved,model,episodes):
    with patch.object(old_tests,'trainer',t):
        # Reuse the original complete metadata fixture, adding only V10 fields.
        original=t.checkpoint_metadata
        def create(m,e,q,text,digest,*args):
            q.update(data_protocol=deepcopy(t.DATA_PROTOCOL),total_eligible_frames=13051,
                actual_history_arrays_sha256={n:t.coverage.array_hashes(d)for n,d in zip(t.RUN_NAMES,e)},
                reviewed_input_bindings={'synthetic':'separately mocked evidence'})
            text=json.dumps(q,sort_keys=True);digest=hashlib.sha256(text.encode()).hexdigest()
            return original(m,e,q,text,digest,*args)
        with patch.object(t,'checkpoint_metadata',side_effect=create):
            q,metadata,sources=old_tests.metadata_fixture(saved,model,episodes)
    metadata['optimizer_steps']=11
    return q,metadata,sources


def validation_context(stack,sources,episodes,calls):
    stack.enter_context(patch.object(t,'source_hashes',return_value=sources))
    stack.enter_context(patch.object(t,'validate_authorization'))
    stack.enter_context(patch.object(t,'verify_dataset_evidence',side_effect=lambda i,p,e:calls.append((i,Path(p).name))))
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path.name=='projection_parameters.json':
            return json.dumps(next(d['parameters']for d in episodes if d['episode_id']==path.parent.name))
        return original(path,*args,**kwargs)
    stack.enter_context(patch.object(Path,'read_text',read))


class V10DataExtensionTests(unittest.TestCase):
    def test_fixed_order_new_schema_and_unchanged_selection_scope(self):
        self.assertEqual(t.RUN_NAMES,t.v9.RUN_NAMES+tuple(t.coverage.prefix.RUNS))
        self.assertEqual(len(t.RUN_NAMES),11)
        self.assertNotEqual(t.CHECKPOINT_SCHEMA,t.policy.SCHEMA)
        self.assertIs(t.policy,t.v9.policy)
        self.assertEqual(t.SELECTION_CONTRACT,t.v9.SELECTION_CONTRACT)
        self.assertEqual(t.DATA_PROTOCOL['ordered_history_names'],list(t.RUN_NAMES))
        self.assertFalse(t.DATA_PROTOCOL['unreturned_query_after_state_or_label_fabricated'])

    def test_unchanged_assessor_objective_device_and_training_logic(self):
        def functions(file):
            return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(file).read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=functions(t.v9.__file__),functions(t.__file__)
        for name in ('build_model','dataset_entry','verify_model_named_contract','validate_qualification',
                     'training_config','authorization_template','validate_authorization','assess','selection_key',
                     'verify_device_definition','atomic_torch_save','save_candidate'):
            self.assertEqual(old[name],new[name],name)
        source=Path(t.__file__).read_text().replace('V10_DEVELOPMENT_FIT','V9_DEVELOPMENT_FIT')
        train=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='train')
        self.assertEqual(old['train'],ast.dump(train,include_attributes=False))

    def test_evidence_routes_original_nine_and_new_two_without_actor_load(self):
        calls=[]
        def consumer(index):
            return SimpleNamespace(verify_saved_datasets=lambda m,paths:calls.append(('old',index,paths,m)))
        def new(evidence,directory):calls.append(('new',directory,evidence));return evidence
        with patch.object(t.v9,'dataset_consumer',side_effect=consumer),patch.object(t.coverage,'verify_saved_exception_evidence',side_effect=new),\
             patch.object(t.coverage.prefix,'load_candidate',side_effect=AssertionError('data/actor reload forbidden')):
            for i,name in enumerate(t.RUN_NAMES):t.verify_dataset_evidence(i,name,{'directory':name})
        self.assertEqual([x[:2]for x in calls[:9]],[('old',i)for i in range(9)])
        self.assertEqual([x[:2]for x in calls[9:]],[('new',n)for n in t.RUN_NAMES[9:]])
        for invalid in (-1,11,True,1.5):
            with self.assertRaises(ValueError):t.verify_dataset_evidence(invalid,'unused',{})
        with patch.object(t.coverage,'verify_saved_exception_evidence',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'evidence changed'):t.verify_dataset_evidence(9,'unused',{})

    def test_loader_rejects_wrong_order_or_noncanonical_paths_before_data_read(self):
        with patch.object(t.coverage,'load_histories',side_effect=AssertionError('should not read')):
            for paths in (t.RUN_NAMES[::-1],t.RUN_NAMES[:-1],t.RUN_NAMES):
                with self.assertRaises(ValueError):t.load_histories(paths,{},'unused')

    def test_eleven_full_histories_reset_once_each_without_relabelling_new_boundaries(self):
        _,model,episodes=fixtures();before=[d['labels'].tobytes()for d in episodes]
        calls=[];original=model.forward_with_diagnostics
        def forward(x,previous,hidden=None):calls.append(hidden is None);return original(x,previous,hidden)
        with patch.object(model,'forward_with_diagnostics',side_effect=forward):
            result=t.assess(episodes,model,streaming=True,boundary_only=True)
        self.assertEqual(len(calls),11*561)
        self.assertEqual([i for i,v in enumerate(calls)if v],[i*561 for i in range(11)])
        self.assertEqual(sum(r['step499_zero_regression_required']for r in result['runs']),7)
        self.assertEqual(sum(r['step545_hold_regression_required']for r in result['runs']),2)
        self.assertEqual(sum(r['step560_index_hold_regression_required']for r in result['runs']),1)
        self.assertEqual(result['step499_regression_violation_count'],0)
        self.assertEqual(result['confusion_matrix_true_by_predicted'][2][0],24)
        self.assertEqual(before,[d['labels'].tobytes()for d in episodes])
        for row in result['runs'][9:]:
            self.assertFalse(row['step499_zero_regression_required'])
            self.assertFalse(row['step545_hold_regression_required'])
            self.assertFalse(row['step560_index_hold_regression_required'])

    def test_metadata_roundtrip_sorted_json_and_eleven_dispatch(self):
        saved,model,episodes=fixtures();_,metadata,sources=metadata_fixture(saved,model,episodes);calls=[]
        with ExitStack()as stack:
            validation_context(stack,sources,episodes,calls)
            t.verify_metadata(metadata,saved,t.BASELINE_SHA,t.RUN_NAMES,t.FAMILY_SHA)
        self.assertEqual(calls,list(enumerate(t.RUN_NAMES)))

    def test_metadata_rejects_old_protocol_counter_and_changed_arrays(self):
        saved,model,episodes=fixtures();_,metadata,sources=metadata_fixture(saved,model,episodes)
        for key,value in (('schema',t.v9.policy.SCHEMA),('model_schema','other'),('optimizer_steps',9),('best_epoch',3),('hidden_size',9)):
            changed=deepcopy(metadata);changed[key]=value
            with ExitStack()as stack:
                validation_context(stack,sources,episodes,[])
                with self.subTest(key=key),self.assertRaises(ValueError):t.verify_metadata(changed,saved,t.BASELINE_SHA,t.RUN_NAMES,t.FAMILY_SHA)
        changed=deepcopy(metadata);changed['actual_history_arrays_sha256'][t.RUN_NAMES[-1]]['labels']='tampered'
        with ExitStack()as stack:
            validation_context(stack,sources,episodes,[])
            with self.assertRaisesRegex(ValueError,'array evidence changed'):t.verify_metadata(changed,saved,t.BASELINE_SHA,t.RUN_NAMES,t.FAMILY_SHA)
        with self.assertRaisesRegex(ValueError,'checkpoint schema'):
            t.v9.verify_metadata(metadata,saved,t.BASELINE_SHA,t.RUN_NAMES,t.FAMILY_SHA)

    def test_source_graph_keeps_original_twenty_four_plus_new_trainer(self):
        with patch.object(t.coverage,'source_hashes',return_value={str(i):'unchanged'for i in range(24)}):
            sources=t.source_hashes();self.assertEqual(len(sources),25)
            self.assertEqual(sources[Path(t.__file__).name],t.sha(t.__file__))
        with patch.object(t.coverage,'source_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'25 modules'):t.source_hashes()

    def test_completed_audit_is_only_version_and_trainer_route_delta(self):
        root=Path(__file__).parent
        old=(root/'audit_completed_arm_hold_increment_v9.py').read_text()
        new=(root/'audit_completed_arm_hold_increment_v10.py').read_text()
        inverse=new.replace('import train_arm_hold_increment_v10 as t','import train_arm_hold_increment_v9 as t')\
            .replace('completed_v10_eleven_history_cpu_product_replay_v1','completed_v9_nine_history_cpu_product_replay_v1')\
            .replace('V10_COMPLETED_PRODUCT_AUDIT','V9_COMPLETED_PRODUCT_AUDIT').replace('Eleven previously reviewed','Nine previously reviewed')
        self.assertEqual(inverse,old)


if __name__=='__main__':unittest.main()
