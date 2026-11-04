"""Independent V9 trainer delta and checkpoint guards; synthetic host fixtures only."""
from copy import deepcopy
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import ast,hashlib,json,tempfile,unittest
import torch
import train_arm_hold_increment_v9 as trainer
from test_train_arm_hold_increment_v9 import fixtures as author_fixtures
from test_arm_hold_increment_policy_v9 import force_modes
from test_phase_residual_v5 import baseline

def fixtures(length=561):
    model,episodes=author_fixtures(length)
    saved=baseline();saved['metadata']['joint_names']=episodes[0]['joint_names']
    for data in episodes:
        data['parameters']=deepcopy(data['parameters']);p=data['parameters'];p['dtype']='torch.float32'
        p['max_target_delta']=[[.025]*56];p['step_limit']=[[.025]*56]
    return saved,model,episodes

def metadata_fixture(saved,model,episodes):
    sources={'synthetic':'not_real_source_evidence'}
    for i,d in enumerate(episodes):
        d['consumer_schema']='mixture' if i==0 else ('legacy' if i<5 else 'registry')
    q={'schema':trainer.QUAL_SCHEMA,'training_authorized':False,'source_sha256':sources,
        'eligible_for_development_bc_transition_supervision':True,
        'physics_family_sha256':trainer.FAMILY_SHA,'frozen_baseline_checkpoint_sha256':trainer.BASELINE_SHA,
        'public_gate_contract':trainer.policy.PUBLIC_GATE_CONTRACT,'action_contract':trainer.policy.ACTION_CONTRACT,
        'classification_contract':trainer.policy.CLASSIFICATION_CONTRACT,'selection_contract':trainer.SELECTION_CONTRACT,
        'datasets':[{'evidence':d['evidence']}for d in episodes],
        'dataset_consumers':[d['consumer_schema']for d in episodes],
        'left_joint_names':list(model.left_joint_names),'update_bound_rad':model.residual_bound.tolist(),
        'update_anchor_is_previous':model.update_anchor_is_previous.tolist(),
        'normalization_mean':model.mean.tolist(),'normalization_std':model.std.tolist(),
        'original_arm_projection_proofs':[trainer.policy.verify_projection_arm_bounds(d['parameters'],d['left_indices'])for d in episodes],
        **{k:False for k in ('eligible_for_rl_episode_training','pure_student_episode_success_claim',
                            'independent_generalization_claim','original_result_files_modified')}}
    text=json.dumps(q);digest=hashlib.sha256(text.encode()).hexdigest()
    with patch.object(trainer,'source_hashes',return_value=sources):
        m=trainer.checkpoint_metadata(model,episodes,q,text,digest,'{}',hashlib.sha256(b'{}').hexdigest(),
            {'epochs':2,'hidden_size':model.hidden_size})
    m.update(best_epoch=1,optimizer_steps=9,offline_candidate_qualified=True,
        initial_cpu_sequential_metrics={'classification_error_count':2,'hard_decoded_projected_fit':{'rmse_rad':.01}},
        final_cpu_sequential_metrics={'inference':'sequential_one_frame_hidden_state',
            'classification_error_count':1,'hard_decoded_projected_fit':{'rmse_rad':.001},
            'step499_reviewed_zero_histories_bitwise_baseline':True,
            'step545_reviewed_hold_histories_exact_projected_hold':True,
            'step560_reviewed_index_history_exact_projected_hold':True,'step560_regression_violation_count':0,
            'step499_regression_violation_count':0,'step545_regression_violation_count':0})
    return q,m,sources

class V9TrainerPeer(unittest.TestCase):
    def test_unchanged_optimizer_objective_and_device_proof_functions(self):
        def funcs(path):
            return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(trainer.v8.__file__),funcs(trainer.__file__)
        for name in ('training_config','authorization_template','validate_authorization','validate_qualification','verify_device_definition','atomic_torch_save','save_candidate'):
            self.assertEqual(old[name],new[name],name)

    def test_boundary_reset_and_exact_561_single_frame_calls(self):
        _,m,e=fixtures(570);resets=[];original=m.forward_with_diagnostics
        def capture(x,previous,hidden=None):
            resets.append(hidden is None);return original(x,previous,hidden)
        with patch.object(m,'forward_with_diagnostics',side_effect=capture):r=trainer.assess(e,m,streaming=True,boundary_only=True)
        self.assertEqual(len(resets),9*561)
        self.assertEqual([i for i,x in enumerate(resets)if x],[i*561 for i in range(9)])
        self.assertTrue(all(x['frames_processed']==561 for x in r['runs']))

    def test_hold_mode_passes_old_and_new_hold_boundaries(self):
        _,m,e=fixtures();force_modes(m,trainer.policy.HOLD)
        r=trainer.assess(e,m,streaming=True)
        self.assertEqual(r['step545_regression_violation_count'],0)
        self.assertEqual(r['step560_regression_violation_count'],0)
        self.assertEqual(r['runs'][-1]['step560']['projection_exact_HOLD'],[True]*3)

    def test_changed_arm_anchor_or_any_history_cap_is_rejected(self):
        _,m,e=fixtures();trainer.verify_model_named_contract(m,e)
        m.update_anchor_is_previous[0]=False
        with self.assertRaisesRegex(ValueError,'anchor mask'):trainer.verify_model_named_contract(m,e)
        _,m,e=fixtures();e[-1]['parameters']['command_velocity_limits'][0][0]=.1
        with self.assertRaisesRegex(ValueError,'projection caps'):trainer.verify_model_named_contract(m,e)

    def validation_context(self,stack,sources,episodes,calls):
        stack.enter_context(patch.object(trainer,'source_hashes',return_value=sources))
        stack.enter_context(patch.object(trainer,'validate_authorization'))
        for i,consumer in enumerate((trainer.prior.mixture_reader,trainer.prior.reader,trainer.v8.registry,trainer.registry)):
            stack.enter_context(patch.object(consumer,'verify_saved_datasets',side_effect=lambda m,paths,i=i:calls.append((i,Path(paths[0]).name))))
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path.name=='projection_parameters.json':
                row=next(d for d in episodes if d['episode_id']==path.parent.name)
                return json.dumps(row['parameters'])
            return original(path,*args,**kwargs)
        stack.enter_context(patch.object(Path,'read_text',read))

    def test_metadata_dispatch_and_real_cap_proof_readback(self):
        saved,m,e=fixtures();_,md,sources=metadata_fixture(saved,m,e);calls=[]
        with ExitStack()as st:
            self.validation_context(st,sources,e,calls)
            trainer.verify_metadata(md,saved,trainer.BASELINE_SHA,trainer.RUN_NAMES,trainer.FAMILY_SHA)
        self.assertEqual(calls,[(0 if i==0 else 1 if i<5 else 2 if i<8 else 3,n)for i,n in enumerate(trainer.RUN_NAMES)])
        md['original_arm_projection_proofs'][0]['update_bounds_rad'][0]=.08
        with ExitStack()as st:
            self.validation_context(st,sources,e,[])
            with self.assertRaisesRegex(ValueError,'projection cap changed'):
                trainer.verify_metadata(md,saved,trainer.BASELINE_SHA,trainer.RUN_NAMES,trainer.FAMILY_SHA)

    def test_metadata_requires_nine_updates_per_epoch_and_exact_hidden(self):
        saved,m,e=fixtures();_,md,sources=metadata_fixture(saved,m,e)
        for key,value in (('optimizer_steps',8),('best_epoch',3),('hidden_size',9)):
            changed=deepcopy(md);changed[key]=value
            with ExitStack()as st:
                self.validation_context(st,sources,e,[])
                with self.subTest(key=key),self.assertRaises(ValueError):
                    trainer.verify_metadata(changed,saved,trainer.BASELINE_SHA,trainer.RUN_NAMES,trainer.FAMILY_SHA)

    def test_loader_rejects_missing_new560_gate_even_with_other_metrics_qualified(self):
        saved,m,e=fixtures();_,md,_=metadata_fixture(saved,m,e)
        for flag,count in ((False,0),(True,1)):
            altered=deepcopy(md);altered['final_cpu_sequential_metrics']['step560_reviewed_index_history_exact_projected_hold']=flag
            altered['final_cpu_sequential_metrics']['step560_regression_violation_count']=count
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as folder:
                d=Path(folder);cp=d/'model.pt';q=d/'q.json';auth=d/'a.json'
                q.write_text(altered['qualification_text']);auth.write_text(altered['root_authorization_text'])
                torch.save({'model':m.state_dict(),'metadata':altered},cp)
                with patch.object(trainer,'verify_metadata'),patch.object(trainer.prior.frozen,'load_baseline',return_value=(saved,trainer.BASELINE_SHA)):
                    with self.assertRaisesRegex(ValueError,'final CPU development regressions'):
                        trainer.load_checkpoint(cp,'unused',trainer.RUN_NAMES,expected_physics_family=trainer.FAMILY_SHA,
                            expected_checkpoint_sha256=trainer.sha(cp),qualification_path=q,expected_qualification_sha256=trainer.sha(q),
                            authorization_path=auth,expected_authorization_sha256=trainer.sha(auth))

if __name__=='__main__':unittest.main()
