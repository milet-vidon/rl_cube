"""Author regressions for the nine-history V9 trainer; no real optimization."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib,json,random,tempfile,unittest
import numpy as np
import torch
import train_arm_hold_increment_v9 as t
from test_arm_hold_increment_policy_v9 import policy
from test_zero_gated_residual_policy_v7 import frame, PARAMETERS, joint_safety
from test_phase_residual_v5 import baseline

torch.set_num_threads(1)


def fixtures(length=561):
    saved=baseline();names=list(t.policy.EXPECTED_LEFT_NAMES)+['right_joint_'+str(i)for i in range(28)]
    saved['metadata']['joint_names']=names
    model=t.policy.HoldIncrementPolicy(saved,np.zeros(284),np.ones(284),t.policy.candidate_bounds(names[:28]),8,left_joint_names=names[:28])
    parameters=deepcopy(PARAMETERS);parameters['joint_names']=names
    episodes=[];ring=[names.index(n)for n in t.HOLD_BOUNDARY_JOINTS];index=[names.index(n)for n in t.INDEX_HOLD_JOINTS]
    for name in t.RUN_NAMES:
        x=frame().repeat(1,length,1);x[:,:499]=frame('squeeze',71/84)
        previous=torch.zeros(1,length,28)
        if name in t.HOLD_BOUNDARY_RUNS:previous[0,545,ring]=.02
        if name in t.INDEX_HOLD_RUNS:previous[0,560,index]=.02
        with torch.no_grad():labels=t.policy.project_left_targets(model.baseline_targets(x,previous),previous,parameters,list(range(28)),joint_safety.project_target)
        if name in t.HOLD_BOUNDARY_RUNS:labels[0,545,ring]=previous[0,545,ring]
        if name in t.INDEX_HOLD_RUNS:labels[0,560,index]=previous[0,560,index]
        if name==t.old_inspection.RUN_NAMES[1]:labels[0,499,16:24]+=.005
        episodes.append({'x':x[0].numpy(),'previous_left':previous[0].numpy(),'labels':labels[0].numpy(),
            'mask':np.ones(length,dtype=bool),'parameters':parameters,'left_indices':list(range(28)),
            'joint_names':names,'projector':joint_safety.project_target,'episode_id':name,
            'executed_sources':['student']*length,'evidence':{'directory':name}})
    return model,episodes


class V9TrainingContracts(unittest.TestCase):
    def test_authorization_pins_new_sources_and_objective(self):
        config={'epochs':3,'regression_weight_rad2':1e6,'cross_entropy_weight':1.}
        with patch.object(t,'source_hashes',return_value={'trainer':'frozen-v9'}):
            value=t.authorization_template('qualified',config);text=json.dumps(value);digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
            t.validate_authorization(text,digest(text),'qualified',config)
            with self.assertRaisesRegex(ValueError,'exact training definition'):t.validate_authorization(text,digest(text),'qualified',config|{'epochs':4})
            value['approved_for_development_bc_training']=False;text=json.dumps(value)
            with self.assertRaises(ValueError):t.validate_authorization(text,digest(text),'qualified',config)

    def test_order_routes_all_nine_and_keeps_old_boundaries(self):
        self.assertEqual(len(t.RUN_NAMES),9);self.assertEqual(t.ZERO_BOUNDARY_RUNS,t.v8.ZERO_BOUNDARY_RUNS)
        self.assertEqual(t.HOLD_BOUNDARY_RUNS,t.v8.HOLD_BOUNDARY_RUNS);self.assertEqual(t.INDEX_HOLD_RUNS,t.coverage.NEW_RUNS)
        self.assertIs(t.dataset_consumer(0),t.prior.mixture_reader)
        for i in range(1,5):self.assertIs(t.dataset_consumer(i),t.prior.reader)
        for i in range(5,8):self.assertIs(t.dataset_consumer(i),t.v8.registry)
        self.assertIs(t.dataset_consumer(8),t.registry)
        for i in(-1,9,1.2):
            with self.assertRaises(ValueError):t.dataset_consumer(i)
        with self.assertRaisesRegex(ValueError,'nine ordered'):t.load_histories(t.RUN_NAMES[::-1],{},'unused')

    def test_three_development_boundaries_precede_class_error_and_rmse(self):
        def row(a,b,c,d,e):return {'step499_regression_violation_count':a,'step545_regression_violation_count':b,
            'step560_regression_violation_count':c,'classification_error_count':d,'hard_decoded_projected_fit':{'rmse_rad':e}}
        for left,right in [((0,6,3,100,1),(1,0,0,0,0)),((0,0,3,100,1),(0,1,0,0,0)),
            ((0,0,0,100,1),(0,0,1,0,0)),((0,0,0,1,1),(0,0,0,2,0))]:
            self.assertLess(t.selection_key(row(*left)),t.selection_key(row(*right)))

    def test_full_history_stream_preserves_all_labels_and_reports_exact_hold(self):
        model,episodes=fixtures();result=t.assess(episodes,model,streaming=True,boundary_only=True)
        self.assertEqual(result['step499_regression_violation_count'],0)
        self.assertEqual(result['step545_regression_violation_count'],6)
        self.assertEqual(result['step560_regression_violation_count'],3)
        self.assertEqual(sum(r['step499_zero_regression_required']for r in result['runs']),7)
        self.assertEqual(sum(r['step545_hold_regression_required']for r in result['runs']),2)
        self.assertEqual(sum(r['step560_index_hold_regression_required']for r in result['runs']),1)
        self.assertEqual(result['runs'][-1]['step560']['expert_exact_HOLD'],[True]*3)
        self.assertEqual(result['confusion_matrix_true_by_predicted'],[[9*62*28-17,0,0],[9,0,0],[8,0,0]])
        self.assertGreater(result['label_selected_projected_fit']['rmse_rad'],0) # V5 recovery is never discarded.

    def test_excluded_or_relabelled_560_cannot_define_a_boundary(self):
        model,episodes=fixtures();episodes[-1]['mask'][560]=False
        with self.assertRaisesRegex(ValueError,'560 index HOLD frame is excluded'):t.assess(episodes,model,streaming=True,boundary_only=True)
        episodes[-1]['mask'][560]=True;episodes[-1]['labels'][560,episodes[-1]['joint_names'].index(t.INDEX_HOLD_JOINTS[0])]+=.001
        with self.assertRaisesRegex(ValueError,'560 expert is not exact HOLD'):t.assess(episodes,model,streaming=True,boundary_only=True)

    def test_exact_hold_execution_gate_does_not_claim_a_hold_mode_choice(self):
        model,episodes=fixtures()
        with torch.no_grad():
            model.head.weight.zero_();model.head.bias.zero_();model.mode_head.weight.zero_()
            model.mode_head.bias.reshape(28,3).fill_(-1.);model.mode_head.bias.reshape(28,3)[:,t.policy.UPDATE]=1.
        result=t.assess(episodes,model,streaming=True,boundary_only=True)
        self.assertEqual(result['step560_regression_violation_count'],0)
        self.assertEqual(result['runs'][-1]['step560']['predicted_classes'],[t.policy.UPDATE]*3)
        self.assertEqual(result['runs'][-1]['step560']['projection_exact_HOLD'],[True]*3)

    def test_initial_or_batched_candidate_cannot_be_saved(self):
        model=policy()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as folder:
            path=Path(folder)/'model.pt'
            with self.assertRaisesRegex(ValueError,'Untrained'):t.save_candidate(path,model,{},0,1,{}, {})
            with self.assertRaisesRegex(ValueError,'CPU sequential'):t.save_candidate(path,model,{},1,9,{'inference':'batch'},{'inference':'sequential_one_frame_hidden_state'})
            self.assertFalse(path.exists())

    def test_last_state_is_separate_and_keeps_rng_and_optimizer(self):
        model=policy();optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad]);rng=random.Random(19)
        (model.head.bias.square().sum()+model.mode_head.bias.sum()).backward();optim.step()
        before=rng.getstate();cpu=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as folder:
            path=Path(folder)/'last_state.pt';best=Path(folder)/'model.pt';best.write_bytes(b'frozen')
            t.save_last_state(path,model,optim,rng,7,63,{'source':'fixed'},(0,0,0,4,.001));saved=torch.load(path,weights_only=True)
            self.assertEqual(saved['optimizer_steps'],63);self.assertEqual(saved['episode_shuffle_random_state'],before)
            self.assertTrue(torch.equal(saved['torch_cpu_rng_state'],cpu));self.assertEqual(saved['best_selection_key'],[0,0,0,4,.001])
            self.assertFalse(saved['exact_resume_demonstrated']);self.assertEqual(best.read_bytes(),b'frozen')


if __name__=='__main__':unittest.main()
