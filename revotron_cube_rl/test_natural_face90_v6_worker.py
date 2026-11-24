import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import natural_face90_recipe_worker as guided
import natural_face90_v6_long as runner
import natural_face90_v6_worker as worker


class NativeProgress(unittest.TestCase):
    def test_slips_interrupt_contiguity_and_oscillations_do_not_earn_repeated_credit(self):
        track = worker.IndexProgress(2, 'cpu')
        for angle, contact in [(0,True),(1,True),(12,False),(12,True),(11,True),(12,True)]:
            track.update(torch.tensor([float(angle),float(angle)]),torch.tensor([contact,False]))
        torch.testing.assert_close(track.peak,torch.tensor([1.,0.],dtype=torch.float64))
        torch.testing.assert_close(track.driven,torch.tensor([1.,0.],dtype=torch.float64))

    def test_worker_assigns_distal_support_proximity_and_logs_motion_after_each_action(self):
        state = worker.PostPhase.__new__(worker.PostPhase)
        n=3
        yes=torch.ones(n,dtype=torch.bool)
        core=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]]*n,dtype=torch.double)
        state.e=SimpleNamespace(cfg={'success':{'contact_impulse_min':.00001}},
                                views={'Core':SimpleNamespace(get_transforms=lambda:core)})
        state.controller=SimpleNamespace(radial=torch.tensor([[1.,0.]]*n,dtype=torch.double))
        state.req={'seeds':[1,2,3]}
        state.index_tracker=SimpleNamespace(index_impulse=torch.tensor([.1,0.,.1]),
            other_digit_impulse=torch.tensor([0.,.1,0.]),
            pad_position_world=lambda:torch.tensor([[.03,0.,-.045],[.03,0.,-.045],[.037,0.,-.087]],dtype=torch.double))
        state.index_progress=worker.IndexProgress(n,'cpu')
        state.turn_log=io.StringIO();state.retract=~yes;state.learning=yes
        state.guide=SimpleNamespace(drive_contact=yes,sweep_deg=torch.zeros(n),
            recovering=~yes,index_lead_m=torch.zeros(n),goal_local=torch.zeros(n,3))
        state.obs=dict(left_static_impulse=torch.ones(n),target_impulse=torch.ones(n),
            natural_ready=yes,support=yes,structure_ok=yes,allowed_layer_state=yes,
            wrong_contact=~yes,face_deg=torch.zeros(n,dtype=torch.double))
        with patch.object(guided.PostPhase,'observe',return_value={'unsafe':~yes}):
            state.observe(yes,yes,yes,123)
            state.obs['face_deg']+=1
            state.observe(yes,yes,yes,124)
        rows=[json.loads(l) for l in state.turn_log.getvalue().splitlines()]
        self.assertEqual([r['qualified_index_frame'] for r in rows[-3:]],[1.,0.,0.])
        self.assertEqual([r['signed_index_driven_deg'] for r in rows[-3:]],[1.,0.,0.])
        self.assertEqual([r['step'] for r in rows[-3:]],[124]*3)
        self.assertIs(worker.run.__globals__['PostPhase'],worker.PostPhase)
        self.assertIs(guided.run.__globals__['PostPhase'],guided.PostPhase)

    def test_carry_weights_and_only_resume_optimizer_with_matching_action_contract(self):
        old=dict(schema=runner.guided.GUIDED_SCHEMA,software={'base':'123'},action_mode='bounded_residual_on_index_pad_ik')
        new=dict(schema=runner.SCHEMA,software={'new':'345'},action_mode=worker.ACTION_MODE)
        ck=dict(contract=old,iteration=50,policy={},optimizer={'moment':1})
        transfer=dict(schema=old['schema'],iteration=50,sha='a',software=old['software'])
        req=dict(role='train',checkpoint_sha='a',transfer=transfer)
        model,opt=Mock(),Mock()
        worker.restore_turn_expert(model,opt,ck,new,req)
        model.load_state_dict.assert_called_once_with({},strict=True);opt.load_state_dict.assert_not_called()
        with self.assertRaises(ValueError):
            worker.restore_turn_expert(model,opt,dict(ck,contract=dict(old,learned_action_dim=28)),new,req)
        worker.restore_turn_expert(model,opt,dict(ck,contract=new),new,{'role':'validate'})
        opt.load_state_dict.assert_called_once_with(ck['optimizer'])
