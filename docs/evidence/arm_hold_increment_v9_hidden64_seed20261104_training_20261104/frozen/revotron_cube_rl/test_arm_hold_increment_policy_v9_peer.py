"""Independent CPU numeric/contract review; no physical or dataset claim."""
import ast
from copy import deepcopy
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import arm_hold_increment_policy_v9 as p
import hold_increment_policy_v8 as old
from test_arm_hold_increment_policy_v9 import force_modes
from test_phase_residual_v5 import baseline
from test_zero_gated_residual_policy_v7 import frame,joint_safety


def params():
    names=list(p.EXPECTED_LEFT_NAMES)+['right_fixture_'+str(i)for i in range(28)]
    return {'joint_names':names,'dtype':'torch.float32','scales':[[.025]*56],
        'max_target_delta':[[.04]*56],'step_limit':[[.025]*56],
        'command_velocity_limits':[[.75]*56],'control_dt':1/30,
        'limits':[[[-3.,3.]]*56]}


def model():
    b=baseline();b['model']['lower']=torch.full((28,),-3.,dtype=torch.float32);b['model']['upper']=torch.full((28,),3.,dtype=torch.float32)
    return p.HoldIncrementPolicy(b,np.zeros(p.FEATURES),np.ones(p.FEATURES),p.candidate_bounds(p.EXPECTED_LEFT_NAMES),hidden_size=8)


def bits_equal(a,b):return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


class IndependentV9Contracts(unittest.TestCase):
    def test_only_seven_named_arm_bounds_change(self):
        names=p.EXPECTED_LEFT_NAMES;before=old.candidate_bounds(names);after=p.candidate_bounds(names)
        self.assertEqual([n for n,a,b in zip(names,before,after)if a!=b],list(names[:7]))
        self.assertEqual(after[:7],[.025]*7);self.assertEqual(after[7:],before[7:])
        m=model();self.assertEqual(int(m.update_anchor_is_previous.sum()),18)
        self.assertEqual(set(n for n,yes in zip(names,m.update_anchor_is_previous)if yes),p.ARM_JOINTS|p.RECOVERY_JOINTS)

    def test_classification_and_loss_ast_are_identical_to_v8(self):
        def selected(module):
            tree=ast.parse(Path(module.__file__).read_text());out={}
            for node in tree.body:
                if isinstance(node,ast.FunctionDef)and node.name in ('classify_branch_labels','supervision_losses'):out[node.name]=ast.dump(node,include_attributes=False)
                if isinstance(node,ast.ClassDef):
                    for n in node.body:
                        if isinstance(n,ast.FunctionDef)and n.name in ('forward','forward_with_diagnostics','load_state_dict'):out[n.name]=ast.dump(n,include_attributes=False)
            return out
        self.assertEqual(selected(p),selected(old));self.assertEqual(p.CLASSIFICATION_CONTRACT,old.CLASSIFICATION_CONTRACT)

    def test_verify_cap_is_read_only_and_named_permutation_equivalent(self):
        q=params();prior=deepcopy(q);report=p.verify_projection_arm_bounds(q,list(range(28)))
        self.assertEqual(q,prior);self.assertEqual(report['arm_joint_names'],['Joint%d_L'%i for i in range(1,8)])
        reverse=p.verify_projection_arm_bounds(q,list(range(27,-1,-1)))
        self.assertEqual(reverse['update_bounds_rad'],report['update_bounds_rad'][::-1]);self.assertFalse(report['original_projection_changed'])

    def test_one_ulp_higher_or_lower_original_arm_cap_is_rejected(self):
        for direction in (-float('inf'),float('inf')):
            q=params();q['control_dt']=1.;q['command_velocity_limits']=[[.025]*56]
            q['scales']=[[.05]*56];q['step_limit']=[[.04]*56]
            value=torch.nextafter(torch.tensor(.025,dtype=torch.float32),torch.tensor(direction,dtype=torch.float32)).item()
            q['command_velocity_limits'][0][3]=value
            with self.subTest(cap=value),self.assertRaises(ValueError):p.verify_projection_arm_bounds(q,list(range(28)))

    def test_wrong_step_definition_nonfinite_and_wrong_dtype_rejected(self):
        mutations=(('step_limit',.039),('scales',float('nan')),('command_velocity_limits',float('inf')),('max_target_delta',0.))
        for key,value in mutations:
            q=params();q[key][0][4]=value
            with self.subTest(key=key),self.assertRaises(ValueError):p.verify_projection_arm_bounds(q,list(range(28)))
        for dt in (0.,-1.,float('nan'),float('inf')):
            q=params();q['control_dt']=dt
            with self.subTest(dt=dt),self.assertRaises(ValueError):p.verify_projection_arm_bounds(q,list(range(28)))
        q=params();q['dtype']='torch.float64'
        with self.assertRaises(ValueError):p.verify_projection_arm_bounds(q,list(range(28)))

    def test_original_joint_names_and_index_types_are_strict(self):
        for bad_names in (params()['joint_names'][:-1],params()['joint_names'][:-1]+['Joint1_L'],params()['joint_names'][:-1]+[''],params()['joint_names'][:-1]+[17]):
            q=params();q['joint_names']=bad_names
            with self.subTest(names=bad_names[-1]),self.assertRaises(ValueError):p.verify_projection_arm_bounds(q,list(range(28)))
        for indices in ([False]+list(range(1,28)),[-56]+list(range(1,28)),[56]+list(range(1,28)),[np.int64(0)]+list(range(1,28)),[0]+list(range(27))):
            with self.subTest(indices=indices[:2]),self.assertRaises(ValueError):p.verify_projection_arm_bounds(params(),indices)

    def test_constructor_rejects_wrong_named_bounds_scalar_and_ulp_tampering(self):
        good=torch.tensor(p.candidate_bounds(p.EXPECTED_LEFT_NAMES),dtype=torch.float32)
        variants=[.025,good[:-1],good.clone(),good.clone()]
        variants[2][3]=torch.nextafter(good[3],torch.tensor(float('inf')))
        variants[3][8]=.0061
        for value in variants:
            with self.subTest(value=str(value)[:80]),self.assertRaises(ValueError):
                p.HoldIncrementPolicy(baseline(),np.zeros(p.FEATURES),np.ones(p.FEATURES),value,hidden_size=8)

    def test_real_j4_minus_cap_and_adjacent_float32_previous_have_finite_exact_witness(self):
        # Numbers are the recorded seed04 step1802 input/label, not simulated data.
        m=model();force_modes(m,p.UPDATE);m.head.bias.data.fill_(-100.)
        q=params();previous=torch.zeros(1,1,28,dtype=torch.float32);previous[...,3]=-2.207127332687378
        base=previous.clone();base[...,3]=1.
        with patch.object(m,'baseline_targets',return_value=base):
            proposed,_,_,diag=m.forward_with_diagnostics(frame(),previous)
        applied=p.project_left_targets(proposed,previous,q,list(range(28)),joint_safety.project_target)
        self.assertEqual(float(applied[0,0,3]),-2.2321274280548096)
        self.assertTrue(bits_equal(diag['update_anchor'][...,3],previous[...,3]))
        # The subtraction can exceed scalar .025 through rounding; final projection
        # remains the unchanged source of execution truth, never an epsilon bypass.
        self.assertGreater(float((applied-previous).abs()[0,0,3]),float(torch.tensor(.025)))
        for sign in (-1.,1.):
            m.head.bias.data.fill_(sign*100.)
            for direction in (-float('inf'),float('inf')):
                prev=previous.clone();prev[...,3]=torch.nextafter(previous[...,3],torch.tensor(direction))
                with patch.object(m,'baseline_targets',return_value=base):out,_,_,_=m.forward_with_diagnostics(frame(),prev)
                actual=p.project_left_targets(out,prev,q,list(range(28)),joint_safety.project_target)
                desired=prev.clone();desired[...,3]=prev[...,3]+sign*torch.tensor(.025)
                expected=p.project_left_targets(desired,prev,q,list(range(28)),joint_safety.project_target)
                self.assertTrue(bits_equal(actual[...,3],expected[...,3]))

    def test_original_soft_limit_clamp_still_dominates_update(self):
        m=model();force_modes(m,p.UPDATE);m.head.bias.data.fill_(100.)
        q=params();previous=torch.full((1,1,28),2.99)
        with patch.object(m,'baseline_targets',return_value=previous):out,_,_,_=m.forward_with_diagnostics(frame(),previous)
        projected=p.project_left_targets(out,previous,q,list(range(28)),joint_safety.project_target)
        self.assertTrue((projected<=3.).all());self.assertEqual(float(projected[0,0,3]),3.)

    def test_anchor_and_bound_checkpoint_tampering_is_rejected(self):
        m=model();s=deepcopy(m.state_dict())
        for key in ('update_anchor_is_previous','residual_bound'):
            changed=deepcopy(s)
            if changed[key].dtype==torch.bool:changed[key][0]=False
            else:changed[key][0]=torch.nextafter(changed[key][0],torch.tensor(float('inf')))
            with self.subTest(key=key),self.assertRaises(ValueError):m.load_state_dict(changed)


if __name__=='__main__':unittest.main()
