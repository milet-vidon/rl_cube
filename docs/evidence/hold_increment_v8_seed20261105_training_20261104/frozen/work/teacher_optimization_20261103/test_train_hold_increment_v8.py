"""Local artifact and finite-definition checks; no real-data optimizer runs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,random,tempfile,unittest
import numpy as np
import torch
import train_hold_increment_v8 as t
from test_hold_increment_policy_v8 import policy


class TrainingArtifacts(unittest.TestCase):
    def test_authorization_pins_candidate_sources_and_objective(self):
        config={'epochs':3,'regression_weight_rad2':1e6,'cross_entropy_weight':1.}
        with patch.object(t,'source_hashes',return_value={'trainer':'frozen'}):
            expected=t.authorization_template('qualified',config)
            text=json.dumps(expected)
            import hashlib
            digest=lambda x:hashlib.sha256(x.encode()).hexdigest()
            t.validate_authorization(text,digest(text),'qualified',config)
            changed=dict(config,epochs=4)
            with self.assertRaisesRegex(ValueError,'exact training definition'):
                t.validate_authorization(text,digest(text),'qualified',changed)
            expected['approved_for_development_bc_training']=False
            text=json.dumps(expected)
            with self.assertRaisesRegex(ValueError,'exact training definition'):
                t.validate_authorization(text,digest(text),'qualified',config)

    def test_candidate_cannot_save_initial_or_batched_metrics(self):
        m=policy()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as folder:
            path=Path(folder)/'model.pt'
            with self.assertRaisesRegex(ValueError,'Untrained'):
                t.save_candidate(path,m,{},0,1,{}, {})
            with self.assertRaisesRegex(ValueError,'CPU sequential'):
                t.save_candidate(path,m,{},1,8,{'inference':'complete_causal_history_batch'},
                    {'inference':'sequential_one_frame_hidden_state'})
            self.assertFalse(path.exists())

    def test_last_state_keeps_optimizer_rng_and_best_separate(self):
        m=policy();optim=torch.optim.AdamW([p for p in m.parameters()if p.requires_grad]);rng=random.Random(19)
        loss=m.head.bias.square().sum()+m.mode_head.bias.sum();loss.backward();optim.step()
        before=rng.getstate();cpu=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as folder:
            path=Path(folder)/'last_state.pt';best=Path(folder)/'model.pt';best.write_bytes(b'frozen-best')
            t.save_last_state(path,m,optim,rng,7,56,{'source':'fixed'},(0,0,4,.001))
            saved=torch.load(path,weights_only=True)
            self.assertEqual(saved['epoch'],7);self.assertEqual(saved['optimizer_steps'],56)
            self.assertEqual(saved['episode_shuffle_random_state'],before)
            self.assertTrue(torch.equal(saved['torch_cpu_rng_state'],cpu))
            self.assertEqual(set(saved['optimizer']['state']),set(optim.state_dict()['state']))
            self.assertEqual(saved['best_selection_key'],[0,0,4,.001])
            self.assertFalse(saved['exact_resume_demonstrated']);self.assertFalse(saved['resume_cli_implemented'])
            self.assertEqual(best.read_bytes(),b'frozen-best');self.assertFalse(path.with_name(path.name+'.pending').exists())

    def test_selection_orders_each_development_regression_before_class_error(self):
        def row(a,b,c,d):return {'step499_regression_violation_count':a,'step545_regression_violation_count':b,
            'classification_error_count':c,'hard_decoded_projected_fit':{'rmse_rad':d}}
        self.assertLess(t.selection_key(row(0,6,100,1)),t.selection_key(row(1,0,0,0)))
        self.assertLess(t.selection_key(row(0,0,100,1)),t.selection_key(row(0,1,0,0)))
        self.assertLess(t.selection_key(row(0,0,1,1)),t.selection_key(row(0,0,2,0)))


if __name__=='__main__':unittest.main()
