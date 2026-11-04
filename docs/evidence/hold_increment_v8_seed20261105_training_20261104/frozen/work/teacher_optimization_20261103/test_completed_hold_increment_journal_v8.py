"""CPU tests for artifact review only; no resumed training or physics claim."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train_hold_increment_v8 as trainer
import audit_completed_hold_increment_journal_v8 as audit
from test_train_hold_increment_v8_peer import fixtures

torch.set_num_threads(1)


def last_fixture():
    _,model,_=fixtures()
    config={'epochs':1,'learning_rate':3e-4,'weight_decay':1e-5,'device':'cpu'}
    metadata={'training_config':config,'fixture':'synthetic_not_physics'}
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum()for p in model.parameters()if p.requires_grad).backward();optimizer.step()
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as temporary:
        path=Path(temporary)/'last.pt'
        with patch.object(torch.cuda,'is_available',return_value=False):
            trainer.save_last_state(path,model,optimizer,random.Random(3),1,8,metadata,(0,0,2,.001))
        value=torch.load(path,map_location='cpu',weights_only=True)
    return value,model,config,metadata


class CompletedArtifactContracts(unittest.TestCase):
    def test_epoch_shuffle_is_reconstructed_and_tampered_order_rejected(self):
        ordered=['history'+str(i)for i in range(8)]
        expected=random.Random(321);observed=random.Random(321)
        for _ in range(5):
            indices=list(range(8));expected.shuffle(indices)
            rows=[{'episode_id':ordered[i]}for i in indices]
            audit.validate_recorded_shuffle(observed,ordered,rows)
        self.assertEqual(expected.getstate(),observed.getstate())
        indices=list(range(8));expected.shuffle(indices)
        rows=[{'episode_id':ordered[i]}for i in indices];rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaisesRegex(ValueError,'shuffle RNG'):
            audit.validate_recorded_shuffle(observed,ordered,rows)

    def test_optimizer_rng_inventory_validates_without_resuming_or_changing_global_rng(self):
        value,model,config,metadata=last_fixture()
        py=random.getstate();cpu=torch.get_rng_state().clone();npstate=np.random.get_state()
        result=audit.validate_last_state(value,model,config,metadata,(0,0,2,.001))
        self.assertTrue(result['adamw_moments_and_steps_validated'])
        self.assertFalse(result['exact_resume_demonstrated'])
        self.assertFalse(result['cuda_rng_restore_or_exact_resume_executed'])
        self.assertEqual(random.getstate(),py);self.assertTrue(torch.equal(torch.get_rng_state(),cpu))
        after=np.random.get_state();self.assertEqual(after[0],npstate[0]);np.testing.assert_array_equal(after[1],npstate[1]);self.assertEqual(after[2:],npstate[2:])

    def test_incomplete_steps_changed_moments_static_anchor_and_false_resume_claim_rejected(self):
        original,model,config,metadata=last_fixture()
        for kind in ('epoch','steps','moment_step','moment_shape','negative_moment','anchor','best_key','resume_claim','metadata'):
            value=deepcopy(original);first=next(iter(value['optimizer']['state']))
            if kind=='epoch':value['epoch']=0
            elif kind=='steps':value['optimizer_steps']=7
            elif kind=='moment_step':value['optimizer']['state'][first]['step'].fill_(7)
            elif kind=='moment_shape':value['optimizer']['state'][first]['exp_avg']=torch.tensor([0.])
            elif kind=='negative_moment':value['optimizer']['state'][first]['exp_avg_sq'].fill_(-1.)
            elif kind=='anchor':value['model']['update_anchor_is_previous'][0]=True
            elif kind=='best_key':value['best_selection_key'][2]=99
            elif kind=='resume_claim':value['exact_resume_demonstrated']=True
            else:value['metadata']['fixture']='changed'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                audit.validate_last_state(value,model,config,metadata,(0,0,2,.001))

    def test_metrics_recount_confusion_and_development_boundaries(self):
        _,model,episodes=fixtures();metrics=trainer.assess(episodes,model,streaming=True)
        counts=np.asarray(metrics['confusion_matrix_true_by_predicted']).sum(1).tolist()
        frames={r['episode_id']:r['frames_processed']for r in metrics['runs']}
        audit.validate_metrics(metrics,counts,frames,'sequential_one_frame_hidden_state')
        for kind in ('confusion','499','545','frames','claim'):
            modified=deepcopy(metrics)
            if kind=='confusion':modified['classification_error_count']+=1
            elif kind=='499':modified['step499_regression_violation_count']+=1
            elif kind=='545':modified['step545_regression_violation_count']-=1
            elif kind=='frames':modified['runs'][0]['frames_processed']-=1
            else:modified['teacher_labels_used_as_actor_input']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                audit.validate_metrics(modified,counts,frames,'sequential_one_frame_hidden_state')


if __name__=='__main__':unittest.main()
