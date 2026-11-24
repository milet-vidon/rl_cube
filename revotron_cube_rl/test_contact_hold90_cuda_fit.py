import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from contact_hold90_cuda_fit import Balanced, ROUND15, SCHEMA, _sha, _state, fit, load_batches
from contact_hold_grace_worker import FIT
from contact_hold_loop_data import checkpoint
from contact_hold_policy import restored
from phase_contact_policy import FEATURES, PHASES

PARENT=FIT/'best.pt'


def dataset(path, seed=10_100_000, parent_sha=ROUND15):
    path.mkdir(parents=True)
    rng=np.random.default_rng(317)
    ph=np.array([0]*20+[5]*4+[0]*8+[5]*8, dtype=np.int16); n=len(ph)
    val=np.arange(n)>=24;seeds=np.where(val,seed+1,seed).astype(np.int64)
    x=rng.normal(size=(n,FEATURES)).astype(np.float32)
    x[:,-len(PHASES)-1:-1]=np.eye(len(PHASES))[ph]
    pk=_state(PARENT)['model']
    mid=(pk['lower']+pk['upper']).mul(.5).numpy()
    y=np.repeat(mid[None,:],n,axis=0).astype(np.float32)
    np.savez(path/'batch.npz',x=x,y=y,mask=np.ones(y.shape,bool),phase=ph,seed=seeds,
             val=val,step=np.arange(n),lane=val.astype(np.int32))
    m=dict(schema=SCHEMA,role='collection',teacher_label_source='projected_90mm_feedback_teacher',cube_edge_m=.09,
           data_sha256=_sha(path/'batch.npz'),parent_sha=parent_sha,train_seeds=[seed],loss_val_seeds=[seed+1],
           physics_device='cuda:0',control_hz=30,physics_hz=240,action_dim=56,eval_seeds=[],cross_lane_contacts=0,
           seeds=[seed,seed+1],asset_sha256='a'*64,grasp_sha256='f'*64)
    (path/'report.json').write_text(json.dumps(m))
    return path/'batch.npz'


class TestFit(unittest.TestCase):
    def test_phase_balance_only_train_rows(self):
        with tempfile.TemporaryDirectory() as d:
            a=dataset(Path(d)/'set')
            arr,*_=load_batches([a],{ROUND15},[],[])
            x,y,m,p,v=arr
            x[:,0]=v.astype(np.float32)
            sampler=Balanced(x,y,m,p,v,'cpu')
            rng=torch.Generator().manual_seed(42)
            for step in range(3):
                xx,_,_=sampler.sample(32,step,rng)
                self.assertTrue((xx[:,0]==0).all())
                self.assertEqual(xx[:,-len(PHASES)-1:-1].sum(0).tolist(),[16,0,0,0,0,16,0,0,0,0,0,0])

    def test_reject_eval_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            a=dataset(Path(d)/'set')
            for past,valid,exc in (([10_100_001],[],[]),([],[10_100_000],[]),([],[],[10_100_001])):
                with self.assertRaisesRegex(ValueError,'seed split'):load_batches([a],{ROUND15},past,valid,exc)
            b=dataset(Path(d)/'reserved',20_777_555)
            with self.assertRaisesRegex(ValueError,'seed split'):load_batches([b],{ROUND15},[],[])
            rp=a.with_name('report.json');r=json.loads(rp.read_bytes());r['role']='validate';rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Ineligible'):load_batches([a],{ROUND15},[],[])

    def test_bitwise_resume_same_best_and_sampler_stream(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=dataset(root/'set')
            kw=dict(batch=16,updates=6,lr=2e-4,save_every=2,device='cpu',seed=128)
            base=fit(PARENT,[a],root/'single',**kw)
            paused=fit(PARENT,[a],root/'split',stop_after=3,**kw)
            self.assertFalse(paused['completed'])
            with self.assertRaisesRegex(ValueError,'Changed parent, data'):
                fit(PARENT,[a],root/'split',**(kw|{'lr':1e-4}))
            resumed=fit(PARENT,[a],root/'split',**kw)
            self.assertEqual(checkpoint(root/'split'),resumed['checkpoint_sha256'])
            self.assertEqual(base['best_metrics'],resumed['best_metrics']); self.assertEqual(base['selected_step'],resumed['selected_step'])
            for artifact in ('best.pt','last.pt'):
                l,r=[_state(root/run/artifact) for run in ('single','split')]
                for key in l['model']:self.assertTrue(torch.equal(l['model'][key],r['model'][key]),key)
                if artifact=='last.pt':
                    self.assertTrue(torch.equal(l['rng'],r['rng']))
                    for key,vs in l['optimizer']['state'].items():
                        for f,v in vs.items():self.assertTrue(torch.equal(v,r['optimizer']['state'][key][f]),f)
            child=dataset(root/'set2',10_200_000,resumed['checkpoint_sha256'])
            rr=fit(root/'split'/'best.pt',[a,child],root/'round2',**(kw|{'updates':1}))
            c=_state(root/'round2'/'best.pt');restored(c)
            self.assertEqual(c['metadata']['warm_start_sha256'],ROUND15)
            self.assertFalse(c['metadata']['optimizer_reset'])
            self.assertTrue(rr['checkpoint_reload_verified'])

if __name__=='__main__':unittest.main()
