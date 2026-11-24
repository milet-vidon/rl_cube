"""CPU-only tests for online controls, reward sampling, optimiser resume and seed lineage."""
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from contact_hold90_controls import (ENGINE,DEFAULTS,_save,initialize,control_snapshot,
                                     validate_params,freeze_round,queue_params)
from contact_hold90_objective import Weighted, weights_for_rows, imitation_loss
from contact_hold90_cuda_fit import ROUND15, _sha, _state, load_batches
from contact_hold90_weighted_fit import fit
from contact_hold90_reward_sample import TurnWindow
from test_contact_hold90_cuda_fit import PARENT, dataset

class Boundary(unittest.TestCase):
    def test_config_revision_resume_current_and_total(self):
        with tempfile.TemporaryDirectory() as t:
            h=Path(t);_save(h/'settings.json',{'engine':ENGINE})
            initialize(h,dict(rounds=4))
            r1=freeze_round(h,1)
            q=queue_params(h,dict(batch=128,rounds=2,hold_score=2),expected_revision=r1['revision'])
            self.assertEqual(q['pending']['earliest_round'],2)
            self.assertEqual(freeze_round(h,1),r1)
            self.assertEqual(control_snapshot(h)['effective']['hold_score'],0)
            with self.assertRaises(ValueError):queue_params(h,dict(updates=128),expected_revision=r1['revision'])
            with self.assertRaises(ValueError):queue_params(h,dict(envs=4))
            r2=freeze_round(h,2)
            self.assertEqual(r2['revision'],q['pending']['revision'])
            self.assertEqual(r2['params']['hold_score'],2)
            self.assertIsNone(freeze_round(h,3))
            self.assertTrue(control_snapshot(h)['completed'])
            with self.assertRaises(ValueError):queue_params(h,dict(rounds=20))
    def test_fail_closed_inputs(self):
        for p in ({'lr':float('nan')},{'updates':False},{'envs':5},{'hold_score':None},{'unknown':1}):
            with self.assertRaises(ValueError):validate_params(p)
    def test_legacy_cannot_be_modified(self):
        with tempfile.TemporaryDirectory() as t:
            h=Path(t);_save(h/'settings.json',{'rounds':5})
            self.assertFalse(control_snapshot(h)['supported'])
            with self.assertRaises(ValueError):queue_params(h,{'batch':16})
            self.assertEqual([p.name for p in h.iterdir()],['settings.json'])

class Campaign(unittest.TestCase):
    def test_real_round_dispatch_frozen_controls_and_extension(self):
        """Exercise real orchestration, replacing only scarce simulation and GPU/optimizer calls."""
        import contact_hold90_tunable as engine
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as t:
            h=Path(t);params=validate_params({'rounds':1,'batch':32,'envs':4,'val_envs':4,'updates':1})
            c=engine.configuration(params,engine.FIT)
            _save(h/'settings.json',c);initialize(h,params)
            s=dict(completed_round=0,current=str(engine.FIT),current_sha=engine.CHECKPOINT,wave_cap=4)
            fit_calls=[]
            def wave(home,state,config,round_id,parent,role,fd):
                d=home/f'round_{round_id:03d}'/role;d.mkdir(exist_ok=True)
                if role=='collection' and round_id==1:
                    queue_params(home,{'rounds':2,'batch':64,'hold_score':3})
                elif role=='validate':
                    _save(d/'report.json',{'role':role,'per_lane':[dict(seed=20000000,ever_hold=True,ending_hold=False,
                                ever_whole_turn=False,max_height_m=.05)]})
                return [d]
            def fakefit(cmd,**kw):
                out=Path(cmd[cmd.index('--outdir')+1]);out.mkdir()
                self.assertTrue(cmd[3].endswith('contact_hold90_weighted_fit.py'))
                receipt=json.loads(Path(cmd[cmd.index('--control-receipt')+1]).read_bytes())
                batch=int(cmd[cmd.index('--batch')+1]);self.assertEqual(receipt['params']['batch'],batch)
                fit_calls.append((receipt['round'],batch,receipt['params']['hold_score']))
                (out/'best.pt').write_bytes(b'fake_test_actor');_save(out/'result.json',{'checkpoint_sha256':_sha(out/'best.pt')})
            with patch.object(engine,'roundsamples',side_effect=wave),patch.object(engine,'gpu_slot',return_value=nullcontext([])),patch.object(engine,'live_progress',return_value=nullcontext()),patch.object(engine.subprocess,'run',side_effect=fakefit):
                engine.campaign(h,c,s,-1)
            self.assertEqual(fit_calls,[(1,32,0),(2,64,3)])
            self.assertTrue(control_snapshot(h)['completed'])
            self.assertEqual(s['status'],'completed');self.assertEqual(s['completed_round'],2)

class Signals(unittest.TestCase):
    def test_support_and_persistent_net_rotation_on_three_frames(self):
        # First lane rotates and stays, second briefly spikes, third loses hold.
        w=TurnWindow(3,'cpu');phase=np.pi/9
        for i in range(30):
            q=torch.zeros(3,4,dtype=torch.float64);q[:,-1]=1
            if i>=8:
                q[[0,2],0]=np.sin(phase);q[[0,2],3]=np.cos(phase)
            if i==13:q[1]=q[0]
            if i%2==0:q=-q  # antipodal quaternions are the same orientation
            w.update(i,torch.tensor([True]*3),torch.tensor([True,True,False]),q)
        self.assertEqual(w.ever.tolist(),[True,False,False])

    def test_new_collector_bootstraps_and_closes_pinned_isaac_entry(self):
        import types,sys
        import contact_hold90_reward_sample as sampler
        from unittest.mock import Mock
        isaac_app=types.ModuleType('isaaclab.app');launcher=Mock();isaac_app.AppLauncher=launcher
        compat=types.ModuleType('isaac_compat');close=Mock();compat.close_application=close
        with patch.dict(sys.modules,{'isaaclab':types.ModuleType('isaaclab'),'isaaclab.app':isaac_app,'isaac_compat':compat}),patch.object(sampler,'sample') as collect,patch.object(sys,'argv',['sampler','--plan','p','--output','o','--role','collection']):
            sampler.main()
        launcher.assert_called_once_with(headless=True,device='cuda:0')
        close.assert_called_once_with(launcher.return_value.app)
        self.assertEqual(collect.call_args.args[0].role,'collection')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.home=Path(self.tmp.name)
        self.a=dataset(self.home/'set')
        with np.load(self.a,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        seed=int(arrays['seed'][0]);arrays['seed'][:8]=seed+2
        np.savez(self.a,**arrays)
        rp=self.a.with_name('report.json');r=json.loads(rp.read_bytes());r.update(data_sha256=_sha(self.a))
        r['seeds'].append(seed+2);r['train_seeds'].append(seed+2)
        r['per_lane']=[{'seed':seed,'ever_hold':False,'ending_hold':False,'reward_supported_turn':False},
            {'seed':seed+2,'ever_hold':True,'ending_hold':True,'reward_supported_turn':True},
            {'seed':seed+1,'ever_hold':True}]  # held-out outcome deliberately incomplete: may never be consulted
        rp.write_text(json.dumps(r))
    def test_weights_alignment_phase_and_real_gradient(self):
        arr,details,*_=load_batches([self.a],{ROUND15},[],[]);x,y,m,ph,v=arr
        p=validate_params(dict(hold_score=3,ending_score=2,turn_score=5,lift_weight=5))
        w,hits=weights_for_rows(details,p)
        self.assertTrue((w[:8]==11).all() and (w[8:]==1).all())
        self.assertEqual(hits,dict(hold_score=1,ending_score=1,turn_score=1))
        g=Weighted(x,y,m,ph,v,'cpu',w,p)
        counts=g.counts(128,1);self.assertEqual(sum(counts),128);self.assertGreater(counts[1],counts[0]*4)
        z=torch.tensor([[1.],[3.]],requires_grad=True)
        loss=imitation_loss(z,torch.tensor([[True],[True]]),torch.tensor([1.,3.]))
        loss.backward();self.assertEqual(loss.item(),7.);self.assertEqual(z.grad.flatten().tolist(),[.5,4.5])
    def test_cpu_optimizer_and_frozen_objective_on_resume(self):
        kw=dict(batch=16,updates=3,lr=2e-4,save_every=1,device='cpu',seed=128)
        p=dict(hold_score=9,turn_weight=3)
        unweighted=fit(PARENT,[self.a],self.home/'base',**kw)
        fit(PARENT,[self.a],self.home/'tuned',controls=p,stop_after=1,**kw)
        with self.assertRaisesRegex(ValueError,'Changed parent, data'):fit(PARENT,[self.a],self.home/'tuned',**kw)
        tuned=fit(PARENT,[self.a],self.home/'tuned',controls=p,**kw)
        repeat=fit(PARENT,[self.a],self.home/'repeat',controls=p,**kw)
        self.assertEqual(tuned['best_metrics'],repeat['best_metrics'])
        self.assertEqual(tuned['training_episode_outcome_counts']['hold_score'],1)
        a,b,c=[_state(self.home/k/'last.pt')['model'] for k in ('base','tuned','repeat')]
        self.assertTrue(any(not torch.equal(a[n],b[n]) for n in a))
        self.assertTrue(all(torch.equal(c[n],b[n]) for n in b))

if __name__=='__main__':unittest.main()
