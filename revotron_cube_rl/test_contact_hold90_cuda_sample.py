"""CPU tests of lane separation and the physical window scorer; no GPU or simulator."""
import unittest
import numpy as np
import torch
from contact_hold90_cuda_sample import partitions,PhaseBank,HoldMeter,whole_score

class Sampler(unittest.TestCase):
    def test_evaluation_can_never_become_training(self):
        tr,val=partitions(list(range(10_000_000,10_000_024)),'collection')
        self.assertEqual(set(),set(tr)&set(val));self.assertEqual(len(tr),18)
        for seed,role in ((21611,'collection'),(20_000_000,'collection'),(10_000_000,'validate')):
            with self.assertRaises(ValueError):partitions([seed,seed+1],role)
        self.assertEqual(partitions([20_000_001,20_000_002],'validate'),([],[]))

    def test_retains_small_phase_and_seed_split(self):
        n=80;phase=np.r_[np.zeros(76,int),[3,3,3,3]];val=np.arange(n)%2==0;row=np.arange(n)
        bank=PhaseBank(3,7);bank.add(dict(phase=phase,val=val,tag=row));a=bank.finish()
        for p in (0,3):
            for v in (True,False):
                take=(a['phase']==p)&(a['val']==v);self.assertEqual(int(take.sum()),3 if p==0 else 2)
                self.assertTrue((phase[a['tag'][take]]==p).all());self.assertTrue((val[a['tag'][take]]==v).all())

    def test_diameter_of_all_pairs_and_confirmed_turn(self):
        # A cube alternating between opposite points 6 mm from its first position slips 12 mm.
        N=63;m=HoldMeter(3,'cpu');a=np.zeros((N,3,13));a[:,:,0]=1;a[:,:,8]=.025;a[:,:,10]=.02;a[:,:,12]=1e-5
        for step in range(N):
            a[step,1,1]=[0.,.006,-.006][step%3]
            angle=np.deg2rad(18 if step>=12 else 0);a[step,:,6]=np.sin(angle/2);a[step,:,7]=np.cos(angle/2)
            if step==N-1:a[step,2,0]=0.;a[step,2,9]=.02
            m.update(step,torch.ones(3,dtype=torch.bool),torch.tensor(a[step,:,0],dtype=torch.bool),torch.tensor(a[step,:,1:4]))
        result=[whole_score(a[:,i],N) for i in range(3)]
        self.assertEqual(m.ever.tolist(),[True,False,True]);self.assertEqual(m.end.tolist(),[True,False,False]);self.assertEqual(m.best.tolist(),[30,2,30])
        self.assertTrue(result[0]['ever_hold_and_turn']);self.assertFalse(result[1]['ever_hold_and_turn']);self.assertTrue(result[2]['ever_hold_and_turn']);self.assertFalse(result[2]['ending_hold_and_turn'])

if __name__=='__main__':unittest.main()
