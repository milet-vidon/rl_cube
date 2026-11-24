import unittest,numpy as np,torch
from phase_expert_policy_v3 import PhaseExpertPolicy,FEATURES,restored

class ExpertTests(unittest.TestCase):
    def model(self):return PhaseExpertPolicy(np.zeros(FEATURES),np.ones(FEATURES),np.full(28,-2.),np.full(28,2.),32)
    def test_state_cannot_change_inferred_goal(self):
        m=self.model();x=torch.zeros(1,FEATURES);x[:,185]=1;z=x.clone();z[:,:185]=torch.randn(1,185)*10
        self.assertTrue(torch.equal(m(x),m(z)))
    def test_separate_phase_heads(self):
        m=self.model()
        with torch.no_grad():m.center[0].fill_(.1);m.center[1].fill_(.2)
        x=torch.zeros(2,FEATURES);x[0,185]=1;x[1,186]=1
        self.assertTrue(torch.allclose(m(x)[0],torch.full((28,),.1)));self.assertTrue(torch.allclose(m(x)[1],torch.full((28,),.2)))
    def test_step_cap_and_checkpoint_reload(self):
        m=self.model();x=torch.zeros(1,FEATURES);x[:,185]=1;p=torch.ones(1,28)
        self.assertLessEqual(float((m.targets(x,p)-p).detach().abs().max()),.040001)
        clone=restored({'metadata':{'width':32},'model':m.state_dict()});self.assertTrue(torch.equal(m(x),clone(x)))

if __name__=='__main__':unittest.main()
