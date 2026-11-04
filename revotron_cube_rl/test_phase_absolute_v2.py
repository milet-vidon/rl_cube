import unittest
import numpy as np
import torch
from phase_absolute_policy_v2 import AbsolutePolicy,FEATURES,pair,restored
from test_phase_contact_policy import row

class AbsoluteTests(unittest.TestCase):
    def model(self):return AbsolutePolicy(np.zeros(FEATURES),np.ones(FEATURES),np.full(28,-2.),np.full(28,2.),32)
    def test_absolute_label_is_current_target(self):
        x,y=pair(row(.1),row(.12),list(range(28)),.5)
        self.assertTrue(np.allclose(y,.12));self.assertTrue(np.allclose(x[:56],.1))
    def test_previous_target_cannot_be_copied(self):
        m=self.model()
        with torch.no_grad():m.feedback[-1].weight.normal_()
        x=torch.zeros(1,FEATURES);z=x.clone();z[:,112:168]=10.
        self.assertTrue(torch.equal(m(x),m(z)))
    def test_no_accumulation_under_fixed_goal(self):
        m=self.model();x=torch.zeros(1,FEATURES);previous=torch.ones(1,28)
        for _ in range(100):previous=m.targets(x,previous)
        self.assertTrue(torch.allclose(previous,m(x)))
    def test_step_cap_and_reload(self):
        m=self.model();x=torch.zeros(1,FEATURES);prev=torch.ones(1,28)
        self.assertLessEqual(float((m.targets(x,prev)-prev).detach().abs().max()),.040001)
        clone=restored({'metadata':{'width':32},'model':m.state_dict()});self.assertTrue(torch.equal(clone(x),m(x)))

if __name__=='__main__':unittest.main()
