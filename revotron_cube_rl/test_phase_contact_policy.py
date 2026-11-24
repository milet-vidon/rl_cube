import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from phase_contact_policy import features,pair,phase_progress,ResidualPolicy,FEATURES,accepted_prefix,sha,restored

def row(value=0):
    return {'q':[value]*56,'joint_velocity':[0.]*56,'applied_target':[value]*56,'cube_position':[0.,0.,1.],'cube_quaternion_wxyz':[1.,0.,0.,0.],'finger_contact_points':[],'phase':'ready'}

class ContractTests(unittest.TestCase):
    def test_pre_action_pair(self):
        previous=row(.1);current=row(.12);current['q']=[99.]*56
        x,y=pair(previous,current,list(range(28)),.5)
        self.assertTrue(np.allclose(x[:56],.1));self.assertTrue(np.allclose(y,.02));self.assertEqual(x[-1],.5)
    def test_contact_and_shapes(self):
        r=row();r['finger_contact_points']=[{'side':'left','finger':'thumb','impulse_Ns':.001}]
        x=features(r,'ready',0);self.assertEqual(len(x),FEATURES);self.assertAlmostEqual(x[175],np.log(2),places=6)
        r['q']=[0.]*55
        with self.assertRaises(ValueError):features(r,'ready',0)
    def test_step_and_finite_guards(self):
        with self.assertRaises(ValueError):pair(row(),row(.041),list(range(28)),0)
        r=row();r['cube_position'][0]=float('nan')
        with self.assertRaises(ValueError):features(r,'ready',0)
    def test_progress_transitions(self):
        rows=[{'phase':p} for p in ['ready','ready','ready','lower','lower']]
        self.assertEqual(phase_progress(rows),[0.,.5,1.,0.,1.])
    def test_bounded_output(self):
        m=ResidualPolicy(np.zeros(FEATURES),np.ones(FEATURES),np.full(28,-.2),np.full(28,.2),32)
        with torch.no_grad():m.net[-1].bias.fill_(100.)
        x=torch.zeros(1,FEATURES)
        self.assertLessEqual(float(m(x).detach().abs().max()),.040001)
        self.assertLessEqual(float(m.targets(x,torch.full((1,28),.19)).detach().max()),.200001)
    def test_checkpoint_reload(self):
        m=ResidualPolicy(np.zeros(FEATURES),np.ones(FEATURES),np.full(28,-.2),np.full(28,.2),32)
        with torch.no_grad():m.net[-1].bias.fill_(.15)
        clone=restored({'metadata':{'width':32},'model':m.state_dict()})
        self.assertTrue(torch.equal(m(torch.ones(2,FEATURES)),clone(torch.ones(2,FEATURES))))
    def test_evidence_tampering_rejected_before_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'trajectory.json').write_text('[]')
            (root/'artifact_manifest.json').write_text(json.dumps({'sha256':{'trajectory.json':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'Evidence hash mismatch'):accepted_prefix(root)
    def test_failed_left_episode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data={'trajectory.json':[],'result.json':{'full_episode':True,'dynamic_grasp_passed':False},'config.json':{},'asset_contract.json':{},'center_plan.json':{},'center_result.json':{}}
            for name,value in data.items():(root/name).write_text(json.dumps(value))
            (root/'artifact_manifest.json').write_text(json.dumps({'sha256':{name:sha(root/name) for name in data}}))
            with self.assertRaisesRegex(ValueError,'Left prefix acceptance missing'):accepted_prefix(root)

if __name__=='__main__':unittest.main()
