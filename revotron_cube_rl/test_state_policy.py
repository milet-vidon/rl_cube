"""Safety properties of the learned absolute-target decoder."""
import unittest
import torch
from state_policy import StateActor


class StateTargetTests(unittest.TestCase):
    def test_extreme_latents_stay_within_demonstrated_joint_ranges(self):
        center=torch.linspace(-1.,1.,56)
        scale=torch.linspace(0.,.8,56)
        actor=StateActor(torch.zeros(261),torch.ones(261),center,scale,True)
        latent=torch.randn(1000,56)*1e6
        targets=actor.targets(latent)
        self.assertTrue(torch.isfinite(targets).all())
        self.assertTrue((targets>=center-scale-1e-7).all())
        self.assertTrue((targets<=center+scale+1e-7).all())
        self.assertTrue(torch.equal(targets[:,0],center[0].expand(1000)))

    def test_bounded_decoder_retains_gradients_for_training(self):
        actor=StateActor(torch.zeros(261),torch.ones(261),torch.zeros(56),torch.ones(56),True)
        latent=torch.zeros(1,56,requires_grad=True)
        actor.targets(latent).sum().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertTrue((latent.grad>0).all())

    def test_legacy_decoder_is_unchanged(self):
        actor=StateActor(torch.zeros(261),torch.ones(261),torch.ones(56),torch.full((56,),.2))
        self.assertTrue(torch.allclose(actor.targets(torch.full((1,56),3.)),torch.full((1,56),1.6)))


if __name__=='__main__':unittest.main()
