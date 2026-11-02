"""Offline ABI/learning checks for the Isaac Lab 3 and RSL-RL 5 migration."""
import unittest
import torch
from tensordict import TensorDict
import isaac_compat as compat
from policy import VisionActorCritic
from position_policy import PositionActorCritic
from absolute_policy import AbsolutePositionActorCritic
from rsl_compat import PPO

class MigrationContractTests(unittest.TestCase):
    def test_close_preserves_failure_exit_code(self):
        original = compat.LAB3
        class App:
            def close(self, **kwargs):
                self.kwargs = kwargs
        app = App()
        try:
            compat.LAB3 = True
            compat.close_application(app)
            self.assertEqual(app.kwargs['exit_code'], 0)
            try:
                raise RuntimeError('simulation failure')
            except RuntimeError:
                compat.close_application(app)
            self.assertEqual(app.kwargs['exit_code'], 1)
            compat.LAB3 = False
            compat.close_application(app)
            self.assertNotIn('exit_code', app.kwargs)
        finally:
            compat.LAB3 = original

    def test_native_quaternion_preserves_project_convention(self):
        original=compat.LAB3
        try:
            compat.LAB3=True
            # 90 degrees around Z: xyzw rotation must carry x onto y.
            q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
            native=compat.quaternion_to_native(q)
            v=torch.tensor([[1.,0.,0.]])
            t=2*torch.cross(native[:,:3],v,dim=-1)
            rotated=v+native[:,3:]*t+torch.cross(native[:,:3],t,dim=-1)
            torch.testing.assert_close(rotated,torch.tensor([[0.,1.,0.]]),atol=1e-6,rtol=0)
            torch.testing.assert_close(compat.quaternion_from_native(native),q)
            compat.LAB3=False
            torch.testing.assert_close(compat.quaternion_to_native(q),q)
        finally:compat.LAB3=original

    def test_ppo_updates_all_actor_variants_and_preserves_isolation(self):
        torch.manual_seed(7)
        for variant in ('rgb','position','absolute'):
            with self.subTest(variant=variant):
                actor_dim=224+128*128*3 if variant=='rgb' else 230
                obs=TensorDict({'policy':torch.rand(4,actor_dim), 'critic':torch.randn(4,247)},batch_size=[4])
                if variant=='rgb':policy=VisionActorCritic(obs)
                elif variant=='position':policy=PositionActorCritic(obs)
                else:policy=AbsolutePositionActorCritic(obs,torch.zeros(56),torch.ones(56))
                alg=PPO(policy,num_learning_epochs=1,num_mini_batches=2,device='cpu')
                optimized=[p for g in alg.optimizer.param_groups for p in g['params']]
                self.assertEqual(len(optimized),len({id(p) for p in optimized}))
                self.assertEqual({id(p) for p in optimized},{id(p) for p in policy.parameters()})
                alg.init_storage('rl',4,4,obs,[56])
                actor_before=policy.actor.head[-1].weight.detach().clone()
                critic_before=next(policy.critic.parameters()).detach().clone()
                for _ in range(4):
                    with torch.no_grad():
                        actions=alg.act(obs)
                        self.assertEqual(tuple(actions.shape),(4,56))
                        alg.process_env_step(obs,torch.randn(4),torch.zeros(4,dtype=torch.bool),{})
                with torch.no_grad():alg.compute_returns(obs)
                losses=alg.update()
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
                self.assertFalse(torch.equal(actor_before,policy.actor.head[-1].weight))
                self.assertFalse(torch.equal(critic_before,next(policy.critic.parameters())))
                modified=obs.clone();modified['critic']+=20
                torch.testing.assert_close(policy.act_inference(obs),policy.act_inference(modified),atol=0,rtol=0)
                self.assertFalse(torch.equal(policy.evaluate(obs),policy.evaluate(modified)))
                saved={k:v.clone() for k,v in policy.state_dict().items()}
                expected=policy.act_inference(obs).detach().clone()
                with torch.no_grad():policy.actor.head[-1].weight.add_(1.)
                policy.load_state_dict(saved)
                torch.testing.assert_close(policy.act_inference(obs),expected)

if __name__=='__main__':unittest.main()
