"""Project policy contract over RSL-RL 3.x or the 5.x split-model PPO API.

The CNN/position actor, privileged critic and state_dict names stay project-owned.
PPO optimization is delegated to the installed RSL-RL implementation.
"""
from importlib.metadata import version
import torch
from torch import nn

RSL5 = int(version('rsl-rl-lib').split('.')[0]) >= 5

if not RSL5:
    from rsl_rl.modules import ActorCritic
    from rsl_rl.algorithms import PPO
else:
    from rsl_rl.models import MLPModel
    from rsl_rl.modules.distribution import GaussianDistribution
    from rsl_rl.algorithms import PPO as NativePPO
    from rsl_rl.storage import RolloutStorage

    def _mlp(input_dim, hidden_dims, output_dim):
        layers = []
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ELU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    class ActorCritic(nn.Module):
        """Common container used by the project's three actor architectures."""
        is_recurrent = False

        def __init__(self, obs, obs_groups, num_actions, actor_hidden_dims,
                     critic_hidden_dims, init_noise_std, noise_std_type):
            super().__init__()
            if noise_std_type != 'log':
                raise ValueError('This project requires log-standard-deviation exploration')
            self._obs_groups = obs_groups
            self._example_obs = obs
            self._num_actions = num_actions
            actor_dim = sum(obs[k].shape[-1] for k in obs_groups['policy'])
            critic_dim = sum(obs[k].shape[-1] for k in obs_groups['critic'])
            self.actor = _mlp(actor_dim, actor_hidden_dims, num_actions)
            self.critic = _mlp(critic_dim, critic_hidden_dims, 1)
            self.log_std = nn.Parameter(torch.full((num_actions,), float(torch.log(torch.tensor(init_noise_std)))))

        def act_inference(self, obs):
            return self.actor(torch.cat([obs[k] for k in self._obs_groups['policy']], dim=-1))

        def evaluate(self, obs):
            return self.critic(torch.cat([obs[k] for k in self._obs_groups['critic']], dim=-1))

        def reset(self, dones=None):
            pass

    class PPO(NativePPO):
        """Expose the project's constructor/storage lifecycle over native RSL PPO."""
        def __init__(self, policy, **kwargs):
            groups = {'actor': policy._obs_groups['policy'], 'critic': policy._obs_groups['critic']}
            actor = MLPModel(policy._example_obs, groups, 'actor', policy._num_actions,
                             hidden_dims=[1], obs_normalization=False)
            actor.mlp = policy.actor
            actor.distribution = GaussianDistribution(policy._num_actions, std_type='log')
            actor.distribution.log_std_param = policy.log_std
            critic = MLPModel(policy._example_obs, groups, 'critic', 1,
                              hidden_dims=[1], obs_normalization=False)
            critic.mlp = policy.critic
            super().__init__(actor=actor, critic=critic, storage=None, **kwargs)

        def init_storage(self, training_type, num_envs, num_steps, obs, actions_shape):
            self.storage = RolloutStorage(training_type, num_envs, num_steps, obs,
                                          actions_shape, device=self.device)
