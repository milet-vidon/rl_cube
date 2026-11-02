"""Absolute joint-target actor using explicit visual cube localization."""
from collections import OrderedDict
import torch
from torch import nn
from rsl_compat import ActorCritic

from position_policy import POSITION_DIM


class AbsolutePositionActor(nn.Module):
    """Maps proprioception + 6 visual features to normalized absolute joint targets."""
    def __init__(self, proprio_dim=224, position_dim=POSITION_DIM, joint_center=None, joint_half_range=None):
        super().__init__()
        if joint_center is None or joint_half_range is None:
            raise ValueError('Absolute actor requires joint center and half range')
        center = torch.as_tensor(joint_center, dtype=torch.float32).reshape(-1)
        half = torch.as_tensor(joint_half_range, dtype=torch.float32).reshape(-1)
        if center.numel() != 56 or half.numel() != 56 or not torch.isfinite(torch.cat((center, half))).all() or not (half > 0).all():
            raise ValueError('Invalid 56-joint absolute target contract')
        self.proprio_dim = proprio_dim
        self.position_dim = position_dim
        self.register_buffer('proprio_scale', torch.ones(proprio_dim))
        self.register_buffer('joint_center', center)
        self.register_buffer('joint_half_range', half)
        self.head = nn.Sequential(
            nn.Linear(proprio_dim + position_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 56))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        expected = self.proprio_dim + self.position_dim
        if x.shape[1] != expected:
            raise ValueError(f'Expected {expected} actor inputs, got {x.shape[1]}')
        return self.head(torch.cat((x[:, :self.proprio_dim] * self.proprio_scale, x[:, self.proprio_dim:]), -1))

    @torch.no_grad()
    def decode_targets(self, normalized):
        return self.joint_center.view(1, -1) + normalized * self.joint_half_range.view(1, -1)

    @torch.no_grad()
    def set_target_error_scaling(self, action_scales):
        scales = torch.as_tensor(action_scales, dtype=self.proprio_scale.dtype, device=self.proprio_scale.device).reshape(-1)
        if scales.numel() != 56 or not (scales > 0).all():
            raise ValueError('Expected 56 positive action scales')
        new_gain = scales.reciprocal(); old_gain = self.proprio_scale[168:224].clone()
        self.head[0].weight[:, 168:224].mul_(old_gain / new_gain)
        self.proprio_scale[168:224].copy_(new_gain)

    @torch.no_grad()
    def disable_action_history(self):
        self.proprio_scale[112:168].zero_()


class AbsolutePositionActorCritic(ActorCritic):
    def __init__(self, obs, joint_center, joint_half_range, position_dim=POSITION_DIM):
        proprio_dim = obs['policy'].shape[-1] - position_dim
        if proprio_dim != 224:
            raise ValueError(f'Expected 224 proprioception, got {proprio_dim}')
        dummy = obs.clone(); dummy['policy'] = obs['policy'][:, :proprio_dim]
        super().__init__(dummy, {'policy': ['policy'], 'critic': ['critic']}, 56,
                         actor_hidden_dims=[64], critic_hidden_dims=[256, 128],
                         init_noise_std=.25, noise_std_type='log')
        self.actor = AbsolutePositionActor(proprio_dim, position_dim, joint_center, joint_half_range)

    def load_state_dict(self, state_dict, strict=True):
        compatible = OrderedDict(state_dict)
        if 'actor.proprio_scale' not in compatible:
            compatible['actor.proprio_scale'] = torch.ones_like(self.actor.proprio_scale)
        if 'actor.joint_center' not in compatible:
            compatible['actor.joint_center'] = self.actor.joint_center.detach().clone()
        if 'actor.joint_half_range' not in compatible:
            compatible['actor.joint_half_range'] = self.actor.joint_half_range.detach().clone()
        return super().load_state_dict(compatible, strict=strict)
