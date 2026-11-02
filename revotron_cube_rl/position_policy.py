"""Low-dimensional visual localization and actor for the cube-pickup task."""
from collections import OrderedDict
import torch
from torch import nn
from rsl_compat import ActorCritic

POSITION_DIM = 6


def rgb_to_position_features(rgb: torch.Tensor, resolution: int = 128) -> torch.Tensor:
    """Extract cube position/extent from normalized RGB without feeding RGB to the actor.

    The simulated cube is red. The output is normalized centroid x/y, mask area,
    bounding-box width/height, and a visibility bit. If the mask is absent the
    geometric values are zero and visibility is zero.
    """
    if rgb.ndim == 2:
        rgb = rgb.reshape(-1, resolution, resolution, 3)
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (resolution, resolution, 3):
        raise ValueError(f'Expected RGB [N,{resolution},{resolution},3], got {tuple(rgb.shape)}')
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 1.25 * g) & (r > 1.25 * b) & (r > .25)
    n = mask.shape[0]
    device = rgb.device
    yy = torch.arange(resolution, device=device, dtype=rgb.dtype).view(1, resolution, 1)
    xx = torch.arange(resolution, device=device, dtype=rgb.dtype).view(1, 1, resolution)
    count = mask.sum((1, 2)).to(rgb.dtype)
    visible = (count >= 4).to(rgb.dtype)
    safe = count.clamp_min(1.)
    cx = (mask.to(rgb.dtype) * xx).sum((1, 2)) / safe
    cy = (mask.to(rgb.dtype) * yy).sum((1, 2)) / safe
    inf = torch.full_like(xx, float(resolution))
    x_min = torch.where(mask, xx, inf).amin((1, 2))
    x_max = torch.where(mask, xx, torch.zeros_like(xx)).amax((1, 2))
    y_min = torch.where(mask, yy, inf).amin((1, 2))
    y_max = torch.where(mask, yy, torch.zeros_like(yy)).amax((1, 2))
    features = torch.stack((
        (cx / (resolution - 1.) * 2. - 1.) * visible,
        (cy / (resolution - 1.) * 2. - 1.) * visible,
        # Keep all localization features near the same numeric scale as the
        # normalized centroid/box dimensions (typical cube area is ~0.2).
        (count / float(resolution * resolution) * 100.) * visible,
        ((x_max - x_min + 1.) / resolution) * visible,
        ((y_max - y_min + 1.) / resolution) * visible,
        visible,
    ), dim=-1)
    if features.shape != (n, POSITION_DIM) or not torch.isfinite(features).all():
        raise ValueError('Non-finite or malformed cube-position features')
    return features


class PositionActor(nn.Module):
    def __init__(self, proprio_dim: int = 224, position_dim: int = POSITION_DIM):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.position_dim = position_dim
        self.register_buffer('proprio_scale', torch.ones(proprio_dim))
        self.head = nn.Sequential(
            nn.Linear(proprio_dim + position_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 56))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        if x.shape[1] != self.proprio_dim + self.position_dim:
            raise ValueError(f'Expected {self.proprio_dim + self.position_dim} actor inputs, got {x.shape[1]}')
        return self.head(torch.cat((x[:, :self.proprio_dim] * self.proprio_scale, x[:, self.proprio_dim:]), -1))

    @torch.no_grad()
    def set_target_error_scaling(self, action_scales):
        scales = torch.as_tensor(action_scales, dtype=self.proprio_scale.dtype, device=self.proprio_scale.device).reshape(-1)
        if self.proprio_dim != 224 or scales.numel() != 56 or not (scales > 0).all():
            raise ValueError('Position actor requires 224 proprioception and 56 positive action scales')
        new_gain = scales.reciprocal(); old_gain = self.proprio_scale[168:224].clone()
        self.head[0].weight[:, 168:224].mul_(old_gain / new_gain)
        self.proprio_scale[168:224].copy_(new_gain)

    @torch.no_grad()
    def disable_action_history(self):
        if self.proprio_dim != 224:
            raise ValueError('Action-history ablation requires 224 proprioception')
        self.proprio_scale[112:168].zero_()


class PositionActorCritic(ActorCritic):
    def __init__(self, obs, position_dim: int = POSITION_DIM):
        proprio_dim = obs['policy'].shape[-1] - position_dim
        if proprio_dim != 224:
            raise ValueError(f'Expected 224 proprioception, got {proprio_dim}')
        dummy = obs.clone(); dummy['policy'] = obs['policy'][:, :proprio_dim]
        super().__init__(dummy, {'policy': ['policy'], 'critic': ['critic']}, 56,
                         actor_hidden_dims=[64], critic_hidden_dims=[256, 128],
                         init_noise_std=.25, noise_std_type='log')
        self.actor = PositionActor(proprio_dim, position_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'actor.proprio_scale' not in state_dict:
            compatible = OrderedDict(state_dict)
            compatible['actor.proprio_scale'] = torch.ones_like(self.actor.proprio_scale)
            state_dict = compatible
        return super().load_state_dict(state_dict, strict=strict)
