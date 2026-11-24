"""Compact task-space PPO policy for the right index pad on the moving nZ layer."""
import math

import torch
from torch import nn

from natural_face90_recipe_policy import StateExpert, finite
from natural_face90_turn_guide import _point


INPUT_FIELDS = (
    'marker_minus_pad_core_x_cm', 'marker_minus_pad_core_y_cm', 'marker_minus_pad_core_z_cm',
    'pad_step_core_x_mm', 'pad_step_core_y_mm', 'pad_step_core_z_mm',
    'marker_radial_core_x', 'marker_radial_core_y', 'face_over_target', 'credited_drive_over_target',
    'supported', 'right_loaded', 'index_contact', 'other_or_wrong_contact',
    'aligned', 'drive', 'recovering', 'reached', 'remaining_fraction',
)
ACTION_AXES = ('radial_out', 'positive_tangent', 'inward')
ACTOR_DIM = CRITIC_DIM = len(INPUT_FIELDS)
ACTION_DIM = len(ACTION_AXES)


def material_marker(radial, face_deg):
    """30 mm ring on the nZ face in Core coordinates; positive rotation is about nZ."""
    angle = face_deg.double()
    return _point(radial, angle, torch.zeros_like(angle))


class PadMotion:
    """Measured displacement in moving Core coordinates over one control interval.

    The observer, reward and logger can read the same post-physics step.
    A newly active lane, or one with a missing measurement, starts at zero.
    """
    def __init__(self, n, device):
        self.position = torch.zeros(n, 3, device=device, dtype=torch.float64)
        self.delta = torch.zeros_like(self.position)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.step = None

    def measure(self, step, local_tip, active):
        if local_tip.shape != self.position.shape or active.shape != self.active.shape or active.dtype != torch.bool:
            raise ValueError('Expected one measured 3D pad location and active flag per lane')
        if self.step is not None and step < self.step:
            raise ValueError('Pad motion observations must follow physical control steps')
        pos = local_tip.detach()
        if self.step != step:
            if self.step is not None and step == self.step + 1:
                self.delta.copy_(torch.where(self.active[:, None], pos - self.position, 0.))
            else:
                self.delta.zero_()
            self.active.copy_(active)
            self.position.copy_(pos)
            self.step = step
        else:
            # A lane can hand off just after a logging read of the same physical step.
            new_lane = active & ~self.active
            self.position.copy_(torch.where(new_lane[:, None], pos, self.position))
            self.active |= active
        return torch.where(active[:, None], self.delta, 0.)


def pack_observation(local_tip, pad_delta, radial, measurements, signed_credit_deg, target_deg,
                     remaining, aligned, driving, recovering):
    """19 measured/controller features in INPUT_FIELDS order, shared by actor and critic.

    The first six use centimetres of marker error and millimetres of pad displacement.
    Direction and angle features use the material marker of the current measured layer.
    The stage flags are from the last control decision completed before this action.
    """
    if target_deg <= 0:
        raise ValueError('Expected a positive target face angle')
    face = measurements['face_deg']
    marker = material_marker(radial, face)
    support = (measurements['support'].bool() & measurements['natural_ready'].bool()
               & measurements['static_contact'].bool() & measurements['structure_ok'].bool()
               & measurements['allowed_layer_state'].bool() & ~measurements['wrong_contact'].bool())
    other = measurements['other_digit_contact'].bool() | measurements['wrong_contact'].bool()
    scalar = torch.stack((
        face / target_deg, signed_credit_deg / target_deg,
        support, measurements['right_total_impulse'] != 0., measurements['index_contact'].bool(), other,
        aligned, driving, recovering, measurements['contact_goal_reached'], remaining,
    ), dim=1)
    obs = torch.cat(((marker - local_tip) * 100., pad_delta * 1000., marker[:, :2] / .030, scalar), dim=1)
    if obs.shape != (len(face), ACTOR_DIM):
        raise ValueError(f'Invalid Cartesian observation shape {tuple(obs.shape)}')
    return finite(obs.float())


def _network(output_dim):
    return nn.Sequential(
        nn.Linear(ACTOR_DIM, 64), nn.ELU(),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, output_dim),
    )


class CartesianExpert(nn.Module):
    """Independent Gaussian translations in the rotating radial/tangent/inward frame.

    Existing PPO stores latent actions and their log probabilities. The guide converts
    latents to bounded metric deltas before right-arm IK. A new deterministic actor
    starts at exactly the nominal Cartesian guide target.
    """
    def __init__(self):
        super().__init__()
        self.actor = _network(ACTION_DIM)
        self.critic = _network(1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(.5)))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        if obs.shape[-1] != ACTOR_DIM:
            raise ValueError('Cartesian actor observation dimension mismatch')
        return torch.distributions.Normal(finite(self.actor(obs)), self.log_std.exp())

    def value(self, critic):
        if critic.shape[-1] != CRITIC_DIM:
            raise ValueError('Cartesian critic observation dimension mismatch')
        return finite(self.critic(critic).squeeze(-1))

    sample = StateExpert.sample
