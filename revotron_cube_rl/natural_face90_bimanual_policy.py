"""State PPO for the supported, opposed right-thumb/index grasp of the nZ layer."""
import math

import torch
from torch import nn

from natural_face90_recipe_policy import StateExpert, finite


SCHEMA = 'natural_face90_bimanual_state_ppo_v1'
ACTION_AXES = ('cube_x', 'cube_y', 'cube_z', 'thumb_flex', 'index_flex', 'thumb_opposition', 'face_orbit')
INPUT_FIELDS = (
    'index_error_core_x_cm', 'index_error_core_y_cm', 'index_error_core_z_cm',
    'thumb_error_core_x_cm', 'thumb_error_core_y_cm', 'thumb_error_core_z_cm',
    'index_minus_thumb_core_x_cm', 'index_minus_thumb_core_y_cm', 'index_minus_thumb_core_z_cm',
    'palm_opposition_cos', 'right_thumb_mcp_rad', 'right_index_mcp_rad', 'right_thumb_cmr_rad',
    'log_thumb_target_impulse', 'log_index_target_impulse', 'log_left_static_impulse',
    'log_left_target_impulse', 'log_other_right_target_impulse',
    'thumb_contact', 'index_contact', 'left_static_contact', 'overlap_fraction',
    'supported', 'bilateral_contact', 'structure_ok', 'wrong_contact',
    'measured_face_over_goal', 'sustained_drive_over_goal', 'goal_over_90', 'remaining_fraction',
)
ACTOR_DIM = CRITIC_DIM = len(INPUT_FIELDS)
ACTION_DIM = len(ACTION_AXES)
REWARD_FIELDS = ('approach', 'thumb_first', 'index_first', 'bilateral_first', 'drive', 'success',
                 'unsafe', 'time_effort')
IMPULSE_SCALE_NS = .01
APPROACH_SCALE_M = .035


def _positive_goal(degrees):
    degrees = float(degrees)
    if not math.isfinite(degrees) or degrees <= 0:
        raise ValueError('Expected a positive finite target angle')
    return degrees


def _errors(geometry):
    return (geometry['goal_index_local'] - geometry['index_local'],
            geometry['goal_thumb_local'] - geometry['thumb_local'])


def pack_observation(geometry, contacts, m, drive_deg, goal_deg, remaining):
    """Measured pad/goal geometry is in the moving Core frame. One row per lane."""
    goal = _positive_goal(goal_deg)
    index_error, thumb_error = _errors(geometry)
    span = geometry['index_local'] - geometry['thumb_local']
    impulses = torch.stack([contacts[k] for k in ('thumb_Ns', 'index_Ns', 'left_static_Ns',
                                                  'left_target_Ns', 'other_target_Ns')], dim=1)
    if bool((impulses < 0).any()):
        raise ValueError('Expected nonnegative measured solver impulses')
    face = m['face_deg']
    scalars = torch.stack((contacts['thumb'], contacts['index'], contacts['left_static'],
                           contacts['overlap'] / 8., contacts['supported'], contacts['bilateral'],
                           m['structure_ok'], m['wrong_contact'], face / goal, drive_deg / goal,
                           torch.full_like(face, goal / 90.), remaining), dim=1)
    obs = torch.cat((100. * index_error, 100. * thumb_error, 100. * span,
                     geometry['palm_opposition_cos'][:, None], geometry['hand_q'],
                     torch.log1p(impulses / IMPULSE_SCALE_NS), scalars), dim=1)
    if obs.shape != (len(face), ACTOR_DIM):
        raise ValueError(f'Unexpected bimanual observation shape {tuple(obs.shape)}')
    return finite(obs.float())


def _network(outputs):
    return nn.Sequential(nn.Linear(ACTOR_DIM, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(),
                         nn.Linear(64, outputs))


class BimanualExpert(nn.Module):
    """Seven Gaussian residual actions around the opposed-grasp guide; shared PPO optimizer."""
    def __init__(self):
        super().__init__()
        self.actor = _network(ACTION_DIM)
        self.critic = _network(1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(.5)))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        if obs.shape[-1] != ACTOR_DIM:
            raise ValueError('Bimanual actor observation dimension mismatch')
        return torch.distributions.Normal(finite(self.actor(obs)), self.log_std.exp())

    def value(self, critic):
        if critic.shape[-1] != CRITIC_DIM:
            raise ValueError('Bimanual critic observation dimension mismatch')
        return finite(self.critic(critic).squeeze(-1))

    sample = StateExpert.sample


def _proximity(geometry):
    """Both pad errors matter. The less well placed finger determines most of this score."""
    index, thumb = _errors(geometry)
    distances = torch.stack((index.norm(dim=-1), thumb.norm(dim=-1)), dim=-1)
    proximity = torch.exp(-distances / APPROACH_SCALE_M)
    return .75 * proximity.amin(dim=-1) + .25 * proximity.mean(dim=-1)


class BimanualReward:
    """Per-wave lane-local memory. Each approach/drive improvement and acquisition pays once.

    The worker supplies bilateral from simultaneous native substeps and credit_deg from
    sustained bilateral contact and measured nZ rotation relative to Core.
    """
    def __init__(self, n, device):
        self.best_proximity = torch.zeros(n, dtype=torch.float64, device=device)
        self.peak_drive_deg = torch.zeros_like(self.best_proximity)
        self.seen = torch.zeros((n, 3), dtype=torch.bool, device=device)
        self.finished = torch.zeros(n, dtype=torch.bool, device=device)
        self.totals = self.best_proximity.new_zeros((n, len(REWARD_FIELDS)))

    @torch.no_grad()
    def update(self, valid, unsafe, success, before_geometry, after_geometry, contacts,
               credit_deg, target_deg, latent):
        goal = _positive_goal(target_deg)
        if (valid.shape != self.finished.shape or latent.shape != (len(valid), ACTION_DIM)
                or credit_deg.shape != valid.shape):
            raise ValueError('Unexpected bimanual reward lane or action shape')
        alive = valid & ~self.finished
        safe = alive & ~unsafe
        supported = safe & contacts['supported'] & contacts['left_static']
        bilateral = supported & contacts['bilateral']
        zero = torch.zeros_like(self.best_proximity)

        previous = torch.maximum(self.best_proximity, _proximity(before_geometry))
        proximity = _proximity(after_geometry)
        approach = torch.where(supported, 2. * (proximity - previous).clamp_min(0), zero)
        self.best_proximity = torch.where(supported, torch.maximum(previous, proximity),
                                          self.best_proximity)

        touches = torch.stack((contacts['thumb'], contacts['index'], bilateral), dim=1)
        now = supported[:, None] & touches
        fresh = now & ~self.seen
        self.seen |= now

        # A new contact segment can restart at zero. It must exceed the prior best
        # to pay again. Reversing or drifting while disconnected cannot earn credit.
        current = torch.where(bilateral, credit_deg.double().clamp(0., goal), zero)
        peak = torch.maximum(self.peak_drive_deg, current)
        drive = (5. * (peak - self.peak_drive_deg) / goal +
                 2. * (torch.log1p(peak) - torch.log1p(self.peak_drive_deg)) / math.log1p(goal))
        self.peak_drive_deg = peak

        won = supported & success & (self.peak_drive_deg >= goal - 1e-7)
        lost = alive & unsafe
        self.finished |= won | lost
        cost = -.001 - .002 * latent.tanh().square().mean(dim=-1)
        parts = torch.stack((approach, .5 * fresh[:, 0], .5 * fresh[:, 1], 2. * fresh[:, 2],
                             drive, 20. * won, -20. * lost, torch.where(alive, cost, zero)), dim=1)
        self.totals += parts
        return finite(parts.sum(dim=1).float())
