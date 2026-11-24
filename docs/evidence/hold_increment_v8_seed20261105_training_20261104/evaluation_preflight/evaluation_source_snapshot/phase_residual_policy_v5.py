"""Privileged actual-state history residual over a frozen v4 phase policy.

Left 28 joints only; phase/progress are supplied by the original stage controller.
This is neither a vision actor nor an autonomous cube-solving policy. The GRU
receives no expert memory, target label, or teacher/student execution identity.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn

from dagger_collection_core_v1 import actor_observation, SUPPORTED_PHASES
from phase_basis_policy_v4 import restored as restore_baseline
from phase_contact_policy import features as v4_features, FEATURES as V4_FEATURES, FINGERS

SCHEMA = 'frozen_v4_actual_state_gru_bounded_left28_residual_v5'
FEATURE_SCHEMA = 'v4_observed198_cube_velocity6_impulse10_contact_presence10_centroid30_normal30_v5'
FEATURES = V4_FEATURES + 6 + 10 + 10 + 30 + 30


def features(observation):
    """Whitelist actual observation fields; hidden labels cannot affect output."""
    obs = actor_observation(observation)
    if obs['phase'] not in SUPPORTED_PHASES:
        raise ValueError('V5 only supports ready through tilt_hold')
    base = v4_features(obs, obs['phase'], obs['phase_progress'])
    cube = np.asarray(obs['cube_position'], dtype=np.float64)
    # Quaternion sign is observationally equivalent. The baseline uses phase
    # only, so canonicalizing this state feature does not change its output.
    quat = np.asarray(obs['cube_quaternion_wxyz'], dtype=np.float64)
    if abs(np.linalg.norm(quat) - 1.) > 1e-3:
        raise ValueError('Cube quaternion is not unit length')
    if quat[np.argmax(np.abs(quat))] < 0: base[171:175] *= -1
    impulses, present, centers, normals = [], [], [], []
    for side in ('left', 'right'):
        for finger in FINGERS:
            impulse = float(obs['finger_contact_impulses'][side][finger])
            if not np.isfinite(impulse) or impulse < 0:
                raise ValueError('Invalid measured finger impulse')
            impulses.append(np.log1p(impulse / .001))
            points = [p for p in obs['finger_contact_points'] if p['side'] == side and p['finger'] == finger]
            weight, center, normal = 0., np.zeros(3), np.zeros(3)
            for point in points:
                w = float(point['impulse_Ns'])
                position = np.asarray(point['position_world_m'], dtype=np.float64)
                direction = np.asarray(point['normal_world'], dtype=np.float64)
                if (w < 0 or not np.isfinite(w) or position.shape != (3,) or direction.shape != (3,)
                        or not np.isfinite(position).all() or not np.isfinite(direction).all()):
                    raise ValueError('Invalid observed contact geometry')
                weight += w
                center += w * (position - cube)
                normal += w * direction
            present.append(float(weight > 0))
            centers.extend(center / weight if weight > 0 else center)
            normals.extend(normal / weight if weight > 0 else normal)
    extra = (obs['cube_linear_velocity'] + obs['cube_angular_velocity'] + impulses + present + centers + normals)
    result = np.concatenate((base, np.asarray(extra, dtype=np.float32)))
    if result.shape != (FEATURES,) or not np.isfinite(result).all():
        raise ValueError('Invalid V5 actual-state feature vector')
    return result


class PhaseResidualPolicy(nn.Module):
    def __init__(self, baseline_saved, mean, std, residual_bound, hidden_size=64):
        super().__init__()
        if hidden_size < 1: raise ValueError('Positive GRU width required')
        self.hidden_size = int(hidden_size)
        self.baseline = restore_baseline(baseline_saved)
        for parameter in self.baseline.parameters(): parameter.requires_grad_(False)
        self.baseline.eval()
        bound = torch.as_tensor(residual_bound, dtype=torch.float32).flatten()
        if bound.numel() == 1: bound = bound.repeat(28)
        if bound.shape != (28,) or not torch.isfinite(bound).all() or (bound <= 0).any() or (bound > .08).any():
            raise ValueError('Explicit positive left28 residual bounds must be at most .08 rad')
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (FEATURES,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Finite training-only normalization required')
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clamp_min(1e-3).clone())
        self.register_buffer('residual_bound', bound.clone())
        self.encoder = nn.Sequential(nn.Linear(FEATURES, 96), nn.LayerNorm(96), nn.SiLU())
        self.gru = nn.GRU(96, self.hidden_size, batch_first=True)
        self.head = nn.Linear(self.hidden_size, 28)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def baseline_targets(self, x, previous_left):
        shape = x.shape[:-1]
        with torch.no_grad():
            base = self.baseline.targets(x[..., :V4_FEATURES].reshape(-1, V4_FEATURES),
                                         previous_left.reshape(-1, 28))
        return base.to(dtype=torch.float32).reshape(*shape, 28)

    def forward(self, x, previous_left, hidden=None):
        if x.ndim != 3 or x.shape[-1] != FEATURES or previous_left.shape != (*x.shape[:2], 28):
            raise ValueError('Expected batch/time/features and actual previous left commands')
        if not torch.isfinite(x).all() or not torch.isfinite(previous_left).all():
            raise ValueError('Nonfinite inference state')
        encoded = self.encoder(((x - self.mean) / self.std).clamp(-12., 12.))
        history, next_hidden = self.gru(encoded, hidden)
        residual = torch.tanh(self.head(history)) * self.residual_bound
        base = self.baseline_targets(x, previous_left)
        proposed = torch.clamp(base + residual, self.baseline.lower.float(), self.baseline.upper.float())
        return proposed, next_hidden, residual


def project_left_targets(proposed, previous, parameters, left_indices, projector):
    """Original per-joint projection after the runtime's float32 action roundtrip.

    This is supervision/evaluation math, not an alternative runtime step. Live
    execution must continue through the original environment project_target.
    """
    shape = proposed.shape
    proposed, previous = proposed.reshape(-1, 28).float(), previous.reshape(-1, 28).float()
    def tensor(key):
        return torch.as_tensor(parameters[key], dtype=torch.float32, device=proposed.device)
    scales = tensor('scales')[:, left_indices]
    requested = previous + scales * ((proposed - previous) / scales)
    projected, _ = projector(previous, requested, tensor('limits')[:, left_indices],
        tensor('step_limit')[:, left_indices], tensor('command_velocity_limits')[:, left_indices], parameters['control_dt'])
    return projected.reshape(shape)


class EpisodeActor:
    """Explicit reset and contiguous step IDs prevent hidden-state carryover."""
    def __init__(self, model, left_indices):
        self.model = model.eval()
        self.left = list(left_indices)
        if len(self.left) != 28 or len(set(self.left)) != 28:
            raise ValueError('Exactly 28 distinct left joint indices required')
        self.episode_id = None
        self.hidden = None
        self.next_step = 0

    def reset(self, episode_id):
        self.episode_id, self.hidden, self.next_step = str(episode_id), None, 0

    def act(self, observation, *, episode_id, step):
        if self.episode_id is None or str(episode_id) != self.episode_id or step != self.next_step:
            raise RuntimeError('Explicit episode reset and contiguous actual-state history required')
        device = next(self.model.parameters()).device
        x = torch.from_numpy(features(observation)).to(device).reshape(1, 1, FEATURES)
        previous = torch.as_tensor(np.asarray(observation['applied_target'], dtype=np.float32)[self.left], device=device).reshape(1, 1, 28)
        with torch.no_grad():
            proposed, self.hidden, _ = self.model(x, previous, self.hidden)
        self.hidden = self.hidden.detach()
        self.next_step += 1
        return proposed[0, 0].cpu().tolist()
