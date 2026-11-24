"""State BC for newly audited pickup-to-wrist demonstrations (not V11 resume)."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

SCHEMA = 'hold_contact_state_bc_v12_q56_previous_target56_xyz3_wxyz4_time1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(state, previous_target, step):
    value = np.asarray(state['q'] + list(previous_target) + state['cube_position'] +
                       state['cube_quaternion_wxyz'] + [step / 3000.], dtype=np.float32)
    if value.shape != (120,) or not np.isfinite(value).all():
        raise ValueError('Invalid PRE-action state')
    return value


def load_demonstration(run):
    """Fail closed, then recompute the original physical acceptance on saved rows."""
    run = Path(run).resolve()
    manifest = json.loads((run / 'artifact_manifest.json').read_text())
    required = ('result.json', 'trajectory.json', 'config.json', 'asset_contract.json',
                'initial_state_v12.json', 'hold_candidate_v12.json', 'hold_gain_events_v12.jsonl',
                'runtime_robot_asset.json', 'cube_asset.json', 'experiment.json')
    for name in required:
        if manifest['sha256'].get(name) != sha(run / name):
            raise ValueError('Evidence hash mismatch: ' + name)
    # Verify every saved source/config/evidence artifact, not only the result bit.
    for name, digest in manifest['sha256'].items():
        if Path(name).name != name or sha(run / name) != digest:
            raise ValueError('Archived artifact changed: ' + name)
    result = json.loads((run / 'result.json').read_text())
    if not (result['full_episode'] and not result['execution_interrupted'] and
            result['dynamic_grasp_passed'] and result['wrist']['accepted'] and
            result['measured']['accepted'] and result['failure'] is None and
            result['cube_pose_writes_during_control'] == 0 and result['checkpoint'] is None and
            result['robot_table_impulse_sum_Ns'] <= 1e-6 and
            result['self_contact_impulse_sum_Ns'] <= 1e-6):
        raise ValueError('Source is not an accepted, safe, complete teacher demonstration')
    from palm_handover_audit import assess_handover
    from wrist_audit import assess_wrist
    cfg = json.loads((run / 'config.json').read_text())
    names = cfg['joint_names']
    canonical = json.loads((Path(__file__).parent / 'config.json').read_text())['joint_names']
    if names != canonical or len(names) != 56 or cfg['control_hz'] != 30:
        raise ValueError('Action order or control frequency changed')
    left = [i for i, name in enumerate(names) if name.endswith('_L') or name.startswith('left_')]
    if len(left) != 28:
        raise ValueError('Expected 28 left joints')
    rows = json.loads((run / 'trajectory.json').read_text())
    if len(rows) != result['steps'] or [r['step'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete/nonconsecutive episode')
    measured = assess_handover(rows, names, regrasp=False, full_episode=True)
    wrist = assess_wrist([r if r['layer_measurement']['assembled_for_grasp'] else
                          r | {'phase': 'cube_not_assembled'} for r in rows], names)
    if not measured['accepted'] or not wrist['accepted']:
        raise ValueError('Independent replay of original physical gates failed')
    initial = json.loads((run / 'initial_state_v12.json').read_text())
    limits = np.asarray(initial['joint_limits'], dtype=np.float64)
    if limits.shape != (56, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Invalid recorded joint bounds')
    previous = initial
    targets = np.asarray(initial['applied_target'], dtype=np.float64)
    x, y = [], []
    cap = np.array([min(.025, cfg['arm_max_step_rad'], 3.1416 * cfg['arm_velocity_limit_scale'] / 30)
                    if n.startswith('Joint') else .06 for n in names])
    for row in rows:
        applied = np.asarray(row['applied_target'], dtype=np.float64)
        if applied.shape != (56,) or not np.isfinite(applied).all():
            raise ValueError('Invalid target')
        if np.any(applied < limits[:, 0] - 1e-6) or np.any(applied > limits[:, 1] + 1e-6):
            raise ValueError('Target exceeds original safety intersection')
        if np.any(np.abs(applied - targets) > cap + 1e-6):
            raise ValueError('Target violates original command rate limit')
        if row['drop'] or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit']):
            raise ValueError('Unsafe recorded transition')
        x.append(features(previous, targets, row['step']))
        y.append(applied[left])
        previous, targets = row, applied
    info = dict(run=str(run), seed=result['seed'], samples=len(rows),
                manifest_sha256=sha(run / 'artifact_manifest.json'),
                original_wrist_gate=wrist, original_measured_gate=measured,
                candidate=json.loads((run / 'hold_candidate_v12.json').read_text()))
    return np.array(x), np.asarray(y, dtype=np.float32), names, left, limits[left].astype(np.float32), info


class HoldBC(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(120, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 28))
        self.register_buffer('mean', torch.zeros(120))
        self.register_buffer('std', torch.ones(120))
        self.register_buffer('center', torch.zeros(28))
        self.register_buffer('scale', torch.ones(28))
        self.register_buffer('limits', torch.zeros(28, 2))

    def forward(self, x):
        target = self.net((x - self.mean) / self.std) * self.scale + self.center
        return torch.maximum(self.limits[:, 0], torch.minimum(self.limits[:, 1], target))


def load_model(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Checkpoint hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['metadata']['schema'] != SCHEMA:
        raise ValueError('Checkpoint schema mismatch')
    model = HoldBC(saved['metadata']['hidden_size'])
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    return model.eval(), saved['metadata']
