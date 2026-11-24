"""Unqualified host candidate: learned BASE / HOLD / bounded UPDATE actions.

The original environment still projects every executed action. This module has
no teacher, contact threshold controller, simulator, training authorization or
physical-success claim. Dataset coverage must be reviewed before optimization.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from phase_gated_residual_policy_v6 import (
    PhaseGatedResidualPolicy, GATE_CONTRACT as PUBLIC_GATE_CONTRACT,
    FEATURES, FEATURE_SCHEMA, features, EpisodeActor, project_left_targets,
    EXPECTED_LEFT_NAMES, RECOVERY_JOINTS,
)
from zero_gated_residual_policy_v7 import candidate_bounds

BASE, HOLD, UPDATE = 0, 1, 2
SCHEMA = 'learned_base_hold_previous_increment_left28_v8_candidate'
CLASSIFICATION_CONTRACT = {
    'schema': 'projected_exact_base_then_exact_hold_then_update_v8_candidate',
    'class_order': ['BASE', 'HOLD', 'UPDATE'],
    'comparison_space': 'original float32 action roundtrip and project_target',
    'BASE': 'expert projected target bitwise equals projected frozen V4 target',
    'HOLD': 'not BASE and expert projected target bitwise equals projected previous actual command',
    'UPDATE': 'neither exact branch and distance from BOTH projected branches exceeds 1e-6 rad',
    'exact_branch_overlap_priority': 'BASE',
    'confidence_threshold_rad': .000001,
    'ambiguous': 'neither exact branch and within 1e-6 rad of at least one projected branch',
    'ambiguous_classification': 'excluded from cross entropy only',
    'ambiguous_continuous_regression': 'unmodified expert target assigned to UPDATE head regression',
    'continuous_regression': 'all original labels compared to label-selected projected branch; exact BASE/HOLD errors are zero by construction',
    'runtime_magnitude_deadband': None,
    'classification_labels_are_actor_inputs': False,
}
ACTION_CONTRACT = {
    'schema': SCHEMA,
    'classes': ['BASE', 'HOLD', 'UPDATE'],
    'decision': 'per-joint argmax of learned logits; ties use first class',
    'BASE': 'frozen V4 proposal',
    'HOLD': 'previous actually applied target; original runtime projection retained',
    'UPDATE': 'bounded increment about previous for named 11 recovery joints; otherwise residual about frozen V4',
    'previous_anchor_joint_names': sorted(RECOVERY_JOINTS),
    'original_joint_limits_and_projection_changed': False,
    'public_phase_gate': deepcopy(PUBLIC_GATE_CONTRACT),
    'gru_updates_on_every_frame': True,
    'public_gate_inactive_action': 'exact BASE regardless of logits',
    'features': FEATURE_SCHEMA,
    'feature_count': FEATURES,
    'teacher_hidden_state_or_labels_in_actor': False,
    'contact_impulse_threshold_is_hardcoded_into_actor': False,
    'right_joints_are_learned': False,
    'training_label_contract': deepcopy(CLASSIFICATION_CONTRACT),
    'qualification_required_before_training': True,
}


def classify_branch_labels(expert_projected, base_projected, hold_projected):
    """Training-only classification; preserves every original continuous label."""
    shape = expert_projected.shape
    if len(shape) < 2 or shape[-1] != 28:
        raise ValueError('Expected original float32 projected left28 labels')
    for value in (expert_projected, base_projected, hold_projected):
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Mismatched/nonfinite original float32 branch projections')
        if value.device != expert_projected.device:
            raise ValueError('Projected branches must share a device')
    def bits(value):
        return value.detach().contiguous().view(torch.int32)
    exact_base = bits(expert_projected) == bits(base_projected)
    exact_hold = ~exact_base & (bits(expert_projected) == bits(hold_projected))
    base_distance = (expert_projected.detach().double() - base_projected.detach().double()).abs()
    hold_distance = (expert_projected.detach().double() - hold_projected.detach().double()).abs()
    remaining = ~exact_base & ~exact_hold
    confident_update = remaining & (base_distance > .000001) & (hold_distance > .000001)
    ambiguous = remaining & ~confident_update
    # UPDATE is the regression branch for ambiguous values, never a substituted
    # or clipped label and never a forced runtime branch.
    targets = torch.full(shape, UPDATE, dtype=torch.long, device=expert_projected.device)
    targets = torch.where(exact_hold, HOLD, torch.where(exact_base, BASE, targets))
    return {
        'class_targets': targets,
        'classification_mask': exact_base | exact_hold | confident_update,
        'exact_base': exact_base, 'exact_hold': exact_hold,
        'confident_update': confident_update, 'ambiguous': ambiguous,
        'base_distance_rad': base_distance, 'hold_distance_rad': hold_distance,
    }


class HoldIncrementPolicy(PhaseGatedResidualPolicy):
    def __init__(self, baseline_saved, mean, std, update_bound, hidden_size=64,
                 left_joint_names=EXPECTED_LEFT_NAMES, public_gate_contract=None,
                 action_contract=None):
        if action_contract is not None and action_contract != ACTION_CONTRACT:
            raise ValueError('Unreviewed V8 action contract')
        names = tuple(left_joint_names)
        # Named bounds and anchor semantics must be checked against actual env
        # order by the eventual checkpoint loader before constructing this model.
        candidate_bounds(names)
        super().__init__(baseline_saved, mean, std, update_bound, hidden_size,
                         gate_contract=public_gate_contract)
        self.left_joint_names = names
        self.action_contract = deepcopy(ACTION_CONTRACT)
        self.mode_head = nn.Linear(self.hidden_size, 28 * 3)
        self.register_buffer('update_anchor_is_previous',
                             torch.tensor([n in RECOVERY_JOINTS for n in names], dtype=torch.bool))
        self.register_buffer('training_class_confidence_threshold_rad',
                             torch.tensor(.000001, dtype=torch.float64))
        # Initial logits tie, therefore the untrained model selects BASE.
        nn.init.zeros_(self.mode_head.weight)
        nn.init.zeros_(self.mode_head.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError('Partial loading is not a qualified policy')
        frozen = {'mean', 'std', 'residual_bound', 'residual_phase_active',
                  'squeeze_active_progress', 'update_anchor_is_previous',
                  'training_class_confidence_threshold_rad'}
        for name, expected in self.state_dict().items():
            if name in frozen or name.startswith('baseline.'):
                actual = state_dict.get(name)
                if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                        or actual.dtype != expected.dtype
                        or not torch.equal(
                            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8))):
                    raise ValueError('Checkpoint changes frozen model contract: ' + name)
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint tensor: ' + name)
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def forward_with_diagnostics(self, x, previous_left, hidden=None):
        if x.ndim != 3 or x.shape[-1] != FEATURES or previous_left.shape != (*x.shape[:2], 28):
            raise ValueError('Expected batch/time/features and actual previous left commands')
        if x.dtype != torch.float32 or previous_left.dtype != torch.float32:
            raise ValueError('Actual actor observations must retain float32 execution semantics')
        if not torch.isfinite(x).all() or not torch.isfinite(previous_left).all():
            raise ValueError('Nonfinite actual-state input')
        if hidden is not None:
            if hidden.shape != (1, x.shape[0], self.hidden_size) or not torch.isfinite(hidden).all():
                raise ValueError('Invalid actual-episode GRU history')
        encoded = self.encoder(((x - self.mean) / self.std).clamp(-12., 12.))
        history, next_hidden = self.gru(encoded, hidden)
        logits = self.mode_head(history).reshape(*x.shape[:2], 28, 3)
        base = self.baseline_targets(x, previous_left)
        anchor = torch.where(self.update_anchor_is_previous, previous_left, base)
        bounded_update = torch.tanh(self.head(history)) * self.effective_bounds(x)
        update = torch.clamp(anchor + bounded_update, self.baseline.lower.float(), self.baseline.upper.float())
        active = self.residual_active(x).unsqueeze(-1).expand_as(previous_left)
        mode = torch.where(active, logits.argmax(-1), torch.zeros_like(logits[..., 0], dtype=torch.long))
        proposed = torch.where(mode == BASE, base, torch.where(mode == HOLD, previous_left, update))
        # The third return reports the UPDATE head only; HOLD may lie farther
        # from the moving baseline than update_bound. It is not a hidden clamp.
        emitted_update = torch.where(mode == UPDATE, bounded_update, torch.zeros_like(bounded_update))
        if not all(torch.isfinite(v).all() for v in (logits, next_hidden, update, proposed, emitted_update)):
            raise ValueError('Nonfinite policy output')
        diagnostics = {
            'baseline_proposal': base, 'hold_proposal': previous_left,
            'update_proposal': update, 'update_anchor': anchor,
            'bounded_update': bounded_update, 'mode_logits': logits,
            'public_phase_active': active, 'predicted_class': mode,
        }
        return proposed, next_hidden, emitted_update, diagnostics

    def forward(self, x, previous_left, hidden=None):
        proposed, hidden, update, _ = self.forward_with_diagnostics(x, previous_left, hidden)
        return proposed, hidden, update


def supervision_losses(proposed, diagnostics, previous_left, raw_expert_labels,
                       transition_mask, class_targets, classification_mask,
                       projection_parameters, left_indices, projector):
    """Projected label-selected regression plus learned branch classification.

Every eligible raw target is included unchanged. Exact BASE/HOLD labels are
represented by those exact candidates; UPDATE and ambiguous labels train the
bounded head. The physical hard-decoded error is separately reported. Actor
inference never accepts these labels or this label-selected training path.
"""
    shape = proposed.shape
    if len(shape) != 3 or shape[-1] != 28:
        raise ValueError('Expected batch/time/left28 policy output')
    if previous_left.shape != shape or raw_expert_labels.shape != shape:
        raise ValueError('Continuous supervision shape mismatch')
    if (transition_mask.shape != shape[:2] or transition_mask.dtype != torch.bool
            or class_targets.shape != shape or class_targets.dtype != torch.long
            or classification_mask.shape != shape or classification_mask.dtype != torch.bool):
        raise ValueError('Explicit transition/classification masks and integer class labels required')
    if not all(torch.isfinite(v).all() for v in (proposed, previous_left, raw_expert_labels)):
        raise ValueError('Nonfinite continuous supervision')
    fields = ('baseline_proposal', 'hold_proposal', 'update_proposal', 'public_phase_active', 'predicted_class')
    if any(k not in diagnostics or diagnostics[k].shape != shape for k in fields):
        raise ValueError('Policy diagnostic shape mismatch')
    if not all(torch.isfinite(diagnostics[k]).all() for k in
               ('baseline_proposal', 'hold_proposal', 'update_proposal')):
        raise ValueError('Nonfinite branch proposal')
    if (diagnostics['hold_proposal'].dtype != previous_left.dtype
            or not torch.equal(diagnostics['hold_proposal'].contiguous().view(torch.int32),
                               previous_left.contiguous().view(torch.int32))):
        raise ValueError('HOLD proposal differs from previous actually applied command')
    logits = diagnostics.get('mode_logits')
    if logits is None or logits.shape != (*shape, 3) or not torch.isfinite(logits).all():
        raise ValueError('Invalid mode logits')
    if diagnostics['public_phase_active'].dtype != torch.bool or diagnostics['predicted_class'].dtype != torch.long:
        raise ValueError('Invalid phase/decoded mode types')
    continuous_mask = transition_mask.unsqueeze(-1).expand(shape)
    if not continuous_mask.any():
        raise ValueError('No qualified continuous labels')
    def project(value):
        return project_left_targets(value, previous_left, projection_parameters, left_indices, projector)
    base, hold, update = (project(diagnostics[k]) for k in ('baseline_proposal', 'hold_proposal', 'update_proposal'))
    reviewed = classify_branch_labels(raw_expert_labels, base, hold)
    if not torch.equal(class_targets, reviewed['class_targets']) or not torch.equal(classification_mask, reviewed['classification_mask']):
        raise ValueError('Labels differ from reviewed projected-space branch contract')
    selected = torch.where(class_targets == BASE, base, torch.where(class_targets == HOLD, hold, update))
    # Preserve original forced BASE behavior before the public feedback boundary.
    selected = torch.where(diagnostics['public_phase_active'], selected, base)
    error = selected.double() - raw_expert_labels.double()
    decoded_error = project(proposed).double() - raw_expert_labels.double()
    gate_mask = continuous_mask & classification_mask & diagnostics['public_phase_active']
    ce = F.cross_entropy(logits.reshape(-1, 3), class_targets.reshape(-1), reduction='none').reshape(shape)
    class_loss = ce[gate_mask].mean() if gate_mask.any() else logits.sum() * 0.
    predicted = diagnostics['predicted_class']
    return {
        'regression_mse_rad2': error[continuous_mask].square().mean(),
        'regression_max_abs_rad': error[continuous_mask].abs().max(),
        'hard_decoded_mse_rad2': decoded_error[continuous_mask].square().mean(),
        'hard_decoded_max_abs_rad': decoded_error[continuous_mask].abs().max(),
        'branch_cross_entropy': class_loss,
        'continuous_label_count': int(continuous_mask.sum()),
        'classification_label_count': int(gate_mask.sum()),
        'ambiguous_continuous_label_count': int((continuous_mask & reviewed['ambiguous']).sum()),
        'classification_error_count': int(((predicted != class_targets) & gate_mask).sum()),
        'class_target_counts': [int(((class_targets == k) & gate_mask).sum()) for k in (BASE, HOLD, UPDATE)],
        'raw_expert_labels_unchanged': True,
    }
