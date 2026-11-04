"""Host candidate: learned exact-baseline branch plus bounded left residual.

This module makes no supervision labels and contains no residual-magnitude
deadband. Gate targets must come from a separately reviewed dataset qualification.
The continuous expert targets remain unmodified in the projected regression loss.
The explicit class rule and candidate bounds still require dataset qualification
before any training run; this module alone does not grant that qualification.
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

SCHEMA = 'learned_per_joint_zero_gate_frozen_v4_actual_state_gru_left28_v7_candidate'
CLASSIFICATION_CONTRACT = {
    'schema': 'float32_projected_bitwise_zero_confident_nonzero_ambiguous_v7',
    'space': 'expert projected target versus frozen baseline after identical original float32 action roundtrip and project_target',
    'zero_class': 'float32 bit patterns exactly equal',
    'certain_nonzero_class': 'absolute target difference strictly greater than 1e-6 rad',
    'certain_nonzero_threshold_rad': .000001,
    'ambiguous_class': 'non-bitwise-equal and absolute difference <= 1e-6 rad',
    'ambiguous_classification_BCE': 'excluded',
    'ambiguous_continuous_regression': 'all raw expert labels retained unchanged',
    'ambiguous_values_are_declared_quantization_noise': False,
    'training_confidence_interval_is_runtime_action_deadband': False,
    'runtime_decision': 'logit > 0, regardless of residual amplitude',
}
ZERO_GATE_CONTRACT = {
    'schema': 'per_left_joint_binary_zero_or_bounded_residual_v7_candidate',
    'gate_count': 28,
    'decision': 'correct if and only if logit > 0 and original public phase gate is active',
    'logit_threshold': 0.0,
    'tie_decision': 'exact_frozen_baseline',
    'zero_branch': 'exact frozen V4 target, not the previous executed target',
    'residual_amplitude_deadband': None,
    'gate_and_residual_inputs': 'same whitelisted actual-state GRU history',
    'gru_updates_on_every_frame': True,
    'teacher_hidden_state_or_labels_are_actor_inputs': False,
    'right_joints_are_learned': False,
    'bounds_origin': 'explicit separately qualified named vector; never inferred from actor state',
    'supervision_label_contract': deepcopy(CLASSIFICATION_CONTRACT),
    'continuous_expert_labels_are_clipped_or_replaced': False,
    'public_phase_gate': deepcopy(PUBLIC_GATE_CONTRACT),
}
ARM_BOUND_CANDIDATE = dict(zip(('Joint'+str(i)+'_L' for i in range(1,8)),
                             (.0248,.0243,.0249,.0076,.0257,.0256,.0244)))
BOUND_CANDIDATE_CONTRACT = {
    'schema': 'five_development_history_named_residual_bounds_v7_candidate',
    'arm_bounds_rad': deepcopy(ARM_BOUND_CANDIDATE),
    'arm_rule': 'ceil(observed max abs expert_projected minus frozen_base_proposal, 1e-4 rad) plus 1e-3 rad',
    'recovery_finger_bound_rad': .006,
    'other_finger_bound_rad': .000001,
    'executed_joint_limits_or_project_target_changed': False,
    'physical_safety_or_generalization_claim': False,
    'requires_full_dataset_reachability_qualification': True,
}


def candidate_bounds(left_names):
    """Root-approved inspect candidate; not permission to bypass qualification."""
    if len(left_names)!=28 or len(set(left_names))!=28 or set(left_names)!=set(EXPECTED_LEFT_NAMES):
        raise ValueError('Named V7 bounds require exactly the actual 28 left DOFs')
    return [ARM_BOUND_CANDIDATE[name] if name in ARM_BOUND_CANDIDATE else
            .006 if name in RECOVERY_JOINTS else .000001 for name in left_names]


def classify_correction_labels(expert_projected, baseline_projected):
    """Reviewed training-label rule; never used to decode an actor action."""
    if (expert_projected.shape != baseline_projected.shape or expert_projected.ndim < 2
            or expert_projected.shape[-1] != 28 or expert_projected.dtype != torch.float32
            or baseline_projected.dtype != torch.float32):
        raise ValueError('Classification requires matching original float32 left28 projections')
    if not torch.isfinite(expert_projected).all() or not torch.isfinite(baseline_projected).all():
        raise ValueError('Nonfinite projected class label')
    exact_zero = expert_projected.detach().contiguous().view(torch.int32) == baseline_projected.detach().contiguous().view(torch.int32)
    difference = (expert_projected.detach().double() - baseline_projected.detach().double()).abs()
    correction = difference > CLASSIFICATION_CONTRACT['certain_nonzero_threshold_rad']
    ambiguous = ~exact_zero & ~correction
    return {'correction_target': correction, 'classification_mask': exact_zero | correction,
            'exact_zero': exact_zero, 'ambiguous': ambiguous, 'absolute_difference_rad': difference}


class ZeroGatedResidualPolicy(PhaseGatedResidualPolicy):
    def __init__(self, baseline_saved, mean, std, residual_bound, hidden_size=64,
                 public_gate_contract=None, zero_gate_contract=None):
        if zero_gate_contract is not None and zero_gate_contract != ZERO_GATE_CONTRACT:
            raise ValueError('Unreviewed learned-zero architecture contract')
        super().__init__(baseline_saved, mean, std, residual_bound, hidden_size,
                         gate_contract=public_gate_contract)
        self.zero_gate_contract = deepcopy(ZERO_GATE_CONTRACT)
        self.correction_gate = nn.Linear(self.hidden_size, 28)
        self.register_buffer('correction_logit_threshold', torch.tensor(0., dtype=torch.float32))
        # Serialized for audit only; actor decoding never reads this buffer.
        self.register_buffer('training_class_confidence_threshold_rad', torch.tensor(.000001, dtype=torch.float64))
        # A deterministic baseline initial policy. This is not a trained result.
        nn.init.zeros_(self.correction_gate.weight)
        nn.init.zeros_(self.correction_gate.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Reject changed control buffers / frozen baseline in checkpoint weights.

        The caller must first construct from separately verified baseline,
        normalization and reviewed named bounds. Dataset/source/provenance SHA
        verification belongs to the eventual V7 checkpoint loader, not this hook.
        """
        if not strict:
            raise ValueError('Partial checkpoint loading is not a qualified policy')
        current = self.state_dict()
        frozen = ('mean', 'std', 'residual_bound', 'residual_phase_active',
                  'squeeze_active_progress', 'correction_logit_threshold', 'training_class_confidence_threshold_rad')
        for name, expected in current.items():
            if name in frozen or name.startswith('baseline.'):
                actual = state_dict.get(name)
                if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                        or actual.dtype != expected.dtype
                        or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
                    raise ValueError('Checkpoint changes frozen model contract: ' + name)
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint tensor: ' + name)
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def forward_with_diagnostics(self, x, previous_left, hidden=None):
        if x.ndim != 3 or x.shape[-1] != FEATURES or previous_left.shape != (*x.shape[:2], 28):
            raise ValueError('Expected batch/time/features and actual previous left commands')
        if not torch.isfinite(x).all() or not torch.isfinite(previous_left).all():
            raise ValueError('Nonfinite actual-state input')
        if hidden is not None:
            if hidden.shape != (1, x.shape[0], self.hidden_size) or not torch.isfinite(hidden).all():
                raise ValueError('Invalid actual-episode GRU history')
        encoded = self.encoder(((x - self.mean) / self.std).clamp(-12., 12.))
        history, next_hidden = self.gru(encoded, hidden)
        logits = self.correction_gate(history)
        raw_residual = torch.tanh(self.head(history)) * self.effective_bounds(x)
        base = self.baseline_targets(x, previous_left)
        phase_active = self.residual_active(x).unsqueeze(-1)
        ungated = torch.clamp(base + raw_residual, self.baseline.lower.float(), self.baseline.upper.float())
        ungated = torch.where(phase_active, ungated, base)
        use_correction = phase_active & (logits > self.correction_logit_threshold)
        proposed = torch.where(use_correction, ungated, base)
        # This legacy third return is the selected bounded head output, not the
        # post-clamp/post-project_target physical action or a supervision label.
        emitted_residual = torch.where(use_correction, raw_residual, torch.zeros_like(raw_residual))
        if not all(torch.isfinite(value).all() for value in (logits, next_hidden, ungated, proposed, emitted_residual)):
            raise ValueError('Nonfinite policy output')
        diagnostics = {
            'baseline_proposal': base, 'ungated_proposal': ungated,
            'ungated_bounded_residual': raw_residual, 'correction_logits': logits,
            'public_phase_active': phase_active.expand_as(logits),
            'predicted_correction_class': use_correction,
        }
        return proposed, next_hidden, emitted_residual, diagnostics

    def forward(self, x, previous_left, hidden=None):
        proposed, hidden, residual, _ = self.forward_with_diagnostics(x, previous_left, hidden)
        return proposed, hidden, residual


def supervision_losses(proposed, diagnostics, previous_left, raw_expert_labels,
                       transition_mask, correction_targets, correction_label_mask,
                       projection_parameters, left_indices, projector):
    """Separate raw-target regression and binary classification, with no weights.

    All eligible continuous labels are retained, including any declared float32
    floor. Classification labels and their qualification mask are explicit inputs
    to this training-only function; neither is accepted by the actor forward API.
    Gate labels describe baseline-vs-correction, never previous-command hold.
    The caller must choose/report loss weights only after reviewing data evidence.
    """
    shape = proposed.shape
    if len(shape) != 3 or shape[-1] != 28:
        raise ValueError('Expected batch/time/left28 policy output')
    if raw_expert_labels.shape != shape or previous_left.shape != shape:
        raise ValueError('Continuous supervision shape mismatch')
    if (transition_mask.shape != shape[:2] or transition_mask.dtype != torch.bool
            or correction_targets.shape != shape or correction_targets.dtype != torch.bool
            or correction_label_mask.shape != shape or correction_label_mask.dtype != torch.bool):
        raise ValueError('Explicit boolean transition and per-joint classification masks required')
    if not all(torch.isfinite(t).all() for t in (proposed, previous_left, raw_expert_labels)):
        raise ValueError('Nonfinite continuous supervision')
    required = ('baseline_proposal', 'ungated_proposal', 'correction_logits', 'public_phase_active', 'predicted_correction_class')
    if any(name not in diagnostics or diagnostics[name].shape != shape for name in required):
        raise ValueError('Policy diagnostics shape mismatch')
    if not torch.isfinite(diagnostics['ungated_proposal']).all() or not torch.isfinite(diagnostics['correction_logits']).all():
        raise ValueError('Nonfinite training diagnostics')
    continuous_mask = transition_mask.unsqueeze(-1).expand(shape)
    if not continuous_mask.any():
        raise ValueError('No qualified continuous labels')
    projected_baseline = project_left_targets(diagnostics['baseline_proposal'], previous_left,
                                              projection_parameters, left_indices, projector)
    reviewed_classes = classify_correction_labels(raw_expert_labels, projected_baseline)
    if (not torch.equal(correction_targets, reviewed_classes['correction_target'])
            or not torch.equal(correction_label_mask, reviewed_classes['classification_mask'])):
        raise ValueError('Classification labels differ from the reviewed projected-space contract')
    gate_mask = continuous_mask & correction_label_mask & diagnostics['public_phase_active']
    projected_ungated = project_left_targets(diagnostics['ungated_proposal'], previous_left,
                                             projection_parameters, left_indices, projector)
    projected_decoded = project_left_targets(proposed, previous_left,
                                             projection_parameters, left_indices, projector)
    # Float64 difference retains small float32 label floors in reported losses.
    error = projected_ungated.double() - raw_expert_labels.double()
    decoded_error = projected_decoded.double() - raw_expert_labels.double()
    gate_loss_elements = F.binary_cross_entropy_with_logits(
        diagnostics['correction_logits'], correction_targets.to(diagnostics['correction_logits'].dtype), reduction='none')
    # A batch may contain only pre-feedback frames. Its zero BCE is explicit,
    # while every eligible continuous label still contributes to regression.
    gate_loss = gate_loss_elements[gate_mask].mean() if gate_mask.any() else diagnostics['correction_logits'].sum() * 0.
    predicted = diagnostics['predicted_correction_class']
    return {
        'regression_mse_rad2': error[continuous_mask].square().mean(),
        'regression_max_abs_rad': error[continuous_mask].abs().max(),
        'hard_decoded_mse_rad2': decoded_error[continuous_mask].square().mean(),
        'hard_decoded_max_abs_rad': decoded_error[continuous_mask].abs().max(),
        'gate_binary_cross_entropy': gate_loss,
        'continuous_label_count': int(continuous_mask.sum()),
        'classification_label_count': int(gate_mask.sum()),
        'ambiguous_continuous_label_count': int((continuous_mask & reviewed_classes['ambiguous']).sum()),
        'gate_false_positive_count': int((predicted & ~correction_targets & gate_mask).sum()),
        'gate_false_negative_count': int((~predicted & correction_targets & gate_mask).sum()),
        'raw_expert_labels_unchanged': True,
    }
