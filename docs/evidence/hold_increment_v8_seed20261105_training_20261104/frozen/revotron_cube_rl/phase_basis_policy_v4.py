"""Full-frame phase trajectory compression, with no state-feedback claim.

Inference evaluates learned polynomial/ReLU basis weights. It does not read
teacher targets, select a stored target row, or interpolate a target table.
"""
import torch
from torch import nn
from phase_contact_policy import PHASES, FEATURES, accepted_prefix, phase_progress, features, sha

SCHEMA = 'phase_polynomial_relu_fullframe_compression_left28_v4'


def basis(t, knots):
    t = t.to(dtype=torch.float64).reshape(-1, 1)
    return torch.cat((torch.ones_like(t), t, t**2, t**3, t**4, t**5,
                      torch.relu(t - knots.reshape(1, -1))), dim=1)


class BasisHead(nn.Module):
    def __init__(self, knots, coefficients):
        super().__init__()
        self.register_buffer('knots', torch.as_tensor(knots, dtype=torch.float64))
        self.coefficients = nn.Parameter(torch.as_tensor(coefficients, dtype=torch.float64))

    def forward(self, t):
        return basis(t, self.knots) @ self.coefficients


class PhaseBasisPolicy(nn.Module):
    def __init__(self, lower, upper, heads):
        super().__init__()
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer('upper', torch.as_tensor(upper, dtype=torch.float64))
        self.heads = nn.ModuleList([BasisHead(k, w) for k, w in heads])

    def forward(self, x):
        phase = x[:, 185:197].argmax(1)
        output = torch.empty((len(x), 28), dtype=torch.float64, device=x.device)
        for j, head in enumerate(self.heads):
            selected = phase == j
            if selected.any():
                output[selected] = head(x[selected, -1:])
        return torch.clamp(output, self.lower, self.upper)

    def targets(self, x, previous_left):
        previous_left = previous_left.to(dtype=torch.float64)
        return torch.clamp(self(x), torch.maximum(self.lower, previous_left - .04),
                           torch.minimum(self.upper, previous_left + .04))


def restored(saved):
    if saved['metadata']['schema'] != SCHEMA:
        raise ValueError('Checkpoint schema mismatch')
    state = saved['model']
    heads = [(state[f'heads.{j}.knots'], state[f'heads.{j}.coefficients']) for j in range(len(PHASES))]
    model = PhaseBasisPolicy(state['lower'], state['upper'], heads)
    model.load_state_dict(state)
    return model
