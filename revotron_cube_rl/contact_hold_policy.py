"""State-conditioned absolute left targets for the new successful-demo BC.

Inference still requires a public phase clock and the environment's original
56-joint target projection. This is not compatible with frozen V11 weights.
"""
import torch
from torch import nn
from phase_contact_policy import FEATURES


class ContactHoldPolicy(nn.Module):
    def __init__(self, mean, std, lower, upper, center, scale, width=256):
        super().__init__()
        for name, value in (('mean', mean), ('std', std), ('lower', lower), ('upper', upper), ('center', center), ('scale', scale)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(FEATURES, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 28))

    def forward(self, x):
        y = self.center + self.scale*self.net(((x-self.mean)/self.std).clamp(-15, 15))
        return torch.maximum(self.lower, torch.minimum(self.upper, y))

    def targets(self, x, previous_left, cap):
        # cap must come from the actual environment (including velocity limits).
        return previous_left + torch.clamp(self(x)-previous_left, -cap, cap)


def restored(saved):
    from contact_hold_dataset import SCHEMA
    if saved['metadata']['schema'] != SCHEMA:
        raise ValueError('Incompatible contact-hold checkpoint')
    s = saved['model']
    model = ContactHoldPolicy(*(s[n] for n in ('mean', 'std', 'lower', 'upper', 'center', 'scale')),
                              width=saved['metadata']['width'])
    model.load_state_dict(s, strict=True)
    return model.eval()
