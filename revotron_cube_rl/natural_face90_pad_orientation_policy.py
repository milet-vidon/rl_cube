"""One frozen, unloaded wrist-plan correction using the authored index collision hulls."""
import json
import math

import torch
from torch.nn.functional import normalize

import natural_face90_approach_anchor_probe as anchor
from natural_face90_v5_metrics import conj, rotate, unit

MAX_ANGLE_DEG = 30.
STEP_DEG = 2.
MIN_STANDOFF_M = .020
MIN_PASSIVE_LEAD_M = .002
MIN_RUBBER_LEAD_M = .00025
PAD_NORMAL_BAND_M = .006

FIELDS = ('planned', 'eligible', 'feasible', 'adjusted', 'angle_deg', 'passive_lead_mm',
          'predicted_gap_mm', 'rubber_lead_mm')
FLAGS = FIELDS[:4]


def geometry_tensors(geometry, reference):
    hard, rubber, center, normal = [reference.new_tensor(geometry[key]) for key in (
        'hard_vertices_dip_m', 'rubber_vertices_dip_m', 'anchor_dip_m', 'outward_axis_dip')]
    if any(v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 for v in (hard, rubber)):
        raise ValueError('Expected two 3D DIP-local collision hulls in metres')
    if (center.shape != (3,) or normal.shape != (3,)
            or not all(torch.isfinite(v).all() for v in (hard, rubber, center, normal))
            or abs(normal.norm().item() - 1.) > 1e-5 or center.norm().item() < 1e-8):
        raise ValueError('Invalid DIP-local anchor or outward direction')
    return hard - center, rubber - center, center, normalize(normal, dim=0)


def choose_axis(base_palm, dip_in_world, palm_in_world, vectors, radii, meshes, eligible):
    """Test the original plan, then successive angles on the shortest arc to the exposed pad."""
    hard, rubber, center, outward = meshes
    dip_in_world = unit(dip_in_world)
    base_world = (palm_in_world @ base_palm.unsqueeze(-1)).squeeze(-1)
    base = normalize(rotate(conj(dip_in_world), base_world), dim=1)
    dot = (base * outward).sum(1).clamp(-1., 1.)
    tangent = outward - dot[:, None] * base
    # Opposite directions do not define a unique shortest arc. Evaluate only the original plan.
    has_arc = tangent.norm(dim=1) > 1e-9
    tangent = normalize(tangent, dim=1)
    arc = torch.where(has_arc, torch.acos(dot), 0.)
    ring = torch.deg2rad(torch.arange(0, MAX_ANGLE_DEG + STEP_DEG, STEP_DEG,
                                     device=base.device, dtype=base.dtype))
    angles = torch.minimum(arc[:, None], ring)
    candidates = angles.cos()[:, :, None] * base[:, None] + angles.sin()[:, :, None] * tangent[:, None]
    axes_world = rotate(dip_in_world[:, None], candidates)
    axes_palm = axes_world @ palm_in_world

    gap = (candidates @ rubber.T).amax(-1)
    rubber_lead = gap - (candidates @ hard.T).amax(-1)
    passive_lead = (axes_palm @ vectors.transpose(1, 2) - radii).amin(-1)
    protrusion = candidates @ normalize(center, dim=0)
    usable = ((rubber_lead >= MIN_RUBBER_LEAD_M) & (gap.abs() <= PAD_NORMAL_BAND_M)
              & (passive_lead >= MIN_PASSIVE_LEAD_M) & (protrusion >= .45))
    feasible = eligible & usable.any(dim=1)
    # argmax returns the first valid sample. The original plan remains exact on fallback.
    chosen = torch.where(feasible, usable.long().argmax(1), 0)
    lane = torch.arange(len(base), device=base.device)
    angle_deg = torch.rad2deg(angles[lane, chosen])
    adjusted = feasible & (angle_deg > 1e-8)
    return (torch.where(adjusted[:, None], axes_palm[lane, chosen], base_palm),
            dict(feasible=feasible, adjusted=adjusted, angle_deg=angle_deg,
                 passive_lead_mm=1000 * passive_lead[lane, chosen], predicted_gap_mm=1000 * gap[lane, chosen],
                 rubber_lead_mm=1000 * rubber_lead[lane, chosen]))


class PadOrientationGuide(anchor.ApproachGuide):
    def __init__(self, controller, geometry):
        super().__init__(controller)
        self.meshes = geometry_tensors(geometry, self.reference)
        self.orientation_plan = {k: torch.zeros_like(self.aligned if k in FLAGS else self.pad_gap_m)
                                 for k in FIELDS}

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        self.plan_eligible = (fresh & learned & obs['natural_ready'].bool() & obs['support'].bool()
                              & obs['static_contact'].bool() & obs['structure_ok'].bool()
                              & obs['allowed_layer_state'].bool() & ~obs['wrong_contact'].bool()
                              & (obs['right_total_impulse'] == 0.) & ~obs['right_contact'].bool()
                              & ~obs['index_contact'].bool() & ~obs['other_digit_contact'].bool()
                              & ~obs['contact_goal_reached'].bool() & (local_tip[:, 2] <= -.045 - MIN_STANDOFF_M))
        result = super().command(target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs)
        state = self.control_evidence()
        state['pre_pad_orientation_first_plan'] = fresh
        state.update({'pre_pad_orientation_' + key: value for key, value in self.orientation_plan.items()})
        return result

    def _plan(self, fresh, target, palm, inward, tip):
        super()._plan(fresh, target, palm, inward, tip)
        if not fresh.any():
            return
        saved = self.orientation_plan
        if (fresh & saved['planned']).any():
            raise ValueError('Repeated handoff in a frozen single-episode lane')
        dip = self.c.e.robot.get_link_transforms()[fresh, self.index_link, 3:].double()
        axis, metrics = choose_axis(self.axis[fresh], dip, palm[fresh], self.vectors[fresh],
                                    self.radii, self.meshes, self.plan_eligible[fresh])
        self.axis[fresh] = axis
        saved['eligible'][fresh] = self.plan_eligible[fresh]
        saved['planned'][fresh] = True
        for key, values in metrics.items():
            saved[key][fresh] = values

    def receipt(self):
        return {key + '_by_lane': value.tolist() for key, value in self.orientation_plan.items()}


class PadOrientationPhase(anchor.ProbePhase):
    def __init__(self, e, req, out):
        super().__init__(e, req, out)
        self.guide = PadOrientationGuide(self.controller, req['pad_orientation_geometry'])
        if not torch.allclose(self.pad_offset, self.guide.meshes[2], atol=1e-8, rtol=0.):
            raise ValueError('The collision hull anchor differs from the live calibrated pad')

    def finish_learning(self):
        result = super().finish_learning()
        result['pad_orientation'] = self.guide.receipt()
        return result


def verify_receipt(receipt, trace_path, req):
    """Reconcile per-lane plans with the measured handoff and the subsequent immutable plan."""
    seeds = req['seeds']
    keys = {k + '_by_lane' for k in FIELDS}
    if set(receipt) != keys or any(len(receipt[k]) != len(seeds) for k in keys):
        raise ValueError('Incomplete pad-orientation receipt')
    seen = set()
    with trace_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            lane = row['lane']
            if not 0 <= lane < len(seeds) or row['seed'] != seeds[lane]:
                raise ValueError('Pad-orientation trace has a mismatched lane')
            if (lane not in seen) != bool(row['pre_pad_orientation_first_plan']):
                raise ValueError('Expected exactly one pad-orientation handoff for each learned lane')
            state = {k: row['pre_pad_orientation_' + k] for k in FIELDS}
            if any(not math.isfinite(state[k]) or state[k] != receipt[k + '_by_lane'][lane] for k in FIELDS):
                raise ValueError('Pad-orientation plan changed or differs from receipt')
            if not state['planned']:
                raise ValueError('Missing pad-orientation handoff')
            if lane in seen:
                continue
            seen.add(lane)
            eligible = bool(row['pre_support_ok'] and not row['pre_approach_any_touch']
                            and not row['pre_approach_goal_reached'] and row['pre_normal_gap_m'] >= MIN_STANDOFF_M)
            if eligible != bool(state['eligible']):
                raise ValueError('Pad-orientation plan disagrees with physical unloaded standoff')
            feasible, angle = bool(state['feasible']), state['angle_deg']
            if (not 0. <= angle <= MAX_ANGLE_DEG + 1e-8 or bool(state['adjusted']) != (angle > 1e-8)
                    or (angle > 1e-8 and not feasible)
                    or (feasible and (not eligible or state['passive_lead_mm'] < 1000 * MIN_PASSIVE_LEAD_M
                                      or abs(state['predicted_gap_mm']) > 1000 * PAD_NORMAL_BAND_M
                                      or state['rubber_lead_mm'] < 1000 * MIN_RUBBER_LEAD_M))):
                raise ValueError('Pad-orientation choice violates frozen geometric bounds')
    if any(any(receipt[k][i] for k in keys) for i in range(len(seeds)) if i not in seen):
        raise ValueError('Pad-orientation receipt claimed a plan for an unobserved lane')
