"""Frozen -6° right-index preshape at unloaded standby followed by measured wrist planning."""
import hashlib
import json
import math

import torch
from torch.nn.functional import normalize

from contact_hold90_cuda_domain import _specialize
import natural_face90_approach_anchor_probe as anchor
import natural_face90_pad_orientation_policy as wrist
import natural_face90_v5_controller as control
from natural_face90_v5_metrics import conj, rotate, unit

OFFSET_DEG = -6
JOINTS = tuple('right_index_' + joint + '_joint' for joint in ('MCP', 'PIP', 'DIP'))
ACTUAL_TOL_RAD = .015
TRACE = 'preshape_evidence.jsonl'


def standby_gate(obs, gap):
    return (obs['support'].bool() & obs['natural_ready'].bool() & obs['static_contact'].bool()
            & obs['structure_ok'].bool() & obs['allowed_layer_state'].bool() & ~obs['wrong_contact'].bool()
            & ~obs['index_contact'].bool() & ~obs['other_digit_contact'].bool() & ~obs['right_contact'].bool()
            & (obs['right_total_impulse'] == 0.) & ~obs['contact_goal_reached'].bool()
            & (gap >= wrist.MIN_STANDOFF_M))


class PreshapeController(control.NaturalFace90Controller):
    def __init__(self, e, parameters, deg):
        super().__init__(e, parameters, deg)
        self.index_ids = [e.names.index(j) for j in JOINTS]
        self.slots = [self.fIDs.index(j) for j in self.index_ids]
        self.index_goal = self.fGoal[self.slots] + .08 * self.p[:, 5, None] + math.radians(OFFSET_DEG)
        limits = e.limits.expand(self.n, -1, -1)[:, self.index_ids]
        if not ((self.index_goal >= limits[..., 0]) & (self.index_goal <= limits[..., 1])).all():
            raise ValueError('Frozen -6 degree index preshape outside joint limits')

    def index_targets(self, fingers, target, q, active, ph, obs):
        # Only the original standby shaping interval can change the three targets.
        # A load or a loss of clearance holds the last issued targets and resets the streak.
        staging = active & (ph == 1) & ~self.right_ready
        old = target[:, self.index_ids].double()
        self.shape_sample = None
        gate = torch.zeros_like(staging)
        if staging.any():
            gap = self.live_pad_clearance()
            gate = standby_gate(obs, gap)
            support_safe = (obs['support'].bool() & obs['static_contact'].bool() & obs['natural_ready'].bool()
                            & obs['structure_ok'].bool() & obs['allowed_layer_state'].bool())
            touch = (obs['right_contact'].bool() | obs['index_contact'].bool() | obs['other_digit_contact'].bool()
                     | obs['wrong_contact'].bool())
            self.shape_sample = (staging, gate, gap, obs['right_total_impulse'], old, q[:, self.index_ids],
                                 support_safe, touch, obs['contact_goal_reached'].bool())
        new = torch.where((staging & gate)[:, None], self.index_goal, old)
        fingers = fingers.clone()
        fingers[:, self.slots] = torch.where((ph >= 1)[:, None], new, fingers[:, self.slots])
        err = (q[:, self.index_ids] - self.index_goal).abs().amax(dim=1)
        reached = (old - self.index_goal).abs().amax(dim=1) < 1e-6
        return fingers, (gate & reached & (err <= ACTUAL_TOL_RAD))

    def write_step(self, result, just_ready, failed, shaped):
        if self.shape_sample is None:
            return
        staging, gate, gap, impulse, old, actual, support, touch, reached = self.shape_sample
        measurements = torch.cat((gap[:,None], impulse[:,None], old, actual, self.index_goal,
                                  result[:,self.index_ids].double(), gate[:,None], shaped[:,None],
                                  self.shape_streak[:,None], just_ready[:,None], failed[:,None],
                                  support[:,None], touch[:,None], reached[:,None]), 1)
        self.record_standby(staging, measurements)

    action = _specialize(control.NaturalFace90Controller.action, [
        ('shaped=(q[:,self.fIDs]-fingers).abs().amax(1)<.045',
         'fingers, index_settled = self.index_targets(fingers, target, q, active, ph, obs)\n'
         '    shaped=((q[:,self.fIDs]-fingers).abs().amax(1)<.045) & index_settled'),
        ('self.right_ref=torch.where((active&avoid)[:,None],result[:,ix].double(),self.right_ref)',
         'self.right_ref=torch.where((active&avoid)[:,None],result[:,ix].double(),self.right_ref)\n'
         '    self.write_step(result, just_ready, failed, shaped)'),
    ], {})


def sphere_choice(base_palm, dip_q, palm, vectors, radii, meshes, eligible):
    hard, rubber, center, outward = meshes
    dip_q = unit(dip_q)
    base = normalize(rotate(conj(dip_q), (palm @ base_palm.unsqueeze(-1)).squeeze(-1)), dim=1)
    tangent = outward - (base * outward).sum(dim=1, keepdim=True) * base
    # An aligned or antiparallel reference needs a well-defined azimuth basis.
    unit_axis = torch.eye(3, dtype=base.dtype, device=base.device)[base.abs().argmin(dim=1)]
    tangent = normalize(torch.where(tangent.norm(dim=1,keepdim=True) > 1e-8, tangent,
                                    torch.linalg.cross(base, unit_axis)), dim=1)
    perp = torch.linalg.cross(base, tangent)
    tilt = torch.deg2rad(torch.arange(31, device=base.device, dtype=base.dtype))[:,None,None]
    az = torch.deg2rad(torch.arange(0, 360, 4, device=base.device, dtype=base.dtype))[None,:,None]
    chosen_axis = base_palm.clone()
    rows = []
    # A lane at a time bounds GPU memory for the full two-mesh support projections.
    for i in range(len(base)):
        rays = tilt.cos() * base[i] + tilt.sin() * (az.cos() * tangent[i] + az.sin() * perp[i])
        gap = (rays @ rubber.T).amax(-1)
        rubber_lead = gap - (rays @ hard.T).amax(-1)
        axes = rotate(dip_q[i], rays) @ palm[i]
        margin = (axes @ vectors[i].T - radii).amin(-1)
        good = (eligible[i] & (rubber_lead >= wrist.MIN_RUBBER_LEAD_M)
                & (gap.abs() <= wrist.PAD_NORMAL_BAND_M) & (margin >= wrist.MIN_PASSIVE_LEAD_M)
                & ((rays @ normalize(center, dim=0)) >= .45))
        angle = good.any(dim=1).long().argmax()
        azimuth = margin[angle].masked_fill(~good[angle], -torch.inf).argmax()
        feasible = good.any()
        adjusted = feasible & (angle > 0)
        chosen_axis[i] = torch.where(adjusted, axes[angle, azimuth], base_palm[i])
        rows.append(torch.stack((feasible, adjusted, angle, 1000 * margin[angle, azimuth],
                                 1000 * gap[angle, azimuth], 1000 * rubber_lead[angle, azimuth], azimuth * 4)))
    fields = ('feasible', 'adjusted', 'angle_deg', 'passive_lead_mm', 'predicted_gap_mm', 'rubber_lead_mm','azimuth_deg')
    return chosen_axis, dict(zip(fields, torch.stack(rows).T, strict=True))


class PreshapeGuide(wrist.PadOrientationGuide):
    def __init__(self, c, geometry):
        super().__init__(c, geometry)
        self.azimuth = self.pad_gap_m.clone()
        self.handoff_error = self.pad_gap_m.clone()

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        result = super().command(target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs)
        ev = self.control_evidence()
        ev['pre_preshape_azimuth_deg'] = self.azimuth
        for j, joint in enumerate(JOINTS):
            ev['pre_preshape_q_' + joint] = q[:,self.c.index_ids[j]]
            ev['pre_preshape_goal_' + joint] = self.c.index_goal[:,j]
        return result

    def _plan(self, fresh, target, palm, inward, tip):
        anchor.ApproachGuide._plan(self, fresh, target, palm, inward, tip)
        if not fresh.any():
            return
        saved = self.orientation_plan
        if (fresh & (saved['planned'] | ~self.c.right_ready)).any():
            raise ValueError('Unsettled or repeated preshape handoff')
        live_q = self.c._q()[:,self.c.index_ids]
        self.handoff_error[fresh] = (live_q - self.c.index_goal).abs().amax(dim=1)[fresh]
        dip = self.c.e.robot.get_link_transforms()[fresh, self.index_link, 3:].double()
        axis, scores = sphere_choice(self.axis[fresh], dip, palm[fresh], self.vectors[fresh], self.radii,
                                    self.meshes, self.plan_eligible[fresh] & (self.handoff_error[fresh] <= ACTUAL_TOL_RAD))
        self.axis[fresh] = axis
        self.azimuth[fresh] = scores.pop('azimuth_deg')
        saved['planned'][fresh] = True
        saved['eligible'][fresh] = self.plan_eligible[fresh]
        for key, value in scores.items():
            saved[key][fresh] = value.to(saved[key].dtype)


class PadPreshapePhase(wrist.PadOrientationPhase):
    def init_controller(self, module):
        self.controller = PreshapeController(self.e, self.req['parameters'], self.req['target_deg'])
        self.max_steps = int(module.MAX_POST_STEPS)
        self.phase_names = tuple(module.PHASES)
        self.action_phase = self.controller.phase.clone()

    def __init__(self, e, req, out):
        if type(req.get('pad_preshape_offset_deg')) is not int or req['pad_preshape_offset_deg'] != OFFSET_DEG:
            raise ValueError('Expected frozen -6 degree index preshape')
        super().__init__(e, req, out)
        self.guide = PreshapeGuide(self.controller, req['pad_orientation_geometry'])
        self.controller.live_pad_clearance = self.live_pad_clearance
        self.controller.record_standby = self.record_standby
        self.shape_frames = [0] * self.n
        self.ready_steps = [None] * self.n
        self.shape_trace = (out / TRACE).open('x', buffering=1)

    def live_pad_clearance(self):
        core = self.e.views['Core'].get_transforms().double()
        local = rotate(conj(unit(core[:,3:])), self.index_tracker.pad_position_world() - core[:,:3])
        return (local.abs() - .045).clamp_min(0.).norm(dim=1)

    def record_standby(self, staging, values):
        lanes = staging.nonzero(as_tuple=True)[0]
        step = self.monitor.last_step + 1
        for i, v in zip(lanes.tolist(), values[lanes].tolist(), strict=True):
            row = dict(lane=i, seed=self.req['seeds'][i], step=step, gap_m=v[0], right_impulse_Ns=v[1],
                       previous_q=v[2:5], measured_q=v[5:8], goal_q=v[8:11], requested_q=v[11:14],
                       eligible=bool(v[14]), shaped=bool(v[15]), streak=int(v[16]), ready=bool(v[17]),
                       controller_failed=bool(v[18]), supported_safe=bool(v[19]),
                       contact_or_wrong=bool(v[20]), goal_reached=bool(v[21]))
            self.shape_trace.write(json.dumps(row, allow_nan=False) + '\n')
            self.shape_frames[i] += 1
            if row['ready']:
                self.ready_steps[i] = step

    def finish_learning(self):
        result = super().finish_learning()
        self.shape_trace.close()
        with (self.out / TRACE).open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        result['pad_preshape'] = dict(offset_deg=OFFSET_DEG, joints=list(JOINTS), trace_sha256=digest,
                                      frames_by_lane=self.shape_frames, ready_step_by_lane=self.ready_steps,
                                      handoff_error_rad_by_lane=self.guide.handoff_error.tolist(),
                                      azimuth_deg_by_lane=self.guide.azimuth.tolist(), orientation=result['pad_orientation'])
        return result


def verify_receipt(receipt, trace_path, req):
    wrist.verify_receipt(receipt['orientation'], trace_path, req)
    if (receipt['offset_deg'] != OFFSET_DEG or receipt['joints'] != list(JOINTS)
            or req['pad_preshape_offset_deg'] != OFFSET_DEG):
        raise ValueError('Unexpected preshape receipt')
    seeds = req['seeds']
    n = len(seeds)
    actual_frames, counts, stops = [0] * n, [0] * n, [None] * n
    last_step = [None] * n
    shape_file = trace_path.with_name(TRACE)
    with shape_file.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['trace_sha256']:
            raise ValueError('Preshape trace hash mismatch')
    with shape_file.open() as stream:
        for line in stream:
            row = json.loads(line)
            i, step = row['lane'], row['step']
            if (not 0 <= i < n or seeds[i] != row['seed'] or stops[i] is not None
                    or (last_step[i] is not None and step != last_step[i] + 1)):
                raise ValueError('Invalid standby lane or interrupted shaping record')
            last_step[i] = step
            old, actual, goal, requested = [row[k] for k in ('previous_q', 'measured_q', 'goal_q', 'requested_q')]
            gap, impulse = row['gap_m'], row['right_impulse_Ns']
            if (any(len(v) != 3 for v in (old, actual, goal, requested))
                    or not all(map(math.isfinite, (*old, *actual, *goal, *requested, gap, impulse)))):
                raise ValueError('Invalid preshape physical measurements')
            raw_goal = [.24, .22, .16]
            expected = [v + .08 * req['parameters'][i][5] + math.radians(OFFSET_DEG) for v in raw_goal]
            gate = (row['supported_safe'] and not row['contact_or_wrong'] and not row['goal_reached']
                    and gap >= wrist.MIN_STANDOFF_M and impulse == 0.)
            if row['eligible'] != gate or max(abs(a-b) for a,b in zip(expected, goal)) > 1e-7:
                raise ValueError('Preshape eligibility or target disagrees with frozen sensors/configuration')
            for before, after, wanted in zip(old, requested, goal):
                if ((not gate or row['controller_failed']) and abs(before - after) > 1e-7
                        or abs(before-after) > .0110002
                        or not min(before, wanted)-1e-7 <= after <= max(before, wanted)+1e-7):
                    raise ValueError('Loaded, overshot or unbounded index preshape command')
            counts[i] = counts[i] + 1 if row['shaped'] else 0
            settled = (gate and max(abs(a-b) for a,b in zip(actual,goal)) <= ACTUAL_TOL_RAD
                       and max(abs(a-b) for a,b in zip(old,goal)) < 1e-6)
            if ((row['shaped'] and not settled) or counts[i] != row['streak']
                    or (row['ready'] and (counts[i] < 10 or not settled or row['controller_failed']))):
                raise ValueError('Unverified measured shape or incomplete standby streak')
            actual_frames[i] += 1
            if row['ready']:
                stops[i] = step
    if actual_frames != receipt['frames_by_lane'] or stops != receipt['ready_step_by_lane']:
        raise ValueError('Standby counts differ from trace')
    orientation = receipt['orientation']
    handoff_error = receipt['handoff_error_rad_by_lane']
    azimuth = receipt['azimuth_deg_by_lane']
    if not len(handoff_error) == len(azimuth) == n:
        raise ValueError('Missing preshape handoff evidence')
    for i in range(n):
        if (not math.isfinite(handoff_error[i]) or handoff_error[i] < 0 or azimuth[i] not in range(0,360,4)
                or (orientation['planned_by_lane'][i] and stops[i] is None)
                or (orientation['feasible_by_lane'][i] and handoff_error[i] > ACTUAL_TOL_RAD)
                or (not orientation['planned_by_lane'][i] and (azimuth[i] or handoff_error[i]))):
            raise ValueError('Unverified handoff, wrist direction or measured preshape')

    with trace_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row['pre_pad_orientation_first_plan']:
                continue
            i = row['lane']
            measured = [row['pre_preshape_q_' + j] for j in JOINTS]
            goals = [row['pre_preshape_goal_' + j] for j in JOINTS]
            expected = [v + .08 * req['parameters'][i][5] + math.radians(OFFSET_DEG) for v in (.24,.22,.16)]
            if (not all(map(math.isfinite, (*measured, *goals)))
                    or max(abs(g - e) for g,e in zip(goals,expected)) > 1e-7
                    or abs(max(abs(q-g) for q,g in zip(measured,goals)) - handoff_error[i]) > 1e-7
                    or row['pre_preshape_azimuth_deg'] != azimuth[i] or stops[i] >= row['step']):
                raise ValueError('Measured first learned handoff disagrees with the preshape receipt')
