"""Learn contact and the first two degrees of isolated drive in the original R27 scene."""
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from contact_hold90_cuda_domain import _specialize
import natural_face90_cartesian as base
import natural_face90_cartesian_worker as cart
from natural_face90_hover_d_metrics import CONTACT_IMPULSE
from natural_face90_v5_metrics import wrap


CONTACT_SCHEMA = 'natural_face90_cartesian_contact_curriculum_v1'
FACE_SCHEMA = 'natural_face90_cartesian_contact_to15_v1'
DEGREES = 2.
SCALE_DEG = .02
APPROACH_SCALE_M = .025
APPROACH_MAX = .5
FIRST_BONUS = 1.
DRIVE_MAX = 8.
WRONG_PENALTY = 1.
UNSAFE_PENALTY = 20.
REWARD_FIELDS = ('approach', 'first_contact', 'drive', 'wrong_first', 'unsafe', 'effort_time')


def reward_contract(stage):
    if stage == 'face15':
        return dict(name='original_cartesian_face15_v1')
    if stage != 'contact2':
        raise ValueError('Unknown Cartesian learning stage')
    return dict(name='safe_first_and_sustained_isolated_drive_v1', drive_goal_deg=DEGREES,
                log_scale_deg=SCALE_DEG, approach_scale_m=APPROACH_SCALE_M, approach_max=APPROACH_MAX,
                first_bonus=FIRST_BONUS, drive_max=DRIVE_MAX, wrong_first_penalty=WRONG_PENALTY,
                unsafe_penalty=UNSAFE_PENALTY, sustained_drive_intervals=2,
                weights_from='native_impulses_pad_position_and_measured_layer_rotation')


def restore_stage(model, optimizer, checkpoint, contract, req):
    stage = req['curriculum_stage']
    if req['recipe_schema'] != {'contact2':CONTACT_SCHEMA, 'face15':FACE_SCHEMA}[stage]:
        raise ValueError('Inconsistent stage and checkpoint schema')
    expected = contract
    origin = req.get('contact_origin')
    if origin is not None:
        if (stage != 'face15' or not req['learn'] or req['role'] != 'train'
                or origin['iteration'] != req['iteration']-1 or origin['checkpoint'] != req['checkpoint']
                or origin['sha'] != req['checkpoint_sha']):
            raise ValueError('Invalid contact-to-face checkpoint promotion')
        expected = dict(contract, schema=CONTACT_SCHEMA, curriculum_stage='contact2',
                        reward_contract=reward_contract('contact2'))
    if ('transfer' in req or 'promote_from_15' in req or checkpoint['contract'] != expected
            or checkpoint['iteration'] != req['iteration'] - int(req['learn'])):
        raise ValueError('Expected the committed Cartesian stage checkpoint with identical physical contract')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if req['learn']:
        optimizer.load_state_dict(checkpoint['optimizer'])


class ContactReward:
    """All milestone credit is lane-local and paid once; a new push starts at zero."""
    def __init__(self, n, device):
        self.seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.clean = self.seen.clone()
        self.started = self.seen.clone()
        self.prev_active = self.seen.clone()
        self.prev_net = torch.zeros(n, dtype=torch.float64, device=device)
        self.best_net = self.prev_net.clone()
        self.best_proximity = self.prev_net.clone()
        self.totals = self.prev_net.new_zeros(n, len(REWARD_FIELDS))

    def update(self, valid, unsafe, pre_touch, touch, isolated, active, net, support,
               pre_distance, post_distance, latent):
        safe = valid & ~unsafe
        untouched = ~self.seen & ~pre_touch
        first = safe & untouched & touch
        clean_first = first & isolated
        self.clean |= clean_first
        # Only approach in measured support before the first contact can earn
        # distance credit. Approach already achieved at handoff is the baseline.
        before = 1. / (1. + pre_distance / APPROACH_SCALE_M)
        after = 1. / (1. + post_distance / APPROACH_SCALE_M)
        self.best_proximity = torch.where(valid & ~self.started, before, self.best_proximity)
        self.started |= valid
        improved = torch.where(safe & support & untouched & ~touch,
                               (after-self.best_proximity).clamp_min(0.), 0.)
        self.best_proximity += improved
        self.seen |= valid & (pre_touch | touch)
        active = safe & self.clean & active
        net = net.clamp(0., DEGREES)
        # Two consecutive commanded intervals with qualified physical contacts
        # at both endpoints; cancel spikes, reset on dropouts, never repay rocking.
        sustained = torch.where(active & self.prev_active, torch.minimum(self.prev_net, net), 0.)
        best = torch.maximum(self.best_net, sustained)
        drive_gain = ((torch.log1p(best/SCALE_DEG) - torch.log1p(self.best_net/SCALE_DEG))
                      * (DRIVE_MAX / math.log1p(DEGREES/SCALE_DEG)))
        self.prev_active, self.prev_net, self.best_net = active, net, best
        cost = .001 + .002 * latent.double().tanh().square().mean(-1)
        terms = torch.stack((APPROACH_MAX * improved, FIRST_BONUS * clean_first, drive_gain,
                             -WRONG_PENALTY * (first & ~isolated), -UNSAFE_PENALTY * (valid & unsafe),
                             -torch.where(valid, cost, 0.)), dim=1)
        self.totals += terms
        return terms.sum(dim=1).float()


def check_software(req):
    from natural_face90_cartesian_contact import fingerprint, worker_software
    if req['software'] != worker_software(fingerprint()):
        raise ValueError('Cartesian contact curriculum software changed')
    inherited = base.worker_software(base.fingerprint())
    if any(req['software'].get(k) != v for k,v in inherited.items()):
        raise ValueError('Inherited Cartesian controller/worker software changed')
    return inherited


class ContactPhase(cart.CartesianPhase):
    def __init__(self, e, req, out):
        # The archived component validates its original code separately. The
        # outer rollout and checkpoint bind the complete curriculum software.
        pins = check_software(req)
        super().__init__(e, dict(req, software=pins, checkpoint=None, checkpoint_sha=None), out)
        self.req = req
        self.software = req['software']
        self.contract.update(software=self.software, curriculum_stage=req['curriculum_stage'],
                             reward_contract=reward_contract(req['curriculum_stage']))
        try:
            if req['checkpoint'] is not None:
                path = Path(req['checkpoint'])
                if cart.recipe.sha(path) != req['checkpoint_sha']:
                    raise ValueError('Changed Cartesian curriculum input checkpoint')
                ck = torch.load(path, map_location=e.home.device, weights_only=True)
                restore_stage(self.model, self.algorithm.optimizer, ck, self.contract, req)
            elif req['curriculum_stage'] != 'contact2' or req['iteration'] != 1 or not req['learn']:
                raise ValueError('Only the first contact training wave starts without a checkpoint')
            self.milestones = ContactReward(self.n, self.device)
        except BaseException:
            self.guide.placo.close()
            raise

    def after_step(self, before_post, alive, unsafe_seen, reasons, steps):
        if self.req['curriculum_stage'] == 'face15':
            return super().after_step(before_post, alive, unsafe_seen, reasons, steps)
        if self.pending is None:
            return
        p = self.pending
        valid = p['valid']
        done = valid & ~alive
        before = p['obs'][:, :3].double().norm(dim=1) / 100.
        obs, _, _, _ = self.state(p['command'], ((self.max_steps-steps).float()/self.max_steps).clamp(0,1))
        distance = obs[:, :3].double().norm(dim=1) / 100.
        ev = self.guide.control_evidence()
        pre_touch = ev['pre_contact'] | ev['pre_index_contact'] | ev['pre_other_digit_contact']
        m = self.obs
        touch = (m['target_impulse'] > CONTACT_IMPULSE) | m['index_contact'] | m['other_digit_contact']
        reward = self.milestones.update(valid, unsafe_seen, pre_touch, touch, self.response.previously_isolated,
                                        self.response.active, wrap(m['face_deg'] - self.response.start),
                                        ev['pre_support_ok'], before, distance, p['latent'])
        p.pop('potential')
        p.pop('command')
        p.update(reward=reward, done=done)
        if self.training:
            self.data.append({key:value.detach().clone() for key,value in p.items()})
        self.cumulative_reward += reward
        for i in (done | (valid & (steps%60==0))).nonzero(as_tuple=True)[0].tolist():
            self.reward_log.write(json.dumps(dict(lane=i,seed=self.req['seeds'][i],post_step=int(steps[i]),
                learned_steps=int(self.learn_steps[i]),reward=float(reward[i]),return_sum=float(self.cumulative_reward[i]),
                contact_marker_distance_m=float(distance[i]),unsafe=bool(unsafe_seen[i]),terminal=bool(done[i]),
                reason=reasons[i]),allow_nan=False)+'\n')
        self.pending = None

    def finish_learning(self):
        req = self.req
        try:
            pins = check_software(req)
        except BaseException:
            self.guide.placo.close()
            raise
        # The existing writer checks its own component pins, records the real
        # parent SHA, and saves our full outer contract/model/optimizer.
        self.req = dict(req, software=pins)
        try:
            learn = super().finish_learning()
        finally:
            self.req = req
        learn.update(curriculum_stage=req['curriculum_stage'], reward_contract=self.contract['reward_contract'])
        if req['curriculum_stage'] == 'contact2':
            learn['contact_reward_totals_by_lane'] = self.milestones.totals.tolist()
            learn['contact_reward_fields'] = list(REWARD_FIELDS)
        return learn


run = FunctionType(cart.material.run.__code__, dict(cart.material.run.__globals__, PostPhase=ContactPhase))
main = _specialize(cart.recipe.main, [
    ('from natural_face90_recipe_long import fingerprint, worker_software, validate_request',
     'from natural_face90_cartesian_contact import fingerprint, worker_software, validate_request'),
], dict(run=run, __doc__=__doc__))


class _Replay:
    """Reconstruct one lane's payments using only logged physical readbacks."""
    def __init__(self):
        self.frames = 0
        self.seen = self.clean = self.last_isolated = self.last_active = False
        self.start_angle = self.last_angle = self.last_net = self.peak = 0.
        self.proximity = 0.
        self.totals = np.zeros(len(REWARD_FIELDS))

    def step(self, row, unsafe):
        before_touch = row['pre_contact'] or row['pre_index_contact'] or row['pre_other_digit_contact']
        index = row['index_distal_impulse_Ns'] > CONTACT_IMPULSE
        other = row['other_digit_target_impulse_Ns'] > CONTACT_IMPULSE
        touch = index or other or row['right_target_impulse_Ns'] > CONTACT_IMPULSE
        isolated = bool(row['qualified_index_frame'] and not other)
        untouched = not self.seen and not before_touch
        first = not unsafe and untouched and touch
        self.clean |= first and isolated
        self.totals[1] += FIRST_BONUS * bool(first and isolated)
        self.totals[3] -= WRONG_PENALTY * bool(first and not isolated)
        x, y = (row['pre_material_marker_' + axis + '_m'] for axis in 'xy')
        pre_marker = (x,y,-.045)
        pre_pad = tuple(row['pre_cartesian_pad_' + axis + '_m'] for axis in 'xyz')
        if not self.frames:
            self.proximity = 1. / (1. + math.dist(pre_marker,pre_pad)/APPROACH_SCALE_M)
        if not unsafe and row['pre_support_ok'] and untouched and not touch:
            theta = math.radians(row['face_deg'] - row['pre_material_face_deg'])
            c, s = math.cos(theta), math.sin(theta)
            post_marker = (x*c+y*s,y*c-x*s,-.045)
            pad = tuple(row['pad_' + axis + '_m'] for axis in 'xyz')
            new = max(self.proximity, 1./(1.+math.dist(post_marker,pad)/APPROACH_SCALE_M))
            self.totals[0] += APPROACH_MAX*(new-self.proximity)
            self.proximity = new
        self.seen |= bool(before_touch or touch)
        active = bool(not unsafe and self.clean and row['drive_command'] and isolated and self.last_isolated)
        face = row['face_deg']
        if active and not self.last_active:
            self.start_angle = self.last_angle
        net = min(DEGREES,max(0.,(face-self.start_angle+180.)%360.-180.))
        if active and self.last_active:
            peak = max(self.peak, min(net,self.last_net))
            self.totals[2] += DRIVE_MAX*(math.log1p(peak/SCALE_DEG)-math.log1p(self.peak/SCALE_DEG))/math.log1p(DEGREES/SCALE_DEG)
            self.peak = peak
        self.totals[4] -= UNSAFE_PENALTY*bool(unsafe)
        self.totals[5] -= .001 + .002 * sum(math.tanh(row[f'pre_cartesian_latent_{i}'])**2 for i in range(3))/3.
        self.last_angle, self.last_net = face, net
        self.last_isolated, self.last_active = isolated, active
        self.frames += 1


def verify_stage_report(report, req, rows, learning):
    stage = req['curriculum_stage']
    if learning.get('curriculum_stage') != stage or learning.get('reward_contract') != reward_contract(stage):
        raise ValueError('Missing or mismatched Cartesian curriculum objective')
    if stage != 'contact2':
        return
    if learning.get('contact_reward_fields') != list(REWARD_FIELDS):
        raise ValueError('Unexpected contact reward fields')
    n = len(req['seeds'])
    accum = [_Replay() for _ in range(n)]
    last = [None]*n
    with (Path(report).parent/'turn_contact_evidence.jsonl').open() as stream:
        for line in stream:
            frame = json.loads(line)
            lane = frame['lane']
            if not 0<=lane<n or frame['seed'] != req['seeds'][lane]:
                raise ValueError('Invalid physical lane for curriculum rewards')
            if last[lane] is not None:
                if frame['step'] != last[lane]['step']+1:
                    raise ValueError('Nonconsecutive physical readbacks in contact training')
                accum[lane].step(last[lane], False)
            last[lane] = frame
    for lane, end in enumerate(rows):
        if last[lane] is not None:
            accum[lane].step(last[lane], end['unsafe'])
    actual = np.asarray(learning['contact_reward_totals_by_lane'])
    expected = np.stack([a.totals for a in accum])
    if (actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-6, atol=2e-6)
            or sum(a.frames for a in accum) != learning['learned_transitions']):
        raise ValueError('Contact curriculum rewards differ from native contact and signed layer readbacks')
    terminal = set()
    with (Path(report).parent/'policy_transitions.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if not event['terminal']:
                continue
            i = event['lane']
            if (not 0<=i<n or i in terminal or event['seed'] != req['seeds'][i]
                    or event['learned_steps'] != accum[i].frames or event['unsafe'] != rows[i]['unsafe']
                    or not math.isclose(event['return_sum'],float(expected[i].sum()),abs_tol=.001,rel_tol=1e-5)):
                raise ValueError('PPO return differs from reconstructed contact rewards')
            terminal.add(i)
    if terminal != {i for i in range(n) if accum[i].frames}:
        raise ValueError('Missing completed learned lane in contact rewards')


if __name__ == '__main__':
    main()
