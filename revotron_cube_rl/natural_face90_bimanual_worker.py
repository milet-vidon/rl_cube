"""Frozen R27 pickup and presentation followed by bilateral grip and face-axis PPO."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
from types import FunctionType

import torch

from contact_hold90_cuda_domain import _specialize
import natural_face90_recipe_worker as recipe
import natural_face90_bimanual_policy as policy
from natural_face90_bimanual_contacts import GripContacts, IMPULSE, QUIET, SIDE_BITS, opposed
from natural_face90_recipe_policy import PPO
from natural_face90_v5_metrics import wrap


class GripDrive:
    """Signed rotation within a single uninterrupted, physically qualified push."""
    def __init__(self, n, device):
        z = torch.zeros(n, dtype=torch.float64, device=device)
        self.handoff = z.clone()
        self.start = z.clone()
        self.credit = z.clone()
        self.peak = z.clone()
        self.previous = z.bool()
        self.reached = z.bool()
        self.held = z.bool()

    def update(self, valid, safe, pre_contact, contact, orbit, before, face, target, native_goal):
        drive = valid & safe & pre_contact & contact & (orbit > 0)
        self.start = torch.where(drive & ~self.previous, before, self.start)
        net = torch.minimum(wrap(face-self.start), wrap(face-self.handoff)).clamp_min(0.)
        now = torch.where(drive, net, 0.)
        sustained = torch.where(drive & self.previous, torch.minimum(now, self.credit), 0.)
        self.peak = torch.maximum(self.peak, sustained)
        self.reached |= drive & (sustained >= target) & (face >= target) & native_goal
        self.previous = drive
        self.credit = now
        return sustained


def restore(model, optimizer, contract, req):
    ck_path = req['checkpoint']
    if ck_path is None:
        if not req['learn'] or req['iteration'] != 1 or req['target_deg'] != 2 or req['checkpoint_sha'] is not None:
            raise ValueError('A fresh grip policy starts at iteration 1, target +2')
        return
    if recipe.sha(ck_path) != req['checkpoint_sha']:
        raise ValueError('Changed grip checkpoint')
    ck = torch.load(ck_path, map_location=next(model.parameters()).device, weights_only=True)
    if ck['contract'] != contract or ck['iteration'] != req['iteration'] - int(req['learn']):
        raise ValueError('Incompatible grip policy or checkpoint iteration')
    model.load_state_dict(ck['policy'], strict=True)
    if req['learn']:
        optimizer.load_state_dict(ck['optimizer'])


_import_campaign = ('from natural_face90_recipe_long import fingerprint, worker_software',
                    'from natural_face90_bimanual import fingerprint, worker_software')
_finish = _specialize(recipe.PostPhase.finish_learning, [_import_campaign], {})
# Add the observer's support flag to its existing independent layer/contact log.
_observe = _specialize(recipe.ScriptedPostPhase.observe, [
    ("fields=('natural_ready','turn_started'", "fields=('support','natural_ready','turn_started'"),
], {})


class BimanualPhase(recipe.ScriptedPostPhase):
    def __init__(self, e, req, out, resources):
        from natural_face90_bimanual import fingerprint, worker_software
        from natural_face90_bimanual_control import BimanualGuide
        goal = req['target_deg']
        if goal not in (2, 15, 90):
            raise ValueError('Unknown grip curriculum angle')
        # Presentation uses the original controller up to its first outside approach.
        # The observer and its hold/structure windows see the actual goal starting
        # at the very first physical readback, including the initial +2 milestone.
        super().__init__(e, dict(req, target_deg=15 if goal == 2 else goal), out)
        self.req = req
        self.monitor.target = self.monitor.machine.target = float(goal)
        if req['software'] != worker_software(fingerprint()) or any(
                req['software'].get(k) != v for k, v in self.software.items()):
            raise ValueError('Grip worker source pins changed')
        self.software = req['software']
        self.guide = BimanualGuide(self.controller)
        resources.callback(self.guide.placo.close)
        if self.guide.placo.receipt != req['placo']:
            raise ValueError('Grip IK differs from the requested Placo runtime')
        self.tap = GripContacts(e, self.monitor)
        self.contract = dict(schema=req['recipe_schema'], policy_schema=policy.SCHEMA,
                             parent_sha=recipe.PARENT_HASH, software=self.software, placo=req['placo'],
                             scene=e.domain_receipt['scene_sha256'], joint_names=list(e.names),
                             inputs=list(policy.INPUT_FIELDS), actions=list(policy.ACTION_AXES),
                             goals=[2, 15, 90], control_hz=30, physics_hz=240)
        torch.manual_seed(req['policy_seed'])
        self.model = policy.BimanualExpert().to(self.device)
        self.algorithm = PPO(self.model)
        self.training = req['learn']
        restore(self.model, self.algorithm.optimizer, self.contract, req)
        torch.manual_seed(req['policy_seed'] + req['iteration']*104729 + int(not self.training))
        self.model.eval()
        self.learning = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.hold_target = e.home.clone()
        self.learn_steps = self.learning.long()
        self.counts = {k:self.learn_steps.clone() for k in ('thumb_target_frames','index_target_frames',
                       'bilateral_frames','bilateral_substeps','opposed_substeps','same_layer_overlap_substeps')}
        self.cumulative_reward = self.learning.float()
        self.grip = GripDrive(self.n, self.device)
        self.rewards = policy.BimanualReward(self.n, self.device)
        self.pending = None
        self.data = []
        self.max_steps = 2200
        self.phase_names += ('grip_approach', 'grip_drive', 'grip_retreat')
        self.reward_log = resources.enter_context((out/'policy_transitions.jsonl').open('x', buffering=1))
        self.trace = resources.enter_context((out/'bimanual_contacts.jsonl').open('x', buffering=1))

    def geometry(self):
        q = self.e.robot.get_dof_positions()[:, self.e.ids]
        core = self.e.views['Core'].get_transforms().double()
        geom = self.guide.geometry(q, core, self.obs['face_deg'])
        return q, core, geom

    def propose(self, target, active, steps):
        m = self.obs
        fresh = active & ~self.learning & m['natural_ready'] & self.controller.detour_done & (self.controller.phase >= 2)
        self.hold_target[fresh] = target[fresh]
        self.grip.handoff[fresh] = m['face_deg'][fresh]
        self.learning |= fresh
        live = active & self.learning
        cmd = super().propose(target, active & ~self.learning, steps)
        self.pending = None
        if not live.any():
            return cmd
        q, core, geom = self.geometry()
        contacts = self.tap.read(m)
        remaining = ((self.max_steps-steps).float()/self.max_steps).clamp(0, 1)
        obs = policy.pack_observation(geom, contacts, m, self.grip.credit, self.req['target_deg'], remaining)
        latent, logp, value = self.model.sample(obs, obs, deterministic=not self.training)
        retreat = self.grip.reached & live
        self.retract = torch.where(live, retreat, self.retract)
        self.action_phase = torch.where(live, torch.where(retreat, 8, torch.where(contacts['bilateral'], 7, 6)),
                                        self.action_phase)
        proposed = self.guide.command(target, q, live, latent, core, m['face_deg'], contacts, retreat=retreat)
        # Freeze every left target, including the preload present at the natural handoff.
        if not torch.equal(proposed[live][:, self.left_ids], self.hold_target[live][:, self.left_ids]):
            raise ValueError('Grip action changed frozen left support')
        self.learn_steps += live.long()
        self.pending = dict(obs=obs, critic=obs, latent=latent, logp=logp, value=value, valid=live,
                            before=geom, pre_contact=contacts['bilateral'].clone(),
                            before_angle=m['face_deg'].clone(), orbit=self.guide.orbit_rad.clone())
        return torch.where(live[:,None], proposed, cmd)

    @property
    def left_ids(self):
        return [i for i, n in enumerate(self.e.names) if n.startswith('left_') or n.endswith('_L')]

    def observe(self, active, post, safe, step):
        native = _observe(self, active, post, safe, step)
        native_held = native['success']
        native['success'] = torch.zeros_like(native_held)
        p = self.pending
        if p is None:
            return native
        m = self.obs
        c = self.tap.read(m)
        clean = p['valid'] & safe & ~native['unsafe']
        eligible = c['bilateral'] & clean
        sustained = self.grip.update(p['valid'], clean, p['pre_contact'], eligible, p['orbit'],
                                     p['before_angle'], m['face_deg'], self.req['target_deg'],
                                     m['contact_goal_reached'])
        self.grip.held |= clean & self.grip.reached & native_held
        native['success'] = clean & (self.grip.reached if self.req['target_deg']==2 else self.grip.held)
        self.counts['thumb_target_frames'] += (c['thumb'] & clean).long()
        self.counts['index_target_frames'] += (c['index'] & clean).long()
        self.counts['bilateral_frames'] += eligible.long()
        self.counts['bilateral_substeps'] += torch.where(eligible, c['opposed_substeps'], 0)
        self.counts['opposed_substeps'] += torch.where(clean, c['opposed_substeps'], 0)
        self.counts['same_layer_overlap_substeps'] += torch.where(clean, c['overlap'], 0)
        p.update(contacts=c, clean=clean, sustained=sustained, success=native['success'], native_held=native_held)
        return native

    def record_state(self, mask, step, target, event, reason=None, prefix=None):
        super().record_state(mask & ~self.learning, step, target, event, reason, prefix)
        # The presentation controller's private IK diagnostics are stale in the learned phase.
        lanes = (mask & self.learning).nonzero(as_tuple=True)[0].tolist()
        if not lanes:
            return
        q = self.e.robot.get_dof_positions()[:, self.e.ids]
        dq = self.e.robot.get_dof_velocities()[:, self.e.ids]
        core, bodies = self.e.core(), self.body_poses()
        for i in lanes:
            self.traces.write(json.dumps(dict(event=event, step=step, lane=i, seed=self.req['seeds'][i],
                phase=self.stage_name(i), command=target[i].tolist(), q=q[i].tolist(), dq=dq[i].tolist(),
                core=core[i].tolist(), joint_names=self.e.names, body_names=self.body_names,
                body_poses=bodies[i].tolist(), post_controller_started=True, learned_policy_active=True,
                learned_steps=int(self.learn_steps[i]), bilateral_credit_deg=float(self.grip.credit[i]),
                reason=reason if isinstance(reason,str) or reason is None else reason[i]),
                allow_nan=False)+'\n')

    def after_step(self, before_post, alive, unsafe_seen, reasons, steps):
        p = self.pending
        if p is None:
            return
        _, _, geom = self.geometry()
        valid = p['valid']
        c = p['contacts']
        reward = self.rewards.update(valid, unsafe_seen, p['success'], p['before'], geom, c,
                                     p['sustained'], self.req['target_deg'], p['latent'])
        done = valid & ~alive
        self.cumulative_reward += reward
        if self.training:
            item = {k:p[k] for k in ('obs','critic','latent','logp','value','valid')}
            item.update(done=done, reward=reward)
            self.data.append({k:v.detach().clone() for k,v in item.items()})
        self.write_evidence(p, done, unsafe_seen, reasons, steps, reward)
        self.pending = None

    def write_evidence(self, p, done, unsafe, reasons, steps, reward):
        c = p['contacts']
        keys = ('angle_before', 'angle', 'handoff_deg', 'pre_bilateral', 'supported', 'left_target_Ns',
                'other_target_Ns', 'wrong_Ns', 'clean', 'orbit_rad', 'native_goal', 'native_held',
                'retreat', 'credit_deg', 'reached', 'held', 'reward')
        fields = (p['before_angle'], self.obs['face_deg'], self.grip.handoff, p['pre_contact'], c['supported'],
                  c['left_target_Ns'], c['other_target_Ns'], c['wrong_Ns'], p['clean'], p['orbit'],
                  self.obs['contact_goal_reached'], p['native_held'], self.retract, p['sustained'],
                  self.grip.reached, self.grip.held, reward)
        lanes = p['valid'].nonzero(as_tuple=True)[0]
        values = torch.stack([f.double() for f in fields], 1)[lanes].tolist()
        substeps = self.tap.samples[lanes].tolist()
        sides = self.tap.side_masks[lanes].tolist()
        for i, nums, imp, side in zip(lanes.tolist(), values, substeps, sides, strict=True):
            self.trace.write(json.dumps(dict(step=self.monitor.last_step, lane=i, seed=self.req['seeds'][i],
                substeps_Ns=imp, side_masks=side, **dict(zip(keys, nums, strict=True))),
                separators=(',', ':'), allow_nan=False)+'\n')
        for i in (done | (p['valid'] & (steps % 60 == 0))).nonzero(as_tuple=True)[0].tolist():
            self.reward_log.write(json.dumps(dict(lane=i, seed=self.req['seeds'][i],
                learned_steps=int(self.learn_steps[i]), post_step=int(steps[i]), terminal=bool(done[i]),
                return_sum=float(self.cumulative_reward[i]), unsafe=bool(unsafe[i]), reason=reasons[i]),
                allow_nan=False)+'\n')

    def row(self, i):
        r = super().row(i)
        r.update({k:int(v[i]) for k,v in self.counts.items()})
        peak = float(self.grip.peak[i])
        r.update(learned_steps=int(self.learn_steps[i]), turn_started=peak>=2., peak_bilateral_drive_deg=peak,
                 bimanual_goal_reached=bool(self.grip.reached[i]), bimanual_turn_held=bool(self.grip.held[i]),
                 progress=min(1., peak/self.req['target_deg']))
        return r

    def finish_learning(self):
        learning = _finish(self)
        self.trace.close()
        learning.update(policy_schema=policy.SCHEMA, bimanual_contact_evidence_sha=recipe.sha(self.trace.name),
                        reward_fields=list(policy.REWARD_FIELDS), reward_totals=self.rewards.totals.tolist(),
                        inputs=list(policy.INPUT_FIELDS), actions=list(policy.ACTION_AXES),
                        material_side_bits=list(SIDE_BITS), target_deg=self.req['target_deg'], placo=self.guide.placo.receipt)
        return learning


def run(req, out):
    # Use the archive's 15-degree identifier only to load and verify the frozen R27 pickup.
    # All post-handoff observation, learning, evaluation and reporting receive the actual requested goal.
    def pickup_inputs(original):
        return recipe.inputs(dict(original, target_deg=15))
    with ExitStack() as resources:
        def phase(e, request, folder):
            return BimanualPhase(e, request, folder, resources)
        rollout = FunctionType(recipe.run.__code__, dict(recipe.run.__globals__, inputs=pickup_inputs, PostPhase=phase))
        rollout(req, out)


main = _specialize(recipe.main, [
    ('from natural_face90_recipe_long import fingerprint, worker_software, validate_request',
     'from natural_face90_bimanual import fingerprint, worker_software, validate_request'),
], dict(run=run, __doc__=__doc__))


class LaneEvidence:
    def __init__(self):
        self.step = None
        self.handoff = self.start = self.angle = self.last_credit = self.peak = self.total = 0.
        self.last_contact = self.was_drive = self.reached = self.held = False
        self.frames = self.both = self.overlap = self.opposed = self.any_overlap = self.thumb = self.index = 0

    def read(self, row, native, target):
        samples = row['substeps_Ns']
        if (len(samples) != 8 or any(len(s) != 3 or any(x < 0 or not math.isfinite(x) for x in s) for s in samples)):
            raise ValueError('Invalid per-substep bilateral evidence')
        t, i, left = map(sum, zip(*samples, strict=True))
        overlap = sum(all(x > 0 for x in s) for s in samples)
        sides = row['side_masks']
        if (len(sides)!=8 or any(len(pair)!=2 or any(type(x) is not int or not 0<=x<=15 for x in pair)
                                 for pair in sides)):
            raise ValueError('Invalid nZ material column mask')
        opposed_steps = 0
        for (thumb, index, hold), (t_side, i_side) in zip(samples,sides,strict=True):
            if (t_side and thumb<=IMPULSE/8) or (i_side and index<=IMPULSE/8):
                raise ValueError('Material contact on an unloaded finger')
            opposed_steps += int(opposed(t_side,i_side) and hold>IMPULSE/8)
        def near(a, b):
            return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)
        def signed(delta):
            return (delta + 180.) % 360. - 180.
        face = row['angle']
        before = row['angle_before']
        clean = bool(native['safe'] and native['structure_ok'] and native['allowed_layer_state']
                     and not native['wrong_contact'])
        if (not near(native['face_deg'], face) or not near(t+i+row['other_target_Ns'], native['target_impulse'])
                or not near(left,native['left_static_impulse']) or not near(row['left_target_Ns'], native['left_target_impulse'])
                or not near(row['wrong_Ns'],native['other_impulse']) or clean != bool(row['clean'])
                or bool(native['retract']) != bool(row['retreat'])
                or bool(native['success']) != bool(row['native_held'])
                or bool(native['contact_goal_reached']) != bool(row['native_goal'])):
            raise ValueError('Bimanual contact or layer evidence disagrees with original physical observer')
        supported = (left > IMPULSE and native['support'] and native['natural_ready'] and native['structure_ok']
                     and native['allowed_layer_state'] and not native['wrong_contact'])
        if bool(row['supported']) != supported:
            raise ValueError('Unconfirmed left support in grip evidence')
        right_pair = (supported and min(t, i)>IMPULSE and opposed_steps>0 and
                      max(row['left_target_Ns'], row['other_target_Ns'], row['wrong_Ns'])<=QUIET)
        qualified = clean and right_pair
        if self.step is None:
            self.handoff = before
        elif (self.step+1 != row['step'] or not near(self.angle, before)
              or self.last_contact != bool(row['pre_bilateral'])):
            raise ValueError('Missing or inconsistent predecessor for bilateral drive')
        if not near(row['handoff_deg'], self.handoff):
            raise ValueError('Changed handoff orientation in bilateral evidence')
        drive = qualified and row['pre_bilateral'] and row['orbit_rad']>0
        if drive and not self.was_drive:
            self.start = before
        raw = max(0., min(signed(face-self.start), signed(face-self.handoff))) if drive else 0.
        sustained = min(raw,self.last_credit) if drive and self.was_drive else 0.
        self.reached |= bool(drive and sustained>=target and face>=target and row['native_goal'])
        self.held |= bool(clean and self.reached and row['native_held'])
        if (not near(sustained, row['credit_deg']) or self.reached != bool(row['reached'])
                or self.held != bool(row['held'])):
            raise ValueError('Unverified object rotation or held-goal milestone')
        self.step, self.angle, self.last_credit = row['step'], face, raw
        self.last_contact, self.was_drive = right_pair, drive
        self.peak = max(self.peak,sustained)
        self.frames += 1
        self.thumb += int(clean and t>IMPULSE)
        self.index += int(clean and i>IMPULSE)
        self.both += int(qualified)
        self.overlap += opposed_steps if qualified else 0
        self.opposed += opposed_steps if clean else 0
        self.any_overlap += overlap if clean else 0
        self.total += row['reward']

    def check(self, row, target):
        expected = (self.frames, self.thumb, self.index, self.both, self.overlap, self.opposed,
                    self.any_overlap, self.reached, self.held)
        actual = tuple(row[k] for k in ('learned_steps','thumb_target_frames','index_target_frames',
                                        'bilateral_frames','bilateral_substeps','opposed_substeps',
                                        'same_layer_overlap_substeps','bimanual_goal_reached',
                                        'bimanual_turn_held'))
        success = self.reached if target==2 else self.held
        if (actual != expected or not math.isclose(row['peak_bilateral_drive_deg'],self.peak,abs_tol=1e-8)
                or bool(row['success']) != bool(success and not row['unsafe'])):
            raise ValueError('Episode metrics disagree with verified bilateral evidence')


def verify_wave(report, req):
    from natural_face90_recipe_long import verify_result
    rows, learning = verify_result(report, req)
    trace = report.parent/'bimanual_contacts.jsonl'
    if (learning['bimanual_contact_evidence_sha'] != recipe.sha(trace) or learning['policy_schema'] != policy.SCHEMA
            or learning['inputs']!=list(policy.INPUT_FIELDS) or learning['actions']!=list(policy.ACTION_AXES)
            or learning['target_deg']!=req['target_deg'] or learning['placo']!=req['placo']
            or learning['reward_fields']!=list(policy.REWARD_FIELDS) or learning['material_side_bits']!=list(SIDE_BITS)):
        raise ValueError('Invalid bimanual wave evidence or policy contract')
    lanes = [LaneEvidence() for _ in rows]
    with trace.open() as stream, (report.parent/'post_measurements.jsonl').open() as native:
        for line in stream:
            r = json.loads(line)
            key = r['step'],r['lane']
            if not 0<=r['lane']<len(lanes) or r['seed']!=req['seeds'][r['lane']]:
                raise ValueError('Unexpected bimanual evidence lane or seed')
            for source in native:
                m = json.loads(source)
                at = m['step'],m['lane']
                if at >= key:
                    break
            else:
                raise ValueError('Missing independent posture/layer/contact evidence')
            if at != key or m['seed']!=r['seed']:
                raise ValueError('Unexpected physical evidence order')
            lanes[r['lane']].read(r,m,req['target_deg'])
    if len(learning['reward_totals']) != len(lanes):
        raise ValueError('Missing PPO reward totals')
    for ev,row,parts in zip(lanes, rows, learning['reward_totals'], strict=True):
        ev.check(row, req['target_deg'])
        if (len(parts)!=len(policy.REWARD_FIELDS) or
                not math.isclose(ev.total, sum(parts), rel_tol=1e-5, abs_tol=5e-5)):
            raise ValueError('Incorrect per-lane reward totals')
    if sum(lane.frames for lane in lanes) != learning['learned_transitions']:
        raise ValueError('PPO transition count differs from measured frames')
    return rows, learning


if __name__ == '__main__':
    main()
