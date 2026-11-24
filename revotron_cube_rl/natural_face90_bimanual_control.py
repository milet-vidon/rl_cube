"""Closed-loop R27 opposite-hand grip and nZ orbit. All outputs are actuator targets."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from contact_hold_cuda_teacher_math import Chain, mv, quat_matrix, rotvec
from joint_safety import project_target
from kinematics import RobotKinematics
from natural_face90_demo_control import FLEXION, THUMB_BASE
from natural_face90_placo import MAX_JOINT_STEP
from natural_face90_placo_client import PlacoClient
from natural_face90_v5_controller import HALF_EDGE, _align
from natural_face90_v5_metrics import unit


ACTION_AXES = ('cube_x', 'cube_y', 'cube_z', 'thumb_flex', 'index_flex', 'thumb_opposition', 'face_orbit')
ANCHORS = Path(__file__).with_name('opposition_grasp_v20.json')
ANCHOR_SHA = '77f69b76900df2845cd5da7d465f1a8fc3641ee929533a14a2458bd8c81af29a'
LAYER_Z = -.030  # Center plane of the outer 30 mm row in the 90 mm cube.


def bounded(vector, max_norm):
    return vector * (max_norm / vector.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp_max(1.)


def flank_targets(left_index, left_thumb, face_deg):
    """Opposite flank surfaces, ordered from the measured support-hand finger pair."""
    span = left_index[:, :2] - left_thumb[:, :2]
    dim = span.abs().argmax(1, keepdim=True)
    direction = torch.zeros_like(span).scatter(1, dim, torch.where(span.gather(1, dim) >= 0., 1., -1.).to(span))
    # Mirror across the middle layer into the nZ slab. Use the left fingers'
    # tangential offsets only, to avoid copying their static-layer height.
    tangent = torch.stack((-direction[:, 1], direction[:, 0]), 1)
    phi = -torch.deg2rad(face_deg.to(span))[:, None]
    d = direction * phi.cos() + tangent * phi.sin()
    t = tangent * phi.cos() - direction * phi.sin()
    def on_flank(pad, sign):
        along = (pad[:, :2] * tangent).sum(1, keepdim=True).clamp(-.021, .021)
        xy = sign * HALF_EDGE * d + along * t
        return torch.cat((xy, torch.full_like(along, LAYER_Z)), 1)
    zero = torch.zeros_like(phi)
    return on_flank(left_index, 1.), on_flank(left_thumb, -1.), torch.cat((d, zero), 1)


class BimanualGuide:
    """Maintain the left pickup preload and solve only the right arm and two digits."""
    def __init__(self, controller):
        self.c = controller
        self.e = e = controller.e
        self.arm = controller.rp.ix
        names = list(e.names)
        if len(names) != 56 or abs(e.dt*e.dec - 1./30) > 1e-8:
            raise ValueError('Expected the calibrated 56-joint, 30 Hz scene')
        archive = ANCHORS.read_bytes()
        if hashlib.sha256(archive).hexdigest() != ANCHOR_SHA:
            raise ValueError('Support-hand contact anchors changed')
        anchors = json.loads(archive)['contact_anchors']
        k = RobotKinematics()
        def anchor(finger, right=False):
            a = anchors[finger]
            point = np.array(a['point_local'])
            link = a['link']
            if right:
                # The two URDF hands are mirrored in link-local Y, as in
                # opposed_grasp_plan. This is a geometry estimate, never a sensor.
                link = link.replace('left_', 'right_')
                point *= [1., -1., 1.]
            return Chain(k, link, names, e.home.device, point=point)
        self.left_index, self.left_thumb = anchor('index'), anchor('thumb')
        self.right_thumb = anchor('thumb', right=True)
        self.right_index = controller.tip
        self.hand_obs = [names.index('right_' + joint + '_joint') for joint in
                         ('thumb_MCP', 'index_MCP', 'thumb_CMR')]
        # One synergy per column. Their pad-space Jacobian is recomputed at
        # the measured joint angles before each step.
        self.grip = e.home.new_zeros(56, 3, dtype=torch.float64)
        for j, (finger, gains) in enumerate((('thumb', FLEXION), ('index', FLEXION), ('thumb', THUMB_BASE))):
            for joint, gain in gains.items():
                self.grip[names.index(f'right_{finger}_{joint}_joint'), j] = gain
        self.finger_ids = self.grip.any(1).nonzero(as_tuple=True)[0]
        self.orbit_rad = e.home.new_zeros(len(e.home), dtype=torch.float64)
        self.last_single = torch.zeros_like(self.orbit_rad, dtype=torch.int64)
        self.pivot_local = e.home.new_zeros((len(e.home), 3), dtype=torch.float64)
        self.placo = PlacoClient(names, e.dt * e.dec)

    def _geometry(self, q, core, face_deg):
        q = q.double()
        root = self.e.robot.get_root_transforms().double()
        root_R, core_R = quat_matrix(unit(root[:, 3:])), quat_matrix(unit(core[:, 3:].double()))
        base_from_core = root_R.transpose(1, 2) @ core_R
        core_base = mv(root_R.transpose(1, 2), core[:, :3].double() - root[:, :3])
        to_local = lambda p: mv(base_from_core.transpose(1, 2), p - core_base)
        index, _, io, ia = self.right_index.fk(q[:, self.right_index.ix])
        thumb, _, to, ta = self.right_thumb.fk(q[:, self.right_thumb.ix])
        left_i = self.left_index.fk(q[:, self.left_index.ix])[0]
        left_t = self.left_thumb.fk(q[:, self.left_thumb.ix])[0]
        right_R = self.c.rp.fk(q[:, self.arm])[1]
        left_R = self.c.lp.fk(q[:, self.c.lp.ix])[1]
        gi, gt, flank = flank_targets(to_local(left_i), to_local(left_t), face_deg)
        geom = dict(index_local=to_local(index), thumb_local=to_local(thumb),
                    goal_index_local=gi, goal_thumb_local=gt,
                    palm_opposition_cos=-(right_R[:, :, 0]*left_R[:, :, 0]).sum(1).clamp(-1., 1.),
                    hand_q=q[:, self.hand_obs])
        return geom, (base_from_core, core_base, right_R, left_R, index, thumb, io, ia, to, ta, flank)

    def geometry(self, q, core_world, face_deg):
        return self._geometry(q, core_world, face_deg)[0]

    def command(self, target, q, active, latent, core, face_deg, contacts, *, retreat=None):
        if latent.shape != (len(q), len(ACTION_AXES)) or not torch.isfinite(latent).all():
            raise ValueError('Expected seven finite bimanual actions per lane')
        q = q.double()
        a = latent.double().tanh()
        if retreat is None:
            retreat = torch.zeros_like(active)
        active = active.bool()
        withdrawing = active & retreat.bool()
        supported = active & contacts['left_static'].bool() & ~withdrawing
        drive = supported & contacts['bilateral'].bool()
        move = supported | withdrawing
        single = torch.where(supported & (contacts['index'] ^ contacts['thumb']),
                             contacts['index'].long() - contacts['thumb'].long(), 0)
        fresh_single = (single != 0) & (self.last_single != single)
        self.last_single = single
        # Sampled orbit is a signed measured-axis request; no wrist sweep before
        # native left/static + thumb/nZ + index/nZ contact is confirmed.
        self.orbit_rad = torch.where(drive, (.35 + .50*a[:, 6])*math.pi/180., 0.)
        if not move.any():
            return target.clone()
        g, kin = self._geometry(q, core, face_deg)
        B, center, right_R, left_R, index, thumb, io, ia, to, ta, flank = kin
        hit_i = contacts['index'].bool()
        hit_t = contacts['thumb'].bool()
        touched = hit_i | hit_t
        ri, rt = g['index_local'], g['thumb_local']
        gi, gt = g['goal_index_local'], g['goal_thumb_local']
        # Approach from the free nZ half-space with the pads slightly wider
        # than the face. Once beside the outer slab, converge onto opposite flanks.
        xy_error = torch.maximum((gi[:, :2]-ri[:, :2]).norm(dim=1), (gt[:, :2]-rt[:, :2]).norm(dim=1))
        setback = (xy_error-.014).clamp(0., .030)
        opening = ((-.044 - torch.minimum(ri[:, 2], rt[:, 2])) * .4).clamp(0., .008)
        zi = torch.stack((setback*0., setback*0., setback), 1)
        clearance = opening - .001  # Up to 1 mm nominal preload at the outer flanks.
        gi = gi - zi + clearance[:, None]*flank
        gt = gt - zi - clearance[:, None]*flank
        # With a single contact, lock that measured pad in cube coordinates.
        # The free pad still tracks its own material point at the layer center.
        pivot = torch.where(hit_i[:, None], ri, rt)
        self.pivot_local = torch.where(fresh_single[:, None], pivot, self.pivot_local)
        gi = torch.where((single == 1)[:, None], self.pivot_local, gi)
        gt = torch.where((single == -1)[:, None], self.pivot_local, gt)
        # Anywhere well inside the side face of an outer-row piece is useful.
        # If the first pad arrived at a corner, let the free digit reach the
        # closest tangential position, instead of forcing a second wrist roll.
        tangent = torch.stack((-flank[:, 1], flank[:, 0], flank[:, 2]), 1)
        entry = torch.zeros_like(flank)
        entry[:, 2] = -.036
        free_i = .044*flank + entry + (ri*tangent).sum(1, keepdim=True).clamp(-.024, .024)*tangent
        free_t = -.044*flank + entry + (rt*tangent).sum(1, keepdim=True).clamp(-.024, .024)*tangent
        gi = torch.where((single == -1)[:, None], free_i, gi)
        gt = torch.where((single == 1)[:, None], free_t, gt)
        di, dt = mv(B, gi-ri), mv(B, gt-rt)

        # Align the pad span to the opposite flanks first. Use the remaining
        # orientation freedom around the span to face the supporting palm.
        span = index-thumb
        desired_span = mv(B, gi-gt)
        facing = bounded(_align(span, desired_span), .010)
        axis = torch.nn.functional.normalize(span, dim=1)
        palm_axis, goal_axis = right_R[:, :, 0], -left_R[:, :, 0]
        flat = palm_axis - (palm_axis*axis).sum(1, keepdim=True)*axis
        desired_flat = goal_axis - (goal_axis*axis).sum(1, keepdim=True)*axis
        roll = torch.atan2((axis*torch.cross(flat, desired_flat, dim=1)).sum(1), (flat*desired_flat).sum(1))
        facing += axis*roll.clamp(-.007, .007)[:, None]
        nz = -B[:, :, 2]
        # With a single contact, swivel the free pad toward its opposing
        # flank about the loaded pad. The translation below holds that pivot.
        swivel = bounded(_align(span, desired_span), .006)
        rv = torch.where(touched[:, None], swivel, facing)
        rv = torch.where((drive | withdrawing)[:, None], 0., rv)
        rotation = rotvec(rv + self.orbit_rad[:, None]*nz)
        desired_R = rotation @ right_R

        # Solving the differential finger span leaves the arm to translate
        # the grip center. Hold a loaded digit while the other seeks contact.
        ji = torch.cross(ia, index[:, None]-io, dim=2).transpose(1, 2) @ self.grip[self.right_index.ix]
        jt = torch.cross(ta, thumb[:, None]-to, dim=2).transpose(1, 2) @ self.grip[self.right_thumb.ix]
        free = torch.stack((~hit_t, ~hit_i, ~hit_t), 1)
        J = (ji-jt)*free[:, None, :]
        # The wrist handles span direction. The fingers supply the opening
        # width. Decoupling these avoids extending the thumb into a hard stop
        # while both the wrist and digit servo try to correct the same tilt.
        aperture_jac = (axis[:, :, None]*J).sum(1)
        width_error = desired_span.norm(dim=1)-span.norm(dim=1)
        correction = aperture_jac*(width_error/(aperture_jac.square().sum(1)+.014**2))[:, None]
        finger = bounded(correction, .018) + a[:, 3:6]*torch.where(free, .016, .002).to(q)
        finger = torch.where(withdrawing[:, None], torch.full_like(finger, -.015), finger)
        finger_delta = finger @ self.grip[self.finger_ids].T
        old_fingers = target[:, self.finger_ids].double()
        measured = q[:, self.finger_ids]
        load_budget = (old_fingers-measured).abs().clamp_min(.09)
        finger_target = (old_fingers + finger_delta).clamp(measured-load_budget, measured+load_budget)
        finger_limits = self.e.limits[:, self.finger_ids].double()
        finger_target = finger_target.clamp(finger_limits[..., 0], finger_limits[..., 1])
        finger_cap = torch.minimum(self.e.step, self.e.vel*self.e.dt*self.e.dec)[:, self.finger_ids]
        finger_delta = (finger_target - old_fingers).clamp(-finger_cap, finger_cap)
        actual = torch.zeros_like(q)
        actual[:, self.finger_ids] = finger_delta
        index_motion = mv(torch.cross(ia, index[:, None]-io, dim=2).transpose(1,2), actual[:, self.right_index.ix])
        thumb_motion = mv(torch.cross(ta, thumb[:, None]-to, dim=2).transpose(1,2), actual[:, self.right_thumb.ix])

        # Use the first loaded pad as the pivot; when both are loaded, orbit
        # the rigid pinch about the measured cube center instead.
        weight_i = .5 + .5*(hit_i.double() - hit_t.double())
        wi = weight_i[:, None]
        anchor = wi*index + (1.-wi)*thumb
        translation = bounded(wi*di+(1.-wi)*dt, .0010)
        translation = torch.where(drive[:, None], 0., translation)
        translation += mv(B, bounded(a[:, :3], 1.)*torch.where(touched, .00030, .0013)[:, None])
        tcp_delta = translation + mv(rotation, index-anchor) - (index-anchor)
        tcp_delta -= wi*index_motion + (1.-wi)*thumb_motion
        orbit_delta = mv(rotation, index-center) - (index-center)
        tcp_delta = torch.where(drive[:, None], orbit_delta + translation, tcp_delta)
        tcp_delta = torch.where(withdrawing[:, None], .0015*nz, tcp_delta)
        tcp_delta = bounded(tcp_delta, .0025)

        arm_q = q[:, self.arm]
        batch = self.placo.solve_batch(*(v[move].detach().cpu().numpy() for v in
                                         (q, tcp_delta, desired_R, withdrawing)))
        delta = torch.zeros_like(arm_q)
        delta[move] = torch.as_tensor(batch, dtype=q.dtype, device=q.device) - arm_q[move]
        delta = self.c.clearance.steer(arm_q, delta, index, io, ia).clamp(-MAX_JOINT_STEP, MAX_JOINT_STEP)
        old = target[:, self.arm].double()
        lag = (old-arm_q).abs().clamp_min(.040)
        arm = (old + delta).clamp(arm_q-lag, arm_q+lag)
        command = target.clone()
        command[:, self.arm] = arm.to(target)
        command[:, self.finger_ids] = (old_fingers+finger_delta).to(target)
        safe, _ = project_target(target, command, self.e.limits, self.e.step, self.e.vel, self.e.dt*self.e.dec)
        safe[:, self.arm] = self.c.clearance.project(old, safe[:, self.arm].double()).to(target)
        return torch.where(move[:, None], safe, target)
