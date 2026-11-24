"""Control the calibrated right index TCP with a bounded displacement before arm IK."""
import math

import torch

from contact_hold_cuda_teacher_math import mv, quat_matrix, rotvec
from natural_face90_hover_c_policy import _MeasuredArmCommand
from natural_face90_pad_preshape_policy import PreshapeGuide
from natural_face90_placo_client import PlacoClient
from natural_face90_v5_controller import HALF_EDGE, _align
from natural_face90_v5_metrics import rotate, unit


ACTION_AXES = ('radial_out', 'positive_tangent', 'inward')
MAX_RESIDUAL_M = .0005
MAX_APPROACH_STEP_M = .00075
MAX_DEPTH_M = .0005


def material_axes(radial, face_deg):
    """Core XY unit directions at the measured rotating nZ material marker."""
    angle = torch.deg2rad(face_deg.to(radial.dtype))[:, None]
    tangent = torch.stack((radial[:, 1], -radial[:, 0]), dim=1)
    outward = angle.cos() * radial + angle.sin() * tangent
    return outward, torch.stack((outward[:, 1], -outward[:, 0]), dim=1)


class _PlacoCommand(_MeasuredArmCommand):
    """Replace only the joint solver beneath the original path, preshape and evidence hooks."""
    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        c, arm = self.c, self.arm
        q = q.double()
        measured = q[:, arm]
        # PhysX transforms store xyzw. Target and measured pad contain the same
        # lane origin; subtract them before rotating the error into the robot base.
        root = c.e.robot.get_root_transforms().double()
        world_R_base = quat_matrix(unit(root[:, 3:7]))
        base_R_world = world_R_base.transpose(1, 2)
        _, palm_base, _, _ = c.rp.fk(measured)
        palm_world = world_R_base @ palm_base
        core_q = unit(core[:, 3:7])
        inward = rotate(core_q, q.new_tensor([0., 0., 1.]))
        self._plan(fresh, target, palm_world, inward, tip)
        actual_axis = mv(palm_world.transpose(1, 2), inward)
        self.index_lead_m = ((self.vectors * actual_axis[:, None]).sum(-1) - self.radii).amin(1)
        oriented = ((self.axis * actual_axis).sum(-1) > .985) & (self.index_lead_m >= .002)
        goal = self._path(fresh, learned, c.radial, local_tip, oriented, obs)
        desired_world = rotvec(_align(mv(self.orientation, self.axis), inward)) @ self.orientation
        loaded_recovery = self.recovering & obs['right_contact'].bool()
        desired = base_R_world @ desired_world
        error = mv(base_R_world, core[:, :3] + rotate(core_q, goal) - tip)

        # Retain hover/approach counters for the paired comparison. Placo
        # itself starts at the measured state in every active phase.
        self.filter_residual(torch.zeros_like(measured), fresh, learned, obs)
        self.select_anchor(measured, target[:, arm])
        nominal = measured.clone()
        if learned.any():
            batch = self.placo.solve_batch(*(v[learned].detach().cpu().numpy()
                                             for v in (q, error, desired, loaded_recovery)))
            nominal[learned] = torch.as_tensor(batch, dtype=q.dtype, device=q.device)
        raw_delta = nominal - measured
        tip_base, _, origins, axes = c.tip.fk(q[:, c.tip.ix])
        delta = c.clearance.steer(measured, raw_delta, tip_base, origins, axes)
        cap = torch.minimum(c.e.step, c.e.vel / 30).expand_as(target)[:, arm].double().clamp_max(.012)
        lim = c.e.limits.expand(len(q), -1, -1)[:, arm].double()
        scale = (cap / delta.abs().clamp_min(1e-12)).amin(1, keepdim=True).clamp_max(1.)
        nominal = (measured + scale * delta).clamp(lim[..., 0], lim[..., 1])
        nominal = c.clearance.project(measured, nominal)
        self.reference = torch.where(learned[:, None], nominal, self.reference)
        old = target[:, arm].double()
        result = handoff.clone()
        result[:, arm] = c.clearance.project(old, old + (nominal - old).clamp(-cap, cap)).to(result.dtype)
        self.control_evidence().update(
            pre_placo_solved=learned, pre_ik_anchor_error_max_rad=torch.zeros_like(self.index_lead_m),
            pre_placo_step_max_rad=raw_delta.abs().amax(dim=1),
            pre_placo_fk_pad_error_m=(root[:, :3] + mv(world_R_base, tip_base) - tip).norm(dim=1))
        return torch.where(learned[:, None], result, target)


class CartesianGuide(PreshapeGuide, _PlacoCommand):
    def __init__(self, controller, geometry):
        super().__init__(controller, geometry)
        self.placo = PlacoClient(controller.names, 1. / controller.e.cfg['control_hz'])

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        if latent.shape != (len(q), 3) or not torch.isfinite(latent).all():
            raise ValueError('Expected three finite Cartesian actions per lane')
        self.cartesian_latent = latent
        # Keep the old seven-dimensional residual disabled in all phases.
        return super().command(target, q, handoff, fresh, learned, torch.zeros_like(self.reference),
                               core, tip, local_tip, obs)

    def _path(self, fresh, active, radial, local_tip, orientation_ok, obs):
        nominal = super()._path(fresh, active, radial, local_tip, orientation_ok, obs)
        ev = self.control_evidence()
        ready = (active & orientation_ok & ev['pre_support_ok'] & ~self.recovering
                 & ~obs['contact_goal_reached'].bool())
        unloaded = (ready & ~ev['pre_contact'] & ~ev['pre_index_contact'] & ~ev['pre_other_digit_contact']
                    & (obs['right_total_impulse'] == 0.) & (ev['pre_capture_remaining'] == 0))
        approach = (unloaded & self.aligned & (ev['pre_normal_gap_m'] > 0.)
                    & (nominal[:, 2] > local_tip[:, 2]))
        hover = unloaded & ~self.aligned
        drive = (ready & self.drive_contact & ev['pre_contact'] & ev['pre_index_contact']
                 & ev['pre_index_near'] & ~ev['pre_other_digit_contact'])
        mode = hover.long() + 2 * approach.long() + 3 * drive.long()

        action = self.cartesian_latent.double().tanh()
        delta = MAX_RESIDUAL_M * action / action.norm(dim=1, keepdim=True).clamp_min(1.)
        dr, dt, dz = delta.unbind(dim=1)
        face = obs['face_deg'].double()
        outward, tangent = material_axes(radial, face)
        # Preserve the signed sweep window. In the rotating frame a point
        # with coordinates (r,t) has a positive nZ lead of atan2(t,r).
        r = (nominal[:, :2] * outward).sum(dim=1)
        t = (nominal[:, :2] * tangent).sum(dim=1)
        angle_cap = (self.c.deg + 3. - face).clamp(min=0., max=16.)
        slope = torch.deg2rad(angle_cap).tan()
        # With a terminal face overshoot the nominal guide may already
        # point behind the face. In that case leave its sweep path alone.
        in_window = (t >= -1e-12) & (t <= r * slope + 1e-12)
        if_drive = drive & in_window
        xy_enabled = hover | approach | if_drive
        dr = torch.where(xy_enabled, dr, 0.)
        # Two constraints clip the tangent component. Shrink a negative
        # radial correction too if it would itself violate the lead cap.
        drive_dr_min = (t - r * slope) / slope.clamp_min(1e-12)
        dr = torch.where(if_drive & (slope > 0.), torch.maximum(dr, drive_dr_min.clamp_max(0.)), dr)
        max_extra_t = ((r + dr) * slope - t).clamp_min(0.)
        drive_dt = torch.minimum(torch.maximum(dt, -t.clamp_min(0.)), max_extra_t)
        dt = torch.where(hover | approach, dt, torch.where(if_drive, drive_dt, 0.))
        xy = dr[:, None] * outward + dt[:, None] * tangent

        # The learned displacement can slow the unloaded approach. Any
        # extra inward travel must fit the original approach step and face
        # depth caps; on a loaded drive it may only relieve pressure.
        headroom = (torch.minimum(local_tip[:, 2] + MAX_APPROACH_STEP_M,
                                 torch.full_like(dz, -HALF_EDGE + MAX_DEPTH_M)) - nominal[:, 2]).clamp_min(0.)
        dz = torch.where(approach, torch.minimum(dz, headroom), torch.where(drive, dz.clamp_max(0.), 0.))
        applied = torch.cat((xy, dz[:, None]), dim=1)
        self.goal_local = nominal + applied
        ev['pre_cartesian_mode'] = mode
        ev['pre_cartesian_target_deg'] = torch.full_like(face, self.c.deg)
        for i, axis in enumerate('xyz'):
            ev['pre_cartesian_latent_' + str(i)] = self.cartesian_latent[:, i]
            ev['pre_cartesian_nominal_' + axis + '_m'] = nominal[:, i]
            ev['pre_cartesian_delta_' + axis + '_m'] = applied[:, i]
            ev['pre_cartesian_pad_' + axis + '_m'] = local_tip[:, i]
        return self.goal_local


def verify_frame(row, target_deg=None):
    """Verify a recorded control target and its permitted Cartesian envelope."""
    mode = row['pre_cartesian_mode']
    nominal, applied, local, goal = ([row[prefix + axis + '_m'] for axis in 'xyz'] for prefix in
                                     ('pre_cartesian_nominal_', 'pre_cartesian_delta_', 'pre_cartesian_pad_', 'requested_'))
    action = [row['pre_cartesian_latent_' + str(i)] for i in range(3)]
    face = row['pre_material_face_deg']
    target = row['pre_cartesian_target_deg']
    marker = (row['pre_material_marker_x_m'], row['pre_material_marker_y_m'])
    radius = math.hypot(*marker)
    if (not all(map(math.isfinite, [*nominal, *applied, *local, *goal, *action, face, target, radius]))
            or not math.isclose(radius, .030, rel_tol=0., abs_tol=1e-8)
            or (target_deg is not None and target != target_deg)):
        raise ValueError('Nonfinite TCP action evidence')
    ready = (row['pre_orientation_ok'] and row['pre_support_ok'] and not row['pre_recovering']
             and not row['pre_approach_goal_reached'])
    unloaded = (ready and not row['pre_approach_any_touch'] and not row['pre_capture_remaining'])
    approach = (unloaded and row['pre_aligned'] and row['pre_normal_gap_m'] > 0 and nominal[2] > local[2])
    drive = (ready and row['drive_command'] and row['pre_contact'] and row['pre_index_contact']
             and row['pre_index_near'] and not row['pre_other_digit_contact'])
    expected_mode = 3 if drive else 2 if approach else 1 if unloaded and not row['pre_aligned'] else 0
    if (not row['pre_placo_solved'] or row['pre_ik_anchor_error_max_rad'] != 0.
            or not 0. <= row['pre_placo_step_max_rad'] <= .0120001
            or not math.isfinite(row['pre_placo_fk_pad_error_m']) or row['pre_placo_fk_pad_error_m'] < 0.):
        raise ValueError('Missing measured Placo solve or invalid joint step')
    if (mode != expected_mode or row['pre_arm_residual_baseline_max_rad'] != 0.
            or row['pre_arm_residual_applied_max_rad'] != 0.
            or math.dist(goal, [a + b for a,b in zip(nominal, applied)]) > 1e-10):
        raise ValueError('TCP mode, joint residual or final IK target disagrees with readback')
    if (math.hypot(*applied) > MAX_RESIDUAL_M + 1e-10 or (mode == 0 and any(applied))
            or (mode == 1 and applied[2] != 0.) or (mode == 3 and applied[2] > 0.)
            or (mode == 2 and (goal[2] - local[2] > MAX_APPROACH_STEP_M + 1e-10
                               or goal[2] > -HALF_EDGE + MAX_DEPTH_M + 1e-10))):
        raise ValueError('TCP correction violates its physical phase envelope')
    # Reconstruct the action directly from the logged policy sample in the
    # moving marker frame; this also detects a sign or axis swap before IK.
    raw = [math.tanh(a) for a in action]
    scale = MAX_RESIDUAL_M / max(1., math.hypot(*raw))
    dr, dt, dz = [v * scale for v in raw]
    rx, ry = (v / radius for v in marker)
    if mode == 0:
        dr, dt, dz = 0., 0., 0.
    elif mode == 1:
        dz = 0.
    elif mode == 2:
        ceiling = min(local[2] + MAX_APPROACH_STEP_M, -HALF_EDGE + MAX_DEPTH_M)
        dz = min(dz, max(0., ceiling - nominal[2]))
    else:
        dz = min(dz, 0.)
        r = nominal[0] * rx + nominal[1] * ry
        t = nominal[0] * ry - nominal[1] * rx
        slope = math.tan(math.radians(max(0., min(16., target + 3. - face))))
        if -1e-12 <= t <= r * slope + 1e-12:
            if slope > 0.:
                dr = max(dr, min(0., (t - r * slope) / max(slope, 1e-12)))
            dt = max(-max(0., t), min(dt, max(0., (r + dr) * slope - t)))
        else:
            dr, dt = 0., 0.
    expected = (rx * dr + ry * dt, ry * dr - rx * dt, dz)
    if math.dist(applied, expected) > 1e-9:
        raise ValueError('Applied TCP residual differs from the recorded material-frame action')
