"""Operator one-tick commands for a single physical bimanual cube scene."""
import json
import math

import numpy as np
import torch

from contact_hold_cuda_teacher_math import Chain, mv, quat_matrix, rotvec
from joint_safety import project_target
from kinematics import RobotKinematics
from natural_face90_placo import MAX_JOINT_STEP, PAD, PALM, RIGHT
from natural_face90_placo_client import PlacoClient
from natural_face90_v5_clearance import ForearmClearance


MOVE_M = .001
ROTATE_RAD = math.radians(.5)
FINGER_RAD = math.radians(1.)
ACTION_KEYS = ('move', 'rotate', 'thumb', 'index', 'opposition', 'orbit')
FLEXION = {'MCP': .8, 'PIP': 1., 'DIP': .7}
THUMB_BASE = {'CMP': 1., 'CMR': .65}


def _input(action, key, shape):
    value = np.asarray(action.get(key, np.zeros(shape)))
    if (value.shape != shape or value.dtype.kind not in 'iuf'
            or not np.isfinite(value).all() or np.any(np.abs(value) > 1.)):
        raise ValueError(f'Invalid manual input: {key}')
    return value.astype(np.float64, copy=False)


def face_orbit(core_world, base_world, tip_base, radians):
    """A rigid displacement of the grasp about the measured Core nZ line, in robot coordinates."""
    world_to_base = quat_matrix(base_world[:, 3:]).transpose(1, 2)
    center = mv(world_to_base, core_world[:, :3] - base_world[:, :3])
    axis = mv(world_to_base, -quat_matrix(core_world[:, 3:])[:, :, 2])
    rotation = rotvec(axis * radians)
    radius = tip_base - center
    return mv(rotation, radius) - radius, rotation


class ManualControl:
    """Apply input to the LAST EXECUTED target, beginning at the grasp handoff.

    Translation and rotation are expressed along robot-base axes. Wrist rotation
    pivots about the measured right palm origin. Zero input holds the prior
    actuator target, including its grasp preload; it does not chase measured q.
    """
    def __init__(self, env):
        if env.n != 1 or len(env.names) != 56 or env.cfg['control_hz'] != 30:
            raise ValueError('Expected one 56-joint scene at 30 Hz')
        self.env = env
        self.dt = env.dt * env.dec
        names = env.names
        model = RobotKinematics()
        pad = json.loads(PAD.read_bytes())
        self.palm = Chain(model, PALM, names, env.home.device)
        self.tip = Chain(model, pad['source_link'], names, env.home.device, point=np.array(pad['point_local']))
        self.arm = [names.index(n) for n in RIGHT]
        if self.palm.ix != self.arm or self.tip.ix[:7] != self.arm:
            raise ValueError('Right wrist/pad chains disagree with Placo joints')
        self.clearance = ForearmClearance(model, names, env.home.device)
        self.fingers = {}
        for channel, parts in (('thumb', FLEXION), ('index', FLEXION), ('opposition', THUMB_BASE)):
            finger = 'thumb' if channel == 'opposition' else channel
            self.fingers[channel] = [(names.index(f'right_{finger}_{joint}_joint'), gain)
                                     for joint, gain in parts.items()]
        self.placo = PlacoClient(names, self.dt)

    def step(self, target, action):
        if (not isinstance(action, dict) or set(action).difference(ACTION_KEYS)
                or target.shape != (1, 56) or not target.is_floating_point()):
            raise ValueError('Expected one bounded manual action and the previous 56-joint command')
        move = _input(action, 'move', (3,))
        rotate = _input(action, 'rotate', (3,))
        orbit = float(_input(action, 'orbit', ()))
        if orbit and (move.any() or rotate.any()):
            raise ValueError('Select a wrist move or a face-axis orbit for each tick')
        wrist_active = bool(orbit or move.any() or rotate.any())
        finger = {k:float(_input(action, k, ())) for k in self.fingers}
        requested = target.clone()
        if wrist_active:
            env = self.env
            q = env.robot.get_dof_positions()[:, env.ids].double()
            measured_arm = q[:, self.arm]
            palm, palm_R, _, _ = self.palm.fk(measured_arm)
            tip, _, _, _ = self.tip.fk(q[:, self.tip.ix])
            if orbit:
                tip_delta, rotation = face_orbit(env.views['Core'].get_transforms().double(),
                                                 env.robot.get_root_transforms().double(), tip,
                                                 orbit * ROTATE_RAD)
            else:
                translation = q.new_tensor(move * (MOVE_M / max(1., np.linalg.norm(move))))
                vector = q.new_tensor(rotate * (ROTATE_RAD / max(1., np.linalg.norm(rotate))))
                rotation = rotvec(vector.unsqueeze(0))
                offset = tip - palm
                tip_delta = translation + mv(rotation, offset) - offset
            desired_R = rotation @ palm_R
            solved = self.placo.solve_batch(q.cpu().numpy(), tip_delta.cpu().numpy(),
                                            desired_R.cpu().numpy(), np.array([False]))
            delta = (q.new_tensor(solved) - measured_arm).clamp(-MAX_JOINT_STEP, MAX_JOINT_STEP)
            previous = target[:, self.arm].double()
            # Integrate into the actuator target so the first manual tick preserves its load.
            # Limit added tracking error if contact prevents the requested motion.
            lag_limit = (previous - measured_arm).abs().clamp_min(.040)
            proposal = (previous + delta).clamp(measured_arm-lag_limit, measured_arm+lag_limit)
            requested[:, self.arm] = proposal.to(target.dtype)
        for key, joints in self.fingers.items():
            for idx, gain in joints:
                requested[:, idx] += finger[key] * gain * FINGER_RAD
        env = self.env
        safe, _ = project_target(target, requested, env.limits, env.step, env.vel, self.dt)
        if wrist_active:
            safe[:, self.arm] = self.clearance.project(
                target[:, self.arm].double(), safe[:, self.arm].double()).to(safe.dtype)
        return safe

    def close(self):
        self.placo.close()
