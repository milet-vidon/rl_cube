"""CPU-only Placo IK for the calibrated index contact point on the simulated robot."""
import hashlib
import json
from importlib import metadata
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics import RobotKinematics


HERE = Path(__file__).resolve().parent
URDF = HERE.parent / 'revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf'
URDF_SHA = '598cc4cc94a37e35517eba4e450b3a3b334214549c7e196bf3f1332940e71325'
PAD = HERE / 'face_contact_calibration_v1.json'
PAD_SHA = 'b9ab73c9e7e2c922ffaf1270196fce8c2a4acd635dfe8842def32d98bcbe7b44'
RIGHT = tuple(f'Joint{i}_R' for i in range(1, 8))
PALM = 'right_palm_link'
TCP = 'calibrated_right_index_pad'
MAX_JOINT_STEP = .012
MAX_TCP_STEP_M = .005
MAX_WRIST_STEP_RAD = .018


def _version(placo):
    # cmeel installs extension modules under cmeel.prefix; the wheel's
    # distribution metadata is a sibling of that prefix.
    extension = Path(placo.__file__).resolve()
    prefix = next((p for p in extension.parents if p.name == 'cmeel.prefix'), None)
    search = [prefix.parent] if prefix else None
    dists = metadata.distributions(name='placo', path=search) if search else metadata.distributions(name='placo')
    version = next((d.version for d in dists), None)
    if version != '0.9.23':
        raise RuntimeError(f'Expected Placo 0.9.23 wheel, found {version}')
    return version


def _kinematic_model(urdf, pad):
    xml = ET.fromstring(urdf)
    # The guide uses its original forearm clearance projection and the scene
    # measures all actual contacts. IK only needs link/joint transforms.
    for link in xml.findall('link'):
        for geometry in [*link.findall('visual'), *link.findall('collision')]:
            link.remove(geometry)
    ET.SubElement(xml, 'link', name=TCP)
    joint = ET.SubElement(xml, 'joint', name=TCP + '_fixed', type='fixed')
    ET.SubElement(joint, 'parent', link=pad['source_link'])
    ET.SubElement(joint, 'child', link=TCP)
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, pad['point_local'])), rpy='0 0 0')
    return xml


class PlacoArm:
    """One solver, re-seeded from all measured joint positions for each lane/tick."""
    def __init__(self, joint_names, dt):
        import placo
        self.names = tuple(joint_names)
        version = _version(placo)
        urdf, calibration = URDF.read_bytes(), PAD.read_bytes()
        if hashlib.sha256(urdf).hexdigest() != URDF_SHA or hashlib.sha256(calibration).hexdigest() != PAD_SHA:
            raise ValueError('Unexpected robot URDF or calibrated pad asset')
        if not np.isclose(dt, 1 / 30, rtol=0, atol=1e-9):
            raise ValueError('Expected the 30 Hz Cartesian control period')
        cfg = json.loads((HERE / 'config.json').read_bytes())
        self.pad = json.loads(calibration)
        model = _kinematic_model(urdf, self.pad)
        self.robot = placo.RobotWrapper(str(URDF), placo.Flags.ignore_collisions,
                                        ET.tostring(model, encoding='unicode'))
        if len(self.names) != 56 or set(self.names) != set(self.robot.joint_names()):
            raise ValueError('Expected all 56 distinct measured joint names from the robot')
        self.input_ids = np.array([self.robot.get_joint_offset(n) for n in self.names])
        self.right_ids = np.array([self.robot.get_joint_offset(n) for n in RIGHT])
        self.right_input = np.array([self.names.index(n) for n in RIGHT])
        self.seed = self.robot.state.q.copy()  # identity floating base, expressed in the robot base frame
        self.locked = np.ones(len(self.seed), dtype=bool)
        self.locked[self.right_ids] = False
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        for n in self.names:
            if n not in RIGHT:
                self.solver.mask_dof(n)
                # A physical contact can push a measured finger slightly past
                # its nominal stop. That locked coordinate cannot be corrected
                # by arm IK. Its measured position still enters the TCP FK.
                self.robot.set_joint_limits(n, -np.inf, np.inf)
        self.solver.dt = dt
        self.limits = np.array([cfg['arm_soft_limits'][n] for n in RIGHT])
        self.max_step = np.empty(7)
        for i, (n, (lo, hi)) in enumerate(zip(RIGHT, self.limits)):
            urdf_limits = self.robot.get_joint_limits(n)
            self.limits[i] = max(lo, urdf_limits[0]), min(hi, urdf_limits[1])
            self.robot.set_joint_limits(n, *self.limits[i])
            velocity = float(model.find(f"joint[@name='{n}']/limit").get('velocity'))
            self.max_step[i] = min(MAX_JOINT_STEP, velocity * cfg['arm_velocity_limit_scale'] * dt)
            self.robot.set_velocity_limit(n, self.max_step[i] / dt)
        self.active_limits = self.limits.copy()
        self.solver.enable_velocity_limits(True)
        self.solver.enable_joint_limits(True)
        self.position = self.solver.add_position_task(TCP, np.zeros(3))
        self.position.configure('pad_position', 'soft', 1.)
        self.orientation = self.solver.add_orientation_task(PALM, np.eye(3))
        self.orientation.configure('palm_orientation', 'soft', .065 ** 2)
        self.solver.add_regularization_task(.008 ** 2)
        self.receipt = dict(backend='placo', version=version, urdf_sha256=URDF_SHA, pad_sha256=PAD_SHA,
                            binary_sha256=hashlib.sha256(Path(placo.__file__).read_bytes()).hexdigest(),
                            solver_dt=float(dt), joints=list(RIGHT), masked_joints=49, fixed_base=True,
                            position_frame=TCP, orientation_frame=PALM)

    def _sync(self, measured):
        self.seed[self.input_ids] = measured
        self.robot.state.q = self.seed
        self.robot.update_kinematics()

    def _reachable_limits(self, right):
        # If the physical joint has been forced outside the soft envelope,
        # demand the closest reachable point on the way back this tick. Restore
        # the canonical limits on the next lane or measurement inside bounds.
        bounds = np.column_stack((np.minimum(self.limits[:, 0], right + self.max_step),
                                  np.maximum(self.limits[:, 1], right - self.max_step)))
        for i in np.flatnonzero((bounds != self.active_limits).any(axis=1)):
            self.robot.set_joint_limits(RIGHT[i], *bounds[i])
        self.active_limits = bounds

    def solve_batch(self, q_measured, position_delta_base, desired_palm_R_base, recovering):
        q, errors, rotations = map(lambda v: np.asarray(v, dtype=np.float64),
                                   (q_measured, position_delta_base, desired_palm_R_base))
        hold = np.asarray(recovering)
        n = len(q)
        if (q.shape != (n, 56) or errors.shape != (n, 3) or rotations.shape != (n, 3, 3)
                or hold.shape != (n,) or not all(np.isfinite(v).all() for v in (q, errors, rotations))
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6)):
            raise ValueError('Invalid measured joint, TCP displacement or palm orientation batch')
        step = errors * (MAX_TCP_STEP_M / np.maximum(np.linalg.norm(errors, axis=1, keepdims=True), MAX_TCP_STEP_M))
        out = np.empty((n, 7))
        for lane in range(n):
            self._sync(q[lane])
            self._reachable_limits(q[lane, self.right_input])
            current_R = self.robot.get_T_world_frame(PALM)[:3, :3]
            current_tcp = self.robot.get_T_world_frame(TCP)[:3, 3]
            self.position.target_world = current_tcp + step[lane]
            if hold[lane]:
                target_R = current_R
            else:
                rv = Rotation.from_matrix(rotations[lane] @ current_R.T).as_rotvec()
                rv *= MAX_WRIST_STEP_RAD / max(np.linalg.norm(rv), MAX_WRIST_STEP_RAD)
                target_R = Rotation.from_rotvec(rv).as_matrix() @ current_R
            self.orientation.R_world_frame = target_R
            self.solver.solve(True)
            solved = self.robot.state.q
            if (not np.isfinite(solved).all()
                    or np.max(np.abs(solved[self.locked] - self.seed[self.locked])) > 1e-8
                    or np.any(np.abs(solved[self.right_ids] - self.seed[self.right_ids]) > self.max_step + 1e-7)
                    or np.any(solved[self.right_ids] < self.active_limits[:, 0] - 1e-7)
                    or np.any(solved[self.right_ids] > self.active_limits[:, 1] + 1e-7)):
                raise RuntimeError('Placo violated fixed-base, joint mask or arm speed constraints')
            out[lane] = solved[self.right_ids]
        return out


def self_check():
    """Compare FK to the project's independent implementation and run real IK in two unequal lanes."""
    names = json.loads((HERE / 'config.json').read_bytes())['joint_names']
    arm = PlacoArm(names, 1 / 30)
    endpoint = json.loads((HERE / 'natural_face90_v5_regression.json').read_bytes())['r27_endpoint']['q']
    q = np.tile(endpoint, (2, 1))
    q[1, names.index('Joint3_R')] += .13
    q[1, names.index('right_index_PIP_joint')] += .07
    reference = RobotKinematics()
    tcp, palms, diffs, rot_diffs = [], [], [], []
    for lane in q:
        joints = dict(zip(names, lane))
        arm._sync(lane)
        independent_pad = reference.fk_link(joints, arm.pad['source_link'])
        independent_tcp = independent_pad[:3, 3] + independent_pad[:3, :3] @ arm.pad['point_local']
        independent_palm = reference.fk_link(joints, PALM)
        tcp.append(arm.robot.get_T_world_frame(TCP)[:3, 3].copy())
        palm = arm.robot.get_T_world_frame(PALM)
        palms.append(palm[:3, :3].copy())
        diffs.extend([np.linalg.norm(tcp[-1] - independent_tcp), np.linalg.norm(palm[:3, 3] - independent_palm[:3, 3])])
        rot_diffs.append(np.linalg.norm(Rotation.from_matrix(palms[-1] @ independent_palm[:3, :3].T).as_rotvec()))
    goals = np.array([[.0005, 0., 0.], [0., -.0005, 0.]])
    hold = np.array([False, True])
    out = arm.solve_batch(q, goals, palms, hold)
    repeat = arm.solve_batch(q[::-1], goals[::-1], palms[::-1], hold[::-1])[::-1]
    errors = []
    for i, right in enumerate(out):
        measured = q[i].copy()
        measured[arm.right_input] = right
        arm._sync(measured)
        delta = arm.robot.get_T_world_frame(TCP)[:3, 3] - tcp[i]
        errors.append(np.linalg.norm(delta - goals[i]))
    if (max(diffs) > 1e-8 or max(rot_diffs) > 1e-8 or max(errors) > .00025
            or np.max(np.abs(out - repeat)) > 1e-8):
        raise RuntimeError(f'Placo closed-loop geometry or lane isolation check failed: {diffs}, {rot_diffs}, {errors}')
    return dict(receipt=arm.receipt, backend='placo', version=arm.receipt['version'],
                max_fk_diff_m=float(max(diffs)), max_palm_diff_rad=float(max(rot_diffs)),
                max_position_error_after_step_m=float(max(errors)))
