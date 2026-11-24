"""Single-lane physical readbacks for human-operated R27 demonstrations."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from natural_face90_turn_contacts import _actor_paths
from natural_face90_v5_metrics import DIGIT, LANE, RIGHT_ARM, Layer, conj, degrees, mul


SCHEMA = 'r27_human_two_finger_demo_v2'
COLUMNS = ('thumb_distal', 'thumb_proximal', 'index_distal', 'index_proximal', 'other_right_digit',
           'right_palm_or_arm', 'left_hand', 'other')
FACES = ('nZ', 'remaining')
META = 'demo_metadata.json'
FRAMES = 'demo_frames.jsonl'
SUMMARY = 'demo_summary.json'


def _json_line(data):
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'


def _new_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(_json_line(data))


def _joints(tensor):
    q = torch.as_tensor(tensor).detach().reshape(-1)
    if len(q) != 56 or not q.isfinite().all():
        raise ValueError('Expected 56 finite canonical joint values')
    return q.cpu().tolist()


def contact_column(path):
    match = LANE.fullmatch(str(path))
    if not match or int(match[1]) != 0:
        raise ValueError('Unexpected contact actor outside demo lane: ' + str(path))
    link = match[2]
    if DIGIT.fullmatch(link):
        for finger, distal, proximal in (('thumb', 0, 1), ('index', 2, 3)):
            if link.startswith('/Robot/right_' + finger + '_'):
                return distal if link == f'/Robot/right_{finger}_DIP_link' else proximal
        return 4
    if link.startswith('/Robot/right_') or RIGHT_ARM.fullmatch(link):
        return 5
    if link.startswith('/Robot/left_') or link == '/Robot/Link7_L':
        return 6
    return 7


class DigitContacts:
    """Tap the existing cube-side PhysX packets, preserving the complete original reducer."""
    def __init__(self, env, names):
        self.base = env.reducer
        if isinstance(self.base, DigitContacts):
            raise ValueError('Demo contact readback already installed')
        [(self.view, lanes)] = [(v, ls) for kind, v, ls in env.contact_views if kind == 'cube']
        paths = list(self.view.sensor_paths)
        if len(paths) != len(lanes) or len(paths) != self.view.sensor_count:
            raise ValueError('Incomplete native cube sensors')
        sensor_names = []
        for path, lane in zip(paths, lanes):
            match = LANE.fullmatch(str(path))
            if not match or int(match[1]) != 0 or int(lane) != 0 or not match[2].startswith('/Cube/'):
                raise ValueError('Invalid cube sensor: ' + str(path))
            sensor_names.append(match[2].rsplit('/', 1)[-1])
        if len(sensor_names) != 26 or set(sensor_names) != set(names):
            raise ValueError('Expected one native sensor for each of the 26 cubelets')
        self.faces = np.array([int('nZ' not in n.split('_')) for n in sensor_names])
        self.kinds = {}
        self.serial = 0
        self.substeps = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def clear(self, env):
        self.base.clear(env)
        self.substeps.clear()
        self.serial += 1

    def update(self, env):
        self.base.update(env)
        # dt=1 returns solver impulses in N s. Use only the cube sensor side
        # so a contact is counted once even when both actors have sensors.
        raw = self.view.get_raw_contact_data(1.)
        counts = raw[4].reshape(-1).long().cpu().numpy()
        starts = raw[5].reshape(-1).long().cpu().numpy()
        n = int(counts.sum())
        if len(counts) != len(self.faces) or counts.shape != starts.shape or (counts < 0).any() or n >= self.view.max_contact_data_count:
            raise ValueError('Incomplete or exhausted demo contact buffer')
        result = np.zeros((len(FACES), len(COLUMNS)), dtype=float)
        if n:
            sensor = np.repeat(np.arange(len(counts)), counts)
            indices = np.arange(n) + (starts - (counts.cumsum() - counts))[sensor]
            other = raw[6].reshape(-1).view(torch.int64)
            force = raw[0].reshape(-1)
            if indices.min() < 0 or indices.max() >= min(len(force), len(other)):
                raise ValueError('Invalid demo contact offsets')
            indices = torch.as_tensor(indices, device=force.device)
            actors = other[indices].cpu().numpy()
            impulses = force[indices].detach().double().abs().cpu().numpy()
            if not np.isfinite(impulses).all():
                raise ValueError('Nonfinite native impulse')
            keys, inverse = np.unique(actors, return_inverse=True)
            missing = [int(k) for k in keys if k not in self.kinds]
            if missing:
                paths = _actor_paths(env, missing)
                for k, path in zip(missing, paths, strict=True):
                    self.kinds[k] = contact_column(path)
            col = np.array([self.kinds[k] for k in keys])[inverse]
            np.add.at(result, (self.faces[sensor], col), impulses)
        self.substeps.append(result)

    def validate(self):
        self.base.validate()

    def frame(self, n):
        if len(self.substeps) != n:
            raise ValueError('Incomplete physical interval in demo')
        samples = np.stack(self.substeps)
        target = samples[:, 0]
        thumbs = target[:, :2].sum(axis=1)
        indexes = target[:, 2:4].sum(axis=1)
        return dict(thumb_Ns=float(thumbs.sum()), index_Ns=float(indexes.sum()),
                    other_Ns=float(target[:, 4].sum()),
                    thumb_index_same_substeps=int(np.count_nonzero((thumbs > 0) & (indexes > 0))),
                    impulse_substeps_Ns=samples.tolist())


class Recorder:
    """Begin after reset/observer setup. Record each controlled interval after step_physics."""
    def __init__(self, out, env, metadata=None):
        if env.n != 1 or env.dec != 8 or not math.isclose(env.dt, 1/240):
            raise ValueError('Expected one 240 Hz R27 world with 30 Hz controls')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.links = tuple(env.robot.shared_metatype.link_names)
        if not self.links or len(set(self.links)) != len(self.links):
            raise ValueError('Need distinct physical robot link names for demo recording')
        self.names = tuple(name for name in env.views if name != 'Core')
        core, pieces = self._poses()
        self.layer = Layer(self.names, core, pieces)
        self.tap = DigitContacts(env, self.names)
        self.last_serial = self.tap.serial
        self.handoff = None
        self.n = self.manual = self.both = self.both_substeps = self.unsafe = 0
        self.last_step = None
        self.last_reason = None
        self.angle = float(self.layer.measure(core, pieces, 90.)['face_deg'][0])
        self.peak = self.minimum = 0.
        self.last_manual = None
        self.closed = None
        self.digest = hashlib.sha256()
        self.threshold = float(env.cfg['success']['contact_impulse_min'])
        env_meta = dict(schema=SCHEMA, simulation_only=True, control_hz=30, physics_hz=240,
                        source=dict(metadata or {}), domain=env.domain_receipt, joint_names=list(env.names),
                        cubelet_names=list(self.names), robot_link_names=list(self.links), pose_format='world [x_m,y_m,z_m,qx,qy,qz,qw]',
                        joint_units='radians, radians/second', layer='nZ (nine pieces relative to Core at reset)',
                        impulse_unit='normal solver impulse, N s; cube-side native records at 240 Hz',
                        impulse_axes=['substep','cube_layer','contact_actor'], layers=FACES, actors=COLUMNS,
                        distal_contact='native DIP rigid body (includes its attached shapes)',
                        other_Ns_definition='right middle/ring/little on target layer',
                        control_frame_threshold_Ns=self.threshold)
        for name in (META, FRAMES, SUMMARY):
            if (self.out/name).exists():
                raise FileExistsError(self.out/name)
        _new_json(self.out/META, env_meta)
        self.metadata_sha256 = hashlib.sha256((self.out/META).read_bytes()).hexdigest()
        self.fp = (self.out/FRAMES).open('xb', buffering=0)
        env.reducer = self.tap

    def _poses(self):
        return (self.env.views['Core'].get_transforms().double(),
                torch.stack([self.env.views[n].get_transforms() for n in self.names], 1).double())

    def begin_handoff(self, step, target):
        if self.handoff is not None or self.closed is not None:
            raise ValueError('Duplicate or closed human handoff')
        c, p = self._poses()
        angle = float(self.layer.measure(c, p, 90.)['face_deg'][0])
        self.handoff_core = c.clone()
        self.handoff = dict(step=int(step), command_rad=_joints(target), face_deg=angle,
                            q_rad=_joints(self.env.robot.get_dof_positions()[:,self.env.ids]),
                            dq_rad_s=_joints(self.env.robot.get_dof_velocities()[:,self.env.ids]),
                            cube=dict(zip(('Core',*self.names), torch.cat((c[:,None],p),1)[0].tolist(), strict=True)))
        _new_json(self.out/'demo_handoff.json', self.handoff)

    def record(self, step, phase, action, q_before, command, *, safe=True, end_reason=None):
        if (self.closed is not None or self.env.reducer is not self.tap or self.tap.serial != self.last_serial+1
                or (self.last_step is not None and step != self.last_step+1)):
            raise ValueError('Duplicate, skipped or disconnected demo physics interval')
        if phase == 'manual' and self.handoff is None:
            raise ValueError('Manual input without measured handoff')
        contacts = self.tap.frame(self.env.dec)
        c,p = self._poses()
        measured = {k:v[0].item() for k,v in self.layer.measure(c,p,90.).items()}
        self.angle = measured['face_deg']
        env = self.env
        poses = env.robot.get_link_transforms()[0]
        if poses.shape != (len(self.links),7) or not poses.isfinite().all():
            raise ValueError('Incomplete measured robot poses')
        # Store readbacks at native float precision; no inferred fingertip or
        # FK replacement for rigid-body measurements in the visual replay.
        bodies = dict(zip(self.links, poses.detach().cpu().tolist(), strict=True))
        row = dict(step=int(step), phase=phase, action=action, safe=bool(safe), end_reason=end_reason,
                   q_before_rad=_joints(q_before), command_rad=_joints(command),
                   q_rad=_joints(env.robot.get_dof_positions()[:,env.ids]),
                   dq_rad_s=_joints(env.robot.get_dof_velocities()[:,env.ids]),
                   cube=dict(zip(('Core',*self.names),torch.cat((c[:,None],p),1)[0].tolist(),strict=True)),
                   bodies=bodies, angle_deg=self.angle, structure_ok=measured['structure_ok'], layer=measured,
                   scene_contacts=env.contacts[0].tolist(), **contacts)
        is_manual = phase == 'manual'
        if self.handoff is not None:
            reference = self.handoff_core
            row['core_shift_since_handoff_m'] = float((c[:,:3]-reference[:,:3]).norm())
            row['core_rotation_since_handoff_deg'] = float(degrees(mul(c[:,3:],conj(reference[:,3:])))[0])
            row['face_since_handoff_deg'] = (self.angle - self.handoff['face_deg'] + 180.)%360.-180.
        encoded = _json_line(row).encode('utf-8')
        self.fp.write(encoded)
        self.digest.update(encoded)
        self.n += 1
        if is_manual:
            self.manual += 1
            self.last_manual = row['face_since_handoff_deg']
            self.peak = max(self.peak, self.last_manual)
            self.minimum = min(self.minimum, self.last_manual)
            self.both_substeps += row['thumb_index_same_substeps']
            self.both += int(min(row['thumb_Ns'],row['index_Ns']) > self.threshold
                             and row['thumb_index_same_substeps'] > 0)
        self.unsafe += int(not row['safe'])
        self.last_step, self.last_serial, self.last_reason = step, self.tap.serial, end_reason
        return row

    def finish(self, reason, *, failed=False):
        if self.closed is not None:
            return self.closed
        self.fp.close()
        if self.env.reducer is self.tap:
            self.env.reducer = self.tap.base
        result = dict(schema=SCHEMA, reason=str(reason), status='failed' if failed or self.unsafe else 'stopped',
                      simulation_only=True, independent_evaluation=False, recorded_frames=self.n,
                      manual_frames=self.manual, unsafe_frames=self.unsafe, last_step=self.last_step,
                      last_frame_reason=self.last_reason, observed_face_deg=self.angle,
                      final_manual_face_change_deg=self.last_manual,
                      peak_manual_face_change_deg=self.peak if self.manual else None,
                      min_manual_face_change_deg=self.minimum if self.manual else None,
                      manual_thumb_index_overlap_frames=self.both, manual_thumb_index_overlap_substeps=self.both_substeps,
                      trajectory=FRAMES, metadata_sha256=self.metadata_sha256, frames_sha256=self.digest.hexdigest(), handoff_recorded=self.handoff is not None)
        _new_json(self.out/SUMMARY, result)
        self.closed = result
        return result

    def close(self):
        return self.finish('interrupted', failed=True)
