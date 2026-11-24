"""Offline, diagnostic-only scoring of a supported whole-cube turn; no original gate is changed.

The input is one post-step trajectory: height_m must already be corrected for the
cube's own half-edge. Both durations use the recorded control rate. No finger
count, desired wrist pose, or upper bound on angular speed is imposed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from contact_hold_physical_metrics import COLLISION_EPS, TABLE_EPS

LIFT_M = 0.02
SLIP_M = 0.01
TURN_DEG = 15.
HOLD_S = 1.
POST_S = 0.5
CONFIRM = 3  # repeated, separated baseline and turned observations; never sum tiny steps


def _vec(values, shape, name):
    a = np.asarray(values, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Invalid {name}')
    return a


def _bit(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'Invalid {name} flag')
    return bool(value)


def _assembly(row):
    layer = row['layer_measurement']
    angle = float(layer['maximum_relative_rotation_deg'])
    if not math.isfinite(angle) or angle < 0:
        raise ValueError('Invalid layer angle')
    return _bit(layer['assembled_for_grasp'], 'assembled') and angle <= 30.


def build_evidence(rows, cfg, *, assembly_check=None):
    """start_by_end[i]: earliest index valid through i, or i+1 if unsupported.

    Any later start index (up to i) is also valid: all pairs of palm-relative
    cube centers in the window are <=1 cm apart. assembly_check(row) replaces
    only layer geometry validation, e.g. for legitimate aligned face turns.
    """
    if not rows:
        raise ValueError('Empty trajectory')
    hz = float(cfg['control_hz'])
    robot_z = float(cfg['robot_z'])
    contact_min = float(cfg['success']['contact_impulse_min'])
    if not (math.isfinite(hz) and hz > 0 and math.isfinite(robot_z)
            and math.isfinite(contact_min) and contact_min > 0):
        raise ValueError('Invalid configuration')
    check = _assembly if assembly_check is None else assembly_check
    n = len(rows)
    palmlocal = np.empty((n, 3))
    poses = np.empty((n, 4))
    starts = []
    support = []
    left = 0
    first = rows[0]['step']
    reset = rows[0]['pose_reset_count']
    if type(reset) is not int or reset < 0:
        raise ValueError('Invalid reset counter')
    for i, r in enumerate(rows):
        if type(r['step']) is not int or r['step'] != first+i or r['pose_reset_count'] != reset:
            raise ValueError('Control step gap or pose reset')
        names = ('height_m', 'table_impulse', 'self_contact_impulse', 'robot_table_impulse')
        height, table, collision, robot_table = _vec([r[k] for k in names], (4,), 'physical measurements')
        impulses = _vec(r['contact_impulse'], (2,), 'hand contact')
        if min(table, collision, robot_table, *impulses) < 0:
            raise ValueError('Negative contact impulse')
        tf = _vec(r['measured_hand_frames']['left_palm_link'], (4, 4), 'left palm frame')
        xyz = _vec(r['cube_position'], (3,), 'cube center')
        R = tf[:3,:3]
        if (not np.allclose(R.T @ R, np.eye(3), atol=1e-4, rtol=0) or
                np.linalg.det(R) < .999 or not np.allclose(tf[3], [0,0,0,1], atol=1e-6, rtol=0)):
            raise ValueError('Invalid palm transform')
        palmlocal[i] = R.T @ (xyz - [0,0,robot_z] - tf[:3,3])
        q = _vec(r['cube_quaternion_wxyz'], (4,), 'cube quaternion')
        norm = np.linalg.norm(q)
        if abs(norm-1) > 1e-3:
            raise ValueError('Non-unit cube quaternion')
        poses[i] = q/norm
        flags = [_bit(r[k], k) for k in ('cube_table_contact', 'drop', 'arm_soft_limit_hit',
                                          'actual_arm_soft_limit_violation')]
        limits = r['hard_limit_hit']
        if len(limits) != 56:
            raise ValueError('Expected all 56 hard-limit flags')
        hard_hit = any([_bit(b, 'hard_limit_hit') for b in limits])
        assembled = _bit(check(r), 'assembly decision')
        active = bool(height >= LIFT_M and table < TABLE_EPS and impulses[0] > contact_min and
                      not any(flags) and not hard_hit and collision <= COLLISION_EPS and
                      robot_table <= COLLISION_EPS and assembled)
        support.append(active)
        if not active:
            left = i+1
        elif left < i:
            d = palmlocal[left:i] - palmlocal[i]
            bad = np.flatnonzero(np.einsum('ij,ij->i', d, d) > SLIP_M**2 + 1e-14)
            if bad.size:
                left += int(bad[-1])+1
        starts.append(left)
    return dict(start_by_end=starts, support=support, quaternions=poses, hz=hz,
                hold_frames=math.ceil(HOLD_S*hz), post_frames=math.ceil(POST_S*hz),
                confirm_frames=CONFIRM)


def summarize_rotation(rows, cfg, evidence=None):
    """Report airborne support, controlled hold, repeatable net turn and end state."""
    e = build_evidence(rows, cfg) if evidence is None else evidence
    starts = np.asarray(e['start_by_end'], dtype=int)
    quats, hz = e['quaternions'], e['hz']
    hold, post, conf = e['hold_frames'], e['post_frames'], e['confirm_frames']
    n = len(rows)
    def span(i, j):
        length = max(0,j-i+1)
        return dict(frames=length, seconds=length/hz,
                    step_range=[rows[i]['step'],rows[j]['step']] if length else None)
    best_raw = span(0,-1)
    raw_start = 0
    for i, active in enumerate(e['support']):
        if not active:
            raw_start = i+1
        if i+1-raw_start > best_raw['frames']:
            best_raw = span(raw_start,i)
    lengths = np.arange(n)+1-starts
    i = int(lengths.argmax())
    best = span(int(starts[i]), i)
    max_any = max_eligible = 0.
    first_event = last_anchor = None
    at_end = False
    for a in range(n):
        end = int(np.searchsorted(starts,a,side='right')-1)
        if end-a+1 < 2*conf:
            continue
        lo = a+conf
        # Require rotation between two separate 3-frame blocks. A one-frame
        # spike in either the reference or the later orientation cannot count.
        dots = np.max(np.abs(quats[lo:end+1] @ quats[a:a+conf].T), axis=1)
        angles = np.degrees(2*np.arccos(np.clip(dots,0,1)))
        persistent = np.lib.stride_tricks.sliding_window_view(angles,conf).min(axis=1)
        max_any = max(max_any, float(persistent.max()))
        count = min(len(persistent), end-post-lo+1)
        if end-a+1 < hold or count <= 0:
            continue
        eligible = persistent[:count]
        max_eligible = max(max_eligible,float(eligible.max()))
        hits = np.flatnonzero(eligible >= TURN_DEG-1e-9)
        if not hits.size:
            continue
        turn = lo+int(hits[0])
        finish = max(a+hold-1, turn+post, turn+conf-1)
        if first_event is None or finish < first_event[0]:
            first_event = (finish,a,turn)
        if end == n-1:
            at_end = True
            last_anchor = a
    def first_hit():
        if first_event is None:
            return None
        finish, a, turn = first_event
        return dict(reference_step=rows[a]['step'], threshold_step=rows[turn]['step'],
                    confirmed_step=rows[finish]['step'])
    r = rows[-1]
    return dict(kind='supported_whole_cube_world_turn_v1', diagnostic_only=True,
                criteria=dict(height_m=LIFT_M, palm_diameter_m=SLIP_M, hold_s=HOLD_S,
                              post_turn_s=POST_S, degrees=TURN_DEG, control_hz=hz,
                              hold_frames=hold, additional_post_turn_frames=post,
                              reference_and_turn_confirmation_frames=conf),
                best_supported=best_raw, best_controlled_hold=best,
                ever_hold=bool(best['frames'] >= hold), ending_hold=bool(lengths[-1] >= hold),
                max_controlled_turn_deg=max_any, max_duration_eligible_turn_deg=max_eligible,
                ever_hold_and_turn=first_event is not None, ending_hold_and_turn=at_end,
                first_success=first_hit(), final_success_reference_step=(None if last_anchor is None
                                                                       else rows[last_anchor]['step']),
                final_supported=e['support'][-1], final_height_m=r['height_m'],
                final_hand_contact=bool(r['contact_impulse'][0] > cfg['success']['contact_impulse_min']),
                final_table_contact=bool(r['cube_table_contact'] or r['table_impulse'] >= TABLE_EPS),
                final_drop_flag=bool(r['drop']))


def audit_run(run):
    """Require a manifest-backed full rollout; never edit the source artifacts."""
    run = Path(run)
    raw = (run/'artifact_manifest.json').read_bytes()
    manifest = json.loads(raw)
    def read(name):
        blob = (run/name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'Hash mismatch: {run/name}')
        return json.loads(blob)
    cfg, rows, result = map(read, ('config.json','trajectory.json','result.json'))
    if (not manifest['full_episode'] or result.get('execution_interrupted')
            or result.get('cube_pose_writes_during_control') != 0 or len(rows) != result['steps']):
        raise ValueError('Incomplete trajectory or rewritten cube pose')
    return dict(run=str(run.resolve()), seed=result['seed'], checkpoint_sha256=result.get('checkpoint_sha256'),
                trajectory_sha256=manifest['sha256']['trajectory.json'], manifest_sha256=hashlib.sha256(raw).hexdigest(),
                **summarize_rotation(rows, cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([audit_run(run) for run in args.runs], indent=2, allow_nan=False))
