"""Measure an airborne physical hold from post-step data; finger identity/count are not inputs.

The original wrist, tilt and phase audits remain separate. Palm-relative drift is an
audit measurement, not an additional uncalibrated acceptance threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / 'runs/contact_hold_cuda_long_v1/round_015/validation/panel.json'
TABLE_EPS = 1e-5                 # environment.py: CubePickupEnv.step
COLLISION_EPS = 1e-6            # single_face_experiment.py: robot safety
HOLD_LINEAR = 0.03              # settled_wrist_teacher.py: original hold speed check
HOLD_ANGULAR = 0.20


def hand_contact(row, cfg):
    impulses = row['contact_impulse']   # measured cube/left hand and cube/right hand, including palm
    if len(impulses) != 2 or any(not math.isfinite(v) or v < 0 for v in impulses):
        raise ValueError('Missing or invalid hand/cube impulses')
    return impulses[0] > cfg['success']['contact_impulse_min']


def table_free(row):
    return not row['cube_table_contact'] and 0 <= row['table_impulse'] < TABLE_EPS


def safe_and_assembled(row):
    return (not row['drop'] and 0 <= row['self_contact_impulse'] <= COLLISION_EPS
            and 0 <= row['robot_table_impulse'] <= COLLISION_EPS and not row['arm_soft_limit_hit']
            and not row['actual_arm_soft_limit_violation'] and not any(row['hard_limit_hit'])
            and row['layer_measurement']['assembled_for_grasp']
            and 0 <= row['layer_measurement']['maximum_relative_rotation_deg'] <= 30.)


def supported(row, cfg):
    """Airborne at or above pickup height, in real hand contact and within original safety gates."""
    return bool(row['height_m'] >= cfg['success']['lift_m'] and table_free(row) and
                hand_contact(row, cfg) and safe_and_assembled(row))


def steady(row, cfg):
    s = cfg['success']
    return bool(supported(row, cfg) and 0 <= row['linear_speed'] < s['linear_speed_max']
                and 0 <= row['angular_speed'] < s['angular_speed_max'])


def _span(rows, hz, predicate):
    best = (0, 0)
    start = 0
    for i, row in enumerate(rows):
        if predicate(row):
            if i+1-start > best[1]-best[0]:
                best = (start, i+1)
        else:
            start = i+1
    left, right = best
    n = right-left
    return dict(frames=n, seconds=n/hz, step_range=[rows[left]['step'], rows[right-1]['step']] if n else None,
                ending_streak_frames=len(rows)-start)


def _xyz_in_palm(row, cfg, side):
    # measured_hand_frames has root-relative xyz, but cube_position is in world coordinates.
    # T(root<-palm) already includes the configured physical palm offset and local rotation.
    tf = row['measured_hand_frames'][side+'_palm_link']
    xyz = [v-x-t[3] for v, x, t in zip(row['cube_position'], [0, 0, cfg['robot_z']], tf[:3])]
    p = tuple(sum(tf[k][axis]*xyz[k] for k in range(3)) for axis in range(3))
    if not all(map(math.isfinite, p)):
        raise ValueError('Invalid measured palm-local cube position')
    return p


def _palm_drift(rows, cfg, bounds):
    """Only compare within the same hand's continuous contact interval; no switching-frame artefacts."""
    if bounds is None:
        return []
    trace = [r for r in rows if bounds[0] <= r['step'] <= bounds[1]]
    blocks = []
    for index, side in enumerate(('left','right')):
        pts, steps = [], []
        def finish():
            if not pts: return
            blocks.append(dict(hand=side, step_range=[steps[0], steps[-1]], frames=len(pts),
                               max_displacement_from_first_m=max(math.dist(p, pts[0]) for p in pts),
                               first_to_last_m=math.dist(pts[0], pts[-1])))
            pts.clear(); steps.clear()
        for r in trace:
            if r['contact_impulse'][index] > cfg['success']['contact_impulse_min']:
                pts.append(_xyz_in_palm(r, cfg, side)); steps.append(r['step'])
            else:
                finish()
        finish()
    return blocks


def summarize_hold(rows, cfg):
    """Summarize contiguous observed frames; never extrapolate beyond early termination."""
    if not rows: raise ValueError('No post-step evidence')
    hz = cfg['control_hz']; s = cfg['success']; n = max(1,round(hz*s['hold_seconds']))
    if hz <= 0 or s['hold_seconds']<=0 or not (0<s['contact_impulse_min'] and 0<s['lift_m']):
        raise ValueError('Invalid timing, contact or height limits')
    first = rows[0]['step']
    for i, r in enumerate(rows):
        if r['step']!=first+i or r['pose_reset_count']!=rows[0]['pose_reset_count']:
            raise ValueError('Trajectory has a control-step gap or pose rewrite')
        if any(not math.isfinite(r[k]) for k in ('height_m','linear_speed','angular_speed','table_impulse')):
            raise ValueError('Non-finite physical evidence')
    def low(r): return (r['height_m']>0 and table_free(r) and hand_contact(r,cfg) and safe_and_assembled(r))
    def wrist_speed(r): return (supported(r,cfg) and 0<=r['linear_speed']<HOLD_LINEAR
                               and 0<=r['angular_speed']<HOLD_ANGULAR)
    def return_after(test):
        step = next((r['step'] for r in rows if test(r)), None)
        return next((r['step'] for r in rows if step is not None and r['step']>step and not table_free(r)), None)
    physical = _span(rows,hz,lambda r: steady(r,cfg))
    support = _span(rows,hz,lambda r: supported(r,cfg))
    return dict(kind='left_hand_airborne_hold_v1', steps=len(rows),
                criteria=dict(hand_impulse_min_Ns=s['contact_impulse_min'],lift_m=s['lift_m'],table_epsilon_Ns=TABLE_EPS,
                              physical_linear_max_m_s=s['linear_speed_max'],physical_angular_max_rad_s=s['angular_speed_max'],
                              supplementary_wrist_linear_max_m_s=HOLD_LINEAR,supplementary_wrist_angular_max_rad_s=HOLD_ANGULAR,
                              seconds=s['hold_seconds'],required_frames=n,hz=hz),
                max_height_m=max(r['height_m'] for r in rows), final_height_m=rows[-1]['height_m'],
                final_table_contact=not table_free(rows[-1]),final_hand_contact=hand_contact(rows[-1],cfg),
                first_table_return_after_any_supported_liftoff=return_after(low),
                first_table_return_after_supported_lift=return_after(lambda r:supported(r,cfg)),
                any_height_airborne_contact=_span(rows,hz,low),supported=support,wrist_speed_steady=_span(rows,hz,wrist_speed),
                physical_steady=physical,physical_hold_observed=physical['frames']>=n,
                physical_hold_at_end=physical['ending_streak_frames']>=n,
                palm_drift_during_longest_supported=_palm_drift(rows,cfg,support['step_range']),
                palm_drift_during_longest_steady=_palm_drift(rows,cfg,physical['step_range']))


def audit_run(run):
    """Only read intact completed manifest-backed runs; diagnostic reports do not modify the run."""
    run=Path(run)
    manifest=(run/'artifact_manifest.json').read_bytes()
    m=json.loads(manifest)
    def checked(name):
        data=(run/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=m['sha256'][name]:
            raise ValueError(f'Manifest mismatch for {run/name}')
        return json.loads(data)
    cfg=checked('config.json'); rows=checked('trajectory.json'); result=checked('result.json')
    if not m['full_episode'] or result.get('execution_interrupted') or result.get('cube_pose_writes_during_control')!=0 or len(rows)!=result['steps']:
        raise ValueError('Incomplete episode or cube pose writes')
    return dict(run=str(run.resolve()),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                trajectory_sha256=m['sha256']['trajectory.json'],seed=result['seed'],failure=result['failure'],
                checkpoint_sha256=result.get('checkpoint_sha256'),**summarize_hold(rows,cfg))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='*',type=Path,help='Default: both archived round 15 validation runs')
    parser.add_argument('--output',type=Path,help='New diagnostic file; existing files are never overwritten')
    a=parser.parse_args()
    paths=a.runs or [Path(e['run']) for e in json.loads(PANEL.read_bytes())['episodes']]
    txt=json.dumps([audit_run(p) for p in paths],indent=2,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x') as f: f.write(txt)
    else:
        print(txt,end='')
