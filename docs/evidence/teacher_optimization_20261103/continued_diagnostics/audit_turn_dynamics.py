"""Read-only development diagnostics; never replaces raw-velocity success gates."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation as R

def quantiles(values):
    return dict(zip(('p50', 'p99', 'max'), map(float, np.quantile(values, [.5, .99, 1])))) if len(values) else None

def audit(run):
    trajectory = run / 'trajectory.json'
    rows = json.loads(trajectory.read_text())
    names = json.loads((run / 'config.json').read_text())['joint_names']
    result = json.loads((run / 'result.json').read_text())
    report = {'run': run.name, 'seed': result['seed'], 'split': result['split'],
              'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
              'failure': result['failure'], 'phases': {},
              'interpretation': 'Finite differences are control-interval averages, raw velocities are endpoint samples; disagreement suggests unresolved dynamics or sampling, not proof of a velocity bug. Gates use original raw velocities.'}
    for phase in ('opposition_hold', 'opposed_face_turn', 'opposed_face_hold'):
        selected = [row for row in rows if row['phase'] == phase]
        if not selected: continue
        r = {'frames': len(selected), 'final_face_angle_deg': selected[-1]['layer_measurement']['face_angle_deg'],
             'core_angular_speed_rad_s': quantiles([x['angular_speed'] for x in selected]),
             'selected_angular_speed_rad_s': quantiles([x['layer_measurement']['maximum_selected_angular_velocity_rad_s'] for x in selected]),
             'contacts': {}, 'body_velocity_comparison': {}}
        for finger in ('thumb', 'index', 'middle', 'ring', 'little'):
            present = [x['opposition_audit']['right_contacts']['fingers'][finger] for x in selected]
            first = next((x for x, valid in zip(selected, present) if not valid), None)
            ids = [i for i,n in enumerate(names) if n.startswith('right_' + finger + '_')]
            errors = [max(abs(x['q'][i] - x['applied_target'][i]) for i in ids) for x in selected]
            r['contacts'][finger] = {'fraction': float(np.mean(present)),
                'first_missing_step': first['step'] if first else None,
                'face_angle_at_first_missing_deg': first['layer_measurement']['face_angle_deg'] if first else None,
                'maximum_joint_tracking_error_rad': quantiles(errors)}
        for body in ('Core', 'nZ', 'nZ_pX', 'nX_nY_nZ'):
            raw = [np.linalg.norm(x['cubelets'][body]['angular_velocity_rad_s']) for x in selected]
            diff = []
            for previous, current in zip(selected, selected[1:]):
                def rot(row): return R.from_quat(np.asarray(row['cubelets'][body]['quaternion_wxyz'])[[1,2,3,0]])
                dt = (current['step'] - previous['step']) / 30.
                if dt > 0: diff.append((rot(current) * rot(previous).inv()).magnitude() / dt)
            r['body_velocity_comparison'][body] = {'reported_rad_s': quantiles(raw), 'control_interval_average_rad_s': quantiles(diff)}
        report['phases'][phase] = r
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+', type=Path); p.add_argument('--output', required=True, type=Path); a=p.parse_args()
    reports = [audit(r) for r in a.runs]
    a.output.write_text(json.dumps(reports, indent=2, allow_nan=False))
    for r in reports:
        print(r['run'], r['failure'])
        for phase, x in r['phases'].items():
            print(phase, 'angle', round(x['final_face_angle_deg'], 3), 'contacts', {f: round(c['fraction'], 3) for f,c in x['contacts'].items()})
