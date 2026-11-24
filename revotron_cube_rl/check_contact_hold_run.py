"""Inspect physical hold failures and optionally qualify a successful BC source."""
import argparse
import json
from pathlib import Path

import numpy as np

from contact_hold_dataset import audit_demo
from tilt_audit import contact_topology, tilt_error, FINGERS


def inspect(run):
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    rows = json.loads((run/'trajectory.json').read_text())
    hold = [r for r in rows if r['phase'] == 'tilt_hold']
    topology = [contact_topology(r, 'left') for r in hold]
    report = dict(run=str(run.resolve()), steps=result['steps'], failure=result['failure'],
                  dynamic_grasp_passed=result['dynamic_grasp_passed'],
                  wrist_hold_s=result['wrist']['longest_jointly_qualified_hold_s'], hold_frames=len(hold),
                  contact_frames={f:sum(t[f] for t in topology) for f in FINGERS},
                  angular_speed_pass_frames=sum(r['angular_speed'] < .2 for r in hold),
                  angular_speed_min_median_max_rad_s=np.percentile([r['angular_speed'] for r in hold], [0,50,100]).tolist() if hold else [],
                  tilt_angle_pass_frames=sum(tilt_error(r) <= 5 for r in hold),
                  robot_table_impulse_sum_Ns=result['robot_table_impulse_sum_Ns'],
                  self_contact_impulse_sum_Ns=result['self_contact_impulse_sum_Ns'])
    try:
        *_, source = audit_demo(run)
        report.update(training_source_qualified=True, source=source)
    except (ValueError, KeyError, OSError) as exc:
        report.update(training_source_qualified=False, qualification_error=str(exc))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--require-qualified', action='store_true')
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    report=inspect(a.run)
    text=json.dumps(report, indent=2)
    print(text)
    if a.output:
        with a.output.open('x') as f:
            f.write(text+'\n')
    if a.require_qualified and not report['training_source_qualified']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
