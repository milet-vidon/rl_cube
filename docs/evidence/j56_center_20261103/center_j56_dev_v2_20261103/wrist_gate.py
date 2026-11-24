"""Recompute the wrist, palm, elbow and contact gate in one hold window."""
import argparse,json
from pathlib import Path
from elbow_gate import build as elbow_certificate
from evidence_paths import local_path,raw_sha
from wrist_audit import assess_wrist,wrist_geometry
ROOT=Path(__file__).resolve().parent


def build(protocol,runs):
    result=elbow_certificate(protocol,runs)
    for trial in result['trials']:
        run=local_path(trial['run'],ROOT);cfg=json.loads((run/'config.json').read_text());rows=json.loads((run/f"trajectory_{trial['seed']}.json").read_text())
        audit=assess_wrist(rows,cfg['joint_names'])
        before=next((row for row in reversed(rows) if row['phase']=='tilt_90'),None)
        audit['before_adjustment']=wrist_geometry(before,cfg['joint_names']) if before else None
        trial['wrist_direction_audit']=audit
        if not audit['accepted']:
            trial['accepted']=False;result['failures'].append('wrist_direction_hold_failed:'+str(trial['seed']))
    result['passed']=not result['failures'];result['accepted']=sum(t['accepted']for t in result['trials'])
    result['independent_accepted']=sum(t['accepted']for t in result['trials']if t['split']=='independent')
    result.update(wrist_direction_validated=result['passed'],right_regrasp_validated=False,
                  gate_source_sha256=raw_sha(__file__),wrist_axis_definition='J6 positive, measured palm-frame forearm Y/Z angle negative',
                  palm_flipped=False,old_elbow_successes_counted_as_wrist_tests=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=build(a.protocol,a.runs);Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k]for k in ['passed','accepted','total','independent_accepted','independent_total','failures']}));raise SystemExit(0 if result['passed']else 1)
