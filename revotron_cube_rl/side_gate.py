"""Hash-bound physical side-teacher gate, separate from historical downward grasp."""
from pathlib import Path
from evidence_paths import local_path, verified_sha, verify_source
import argparse,json,hashlib
from acceptance import assess_pickup
ROOT=Path(__file__).resolve().parent
FILES=['side_grasp.py','side_control.py','side_experiment.py','self_contact_audit.py','side_gate.py','kinematics.py','config.json','environment.py','joint_safety.py','acceptance.py','isaac_compat.py']
def sha(path):
    return verified_sha(path,ROOT)
def require_side_gate():
    gate=json.loads((ROOT/'side_teacher_gate.json').read_text())
    if not gate['passed']:raise RuntimeError('Side teacher not validated')
    for f,h in gate['source_sha256'].items():
        verify_source(f,h,ROOT,'side_teacher_gate.json')
    for f,h in (gate['assets_sha256']|gate['evidence_sha256']).items():
        if sha(f)!=h:raise RuntimeError('Stale side teacher evidence or asset: '+f)
    return gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--development',required=True);p.add_argument('--heldout',required=True);a=p.parse_args()
    runs=[ROOT/'runs'/n for n in [a.development,a.heldout]];evidence={};results=[]
    for run in runs:
        cfg=json.loads((run/'runtime_config.json').read_text());d=json.loads((run/'result.json').read_text());params=json.loads((run/'experiment.json').read_text())
        if params['widths']!='.096' or params['offsets']!='.025' or params['wrist']!=35. or params['tilt']!=15. or params['finger_mode']!='curled' or params['hold']!=3.:raise ValueError('Side parameter mismatch')
        if d['controller_sha256']!=sha(ROOT/'side_grasp.py'):raise ValueError('Teacher code changed')
        for f in ['side_grasp.py','side_control.py','side_experiment.py','self_contact_audit.py']:
            if sha(run/f)!=sha(ROOT/f):raise ValueError('Snapshot mismatch')
        accepted=[]
        for trial in d['trials']:
            f=run/f"trajectory_{trial['candidate']}_{trial['seed']}.json";rows=json.loads(f.read_text())
            accepted.append(assess_pickup(rows,cfg)['accepted'] and sum(r['self_contact_impulse'] for r in rows)<=1e-6 and sum(r['arm_body_impulse'] for r in rows)<=1e-6);evidence[str(f)]=sha(f)
        if not accepted or not all(accepted):raise ValueError('Side physical gate failed')
        results.append(d)
        for f in ['result.json','runtime_config.json','experiment.json']:evidence[str(run/f)]=sha(run/f)
    dev={x['seed'] for x in results[0]['trials']};test={x['seed'] for x in results[1]['trials']}
    if len(test)<5 or dev&test:raise ValueError('Invalid side heldout split')
    old=json.loads((ROOT/'staged_teacher_gate.json').read_text())
    gate={'passed':True,'kind':'traditional_side_teacher','heldout_seeds':sorted(test),'accepted':len(test),'total':len(test),'required_hold_seconds':3.,'source_sha256':{f:sha(ROOT/f) for f in FILES},'assets_sha256':old['assets'],'evidence_sha256':evidence,'parameters':{'width':.096,'palm_offset':.025,'wrist_deg':35.,'tilt_deg':15.,'finger_mode':'curled','lift_forward_m':.035,'finger_tip_pitch_m':.028,'little_MPR_rad':-.30,'thumb_CMP_rad':.03},'robot_self_contact_impulse_sum_max_Ns':1e-6}
    (ROOT/'side_teacher_gate.json').write_text(json.dumps(gate,indent=2));print('SIDE_GATE',len(test),'/',len(test))
if __name__=='__main__':main()
