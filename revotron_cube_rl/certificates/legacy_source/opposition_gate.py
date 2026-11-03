"""Recompute asymmetric-teacher acceptance and bind it to frozen evidence."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from acceptance import assess_pickup
from opposition_acceptance import assess_opposition

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_opposition_gate():
    certificate=json.loads((ROOT/'opposition_gate_certificate.json').read_text())
    if not certificate['passed']:raise RuntimeError('The asymmetric teacher has not passed acceptance.')
    for name,digest in certificate['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Teacher source changed: {name}')
    if sha(certificate['grasp_config_path'])!=certificate['grasp_config_sha256']:
        raise RuntimeError('Teacher grasp configuration changed.')
    for path,digest in (certificate['assets_sha256']|certificate['evidence_sha256']).items():
        if sha(path)!=digest:raise RuntimeError(f'Asset or acceptance evidence changed: {path}')
    return certificate

def audit(development,independent,protocol):
    protocol_path=Path(protocol);plan=json.loads(protocol_path.read_text())
    reports=[];evidence={};sources=None;grasp_sha=None;failures=[]
    for split,folders in [('development',development),('independent',independent)]:
        for folder in folders:
            folder=Path(folder).resolve();experiment=json.loads((folder/'experiment.json').read_text())
            cfg=json.loads((folder/'runtime_config.json').read_text())
            result=json.loads((folder/'result.json').read_text())
            if sources is None:sources=experiment['source_sha256'];grasp_sha=experiment['grasp_config_sha256']
            if experiment['source_sha256']!=sources or experiment['grasp_config_sha256']!=grasp_sha:
                failures.append(f'configuration_changed:{folder.name}')
            for trial in result['trials']:
                seed=trial['seed'];trajectory=folder/f'trajectory_{seed}.json';rows=json.loads(trajectory.read_text())
                base=assess_pickup(rows,cfg);opposition=assess_opposition(rows)
                self_impulse=sum(r['self_contact_impulse'] for r in rows)
                accepted=base['accepted'] and opposition['passed'] and self_impulse<=1e-6
                arm=[cfg['joint_names'].index(f'Joint{i}_L') for i in range(1,8)]
                lift_rows=[r['nominal_target'] for r in rows if r['phase']=='primary_lift']
                if not lift_rows:
                    accepted=False;failures.append(f'primary_lift_missing:{seed}');travel=np.zeros(7)
                else:
                    lift=np.array(lift_rows)[:,arm]
                    travel=np.degrees(np.abs(np.diff(lift,axis=0)).sum(axis=0))
                if np.max(travel[4:])>1e-4:accepted=False;failures.append(f'distal_lift_target_moved:{seed}')
                reports.append({'split':split,'seed':seed,'accepted':bool(accepted),'source_run':str(folder),
                    'acceptance':base,'opposition':opposition,'max_height_m':max(r['height_m'] for r in rows),
                    'robot_table_impulse_sum_Ns':sum(r['robot_table_impulse'] for r in rows),
                    'self_contact_impulse_sum_Ns':self_impulse,'cube_table_contacts_final_3s':sum(r['cube_table_contact'] for r in rows[-90:]),
                    'primary_lift_nominal_arm_travel_deg':travel.tolist()})
                evidence[str(trajectory)]=sha(trajectory)
            for name in ['config.json','runtime_config.json','asset_contract.json','experiment.json','grasp_config.json','result.json']:
                evidence[str(folder/name)]=sha(folder/name)
    dev_seeds=[r['seed'] for r in reports if r['split']=='development']
    test_seeds=[r['seed'] for r in reports if r['split']=='independent']
    if sorted(dev_seeds)!=sorted(plan['development_seeds']):failures.append('development_protocol_incomplete')
    if sorted(test_seeds)!=sorted(plan['reserved_independent_seeds']):failures.append('independent_protocol_incomplete')
    if set(dev_seeds)&set(test_seeds):failures.append('development_test_overlap')
    for name,digest in (sources or {}).items():
        if sha(ROOT/name)!=digest:failures.append(f'current_source_changed:{name}')
    if not all(r['accepted'] for r in reports):failures.append('trial_failed')
    from side_gate import require_side_gate
    baseline=require_side_gate()
    evidence[str(protocol_path.resolve())]=sha(protocol_path)
    return {'passed':not failures,'kind':'state_based_traditional_thumb_opposition_teacher',
        'development_seeds':dev_seeds,'independent_seeds':test_seeds,'accepted':sum(r['accepted'] for r in reports),
        'total':len(reports),'failure_reasons':failures,'source_sha256':sources,'grasp_config_sha256':grasp_sha,
        'grasp_config_path':str(ROOT/plan['candidate']),'previous_independent_cohort':plan.get('previous_independent_cohort'),
        'base_config_sha256':sha(ROOT/'config.json'),'assets_sha256':baseline['assets_sha256'],
        'soft_limit_version':'revo3_user_soft_limits_v2','checkpoint':None,'trained_policy':False,
        'evidence_sha256':evidence,'trials':reports}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',nargs='+',required=True);p.add_argument('--independent',nargs='+',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.development,a.independent,a.protocol);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['passed','accepted','total','failure_reasons']}))
    raise SystemExit(0 if result['passed'] else 1)
