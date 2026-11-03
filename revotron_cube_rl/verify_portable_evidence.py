"""Recompute published palm-direction evidence without reading its old paths."""
import argparse
import json
from pathlib import Path
from evidence_paths import local_path,raw_sha
from palm_direction_gate import build_certificate

ROOT=Path(__file__).resolve().parent


def verify(certificate='palm_direction_gate_certificate.json',protocol='palm_direction_protocol_v5.json'):
    cert_path=local_path(certificate,ROOT);cert=json.loads(cert_path.read_text())
    migration=json.loads((ROOT/'portability_migration.json').read_text())
    if raw_sha(cert_path)!=migration['original_certificates'][cert_path.name]:
        raise RuntimeError('Original certificate changed')
    if raw_sha(ROOT/'evidence_paths.py')!=migration['resolver_sha256']:
        raise RuntimeError('Path resolver changed')
    for name,entry in migration['validator_replacements'].items():
        if (raw_sha(ROOT/name)!=entry['replacement_sha256']
            or raw_sha(ROOT/entry['original_snapshot'])!=entry['original_sha256']):
            raise RuntimeError('Validator migration changed: '+name)
    if not cert['passed'] or cert.get('right_regrasp_validated'):
        raise RuntimeError('Ineligible historical scope')
    protocol_path=local_path(protocol,ROOT)
    if raw_sha(protocol_path)!=cert['protocol_sha256']:
        raise RuntimeError('Protocol hash mismatch')
    mappings=[];runs=[]
    for trial in cert['trials']:
        run=local_path(trial['run'],ROOT);runs.append(run)
        for name,key in [('config.json','config_sha256'),('runtime_config.json','runtime_config_sha256'),
                         ('asset_contract.json','asset_contract_sha256'),
                         (f"trajectory_{trial['seed']}.json",'trajectory_sha256')]:
            if raw_sha(run/name)!=trial[key]:raise RuntimeError('Changed evidence: '+str(run/name))
        mappings.append({'original':trial['run'],'resolved':str(run)})
    for path,digest in cert['assets_sha256'].items():
        actual=local_path(path,ROOT)
        if raw_sha(actual)!=digest:raise RuntimeError('Changed asset: '+str(actual))
        mappings.append({'original':path,'resolved':str(actual)})
    result=build_certificate(protocol_path,runs)
    if not result['passed']:raise RuntimeError('Recomputed evidence failed: '+str(result['failures']))
    return {'passed':True,'kind':'offline_relocation_reaudit','original_certificate_sha256':raw_sha(cert_path),
            'protocol_sha256':raw_sha(protocol_path),'path_mappings':mappings,
            'accepted':result['accepted'],'total':result['total'],
            'new_PhysX_episodes':0,'left_elbow_outward_validated':False,
            'right_regrasp_validated':False,'neural_policy_trained':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--certificate',default='palm_direction_gate_certificate.json')
    p.add_argument('--protocol',default='palm_direction_protocol_v5.json');p.add_argument('--out',required=True)
    a=p.parse_args();result=verify(a.certificate,a.protocol)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='path_mappings'}))
