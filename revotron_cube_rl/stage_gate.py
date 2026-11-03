"""Fail closed: a passing integration smoke is never permission for task training."""
from pathlib import Path
from evidence_paths import local_path, verified_sha, verify_source
import hashlib
import json
ROOT=Path(__file__).resolve().parent

def sha256(path):
    return verified_sha(path,ROOT)

def code_identity():
    return {name:sha256(ROOT/name) for name in ('config.json','transfer_protocol.json','traditional_controller.py','environment.py','joint_safety.py','acceptance.py','isaac_compat.py')}

def require_staged_gate():
    path=ROOT/'staged_teacher_gate.json'
    if not path.exists():
        raise RuntimeError('Staged teacher safety gate has not passed. Run audited development and independent heldout evaluation first; PPO/DAgger remain blocked.')
    gate=json.loads(path.read_text())
    if not gate.get('passed') or gate.get('identity')!=code_identity():
        raise RuntimeError('Staged teacher certificate is failed or stale; revalidate the current code/config.')
    protocol=json.loads((ROOT/'transfer_protocol.json').read_text())
    if (gate.get('episodes',0)<protocol['gates']['minimum_heldout_episodes']
        or gate.get('accepted_episodes')!=gate.get('episodes')
        or set(gate.get('heldout_seeds',[]))!=set(protocol['heldout_seeds'])
        or not gate.get('evidence_sha256') or not gate.get('assets')):
        raise RuntimeError('Incomplete independent teacher evidence')
    for filename,digest in (gate['evidence_sha256']|gate['assets']).items():
        if sha256(filename)!=digest:raise RuntimeError('Teacher evidence has changed: '+filename)
    return gate


def require_visual_training_gate():
    require_staged_gate()
    path=ROOT/'state_single_face_gate.json'
    if not path.exists():
        raise RuntimeError('State-only single-face policy has not passed independent validation. The handoff requires this before visual BC/DAgger/PPO; visual training remains blocked.')
    gate=json.loads(path.read_text())
    if not gate.get('passed') or gate.get('observation_mode')!='privileged_state' or not gate.get('independent_seeds') or not gate.get('checkpoint_sha256'):
        raise RuntimeError('Missing independent state-policy evidence')
    if gate.get('config_sha256')!=sha256(ROOT/'config.json'):
        raise RuntimeError('State-policy config mismatch')
    if sha256(gate['checkpoint'])!=gate['checkpoint_sha256'] or not gate.get('evidence_sha256'):
        raise RuntimeError('State-policy checkpoint/evidence mismatch')
    for path,digest in gate['evidence_sha256'].items():
        if sha256(path)!=digest:raise RuntimeError('State-policy evidence changed')
    return gate
