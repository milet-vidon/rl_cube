"""Constrained, clone-local resolution of immutable historical certificates.

Historical absolute names are identifiers, never filesystem fallbacks. Only the
published workspace components can be mapped. Runs may be relocated to
an explicitly selected RL_CUBE_EVIDENCE_ROOT (a project-shaped directory).
"""
import hashlib
import json
import os
from pathlib import Path

LEGACY_WORKSPACE=Path('/home/brainco/rl_cube')
COMPONENTS={'revotron_cube_rl','revotron_revo3','isaac_mujoco_cube','work'}


def raw_sha(path):
    path=Path(path)
    with path.open('rb') as f:
        if f.read(200).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise RuntimeError('Git LFS object not downloaded: '+str(path))
        f.seek(0)
        return hashlib.file_digest(f,'sha256').hexdigest()


def local_path(path, project_root, evidence_root=None):
    project=Path(project_root).resolve();workspace=project.parent
    evidence=Path(evidence_root or os.environ.get('RL_CUBE_EVIDENCE_ROOT',project)).resolve()
    p=Path(path)
    if '..' in p.parts:
        raise ValueError('Parent traversal is not an evidence identifier: '+str(p))
    if p.is_absolute():
        # An explicitly selected evidence root is a legitimate current path.
        if p.is_relative_to(evidence):
            target=p;allowed=evidence
        elif p.is_relative_to(workspace):
            relative=p.relative_to(workspace)
            if not relative.parts or relative.parts[0] not in COMPONENTS:
                raise ValueError('Path outside published components: '+str(p))
            target=p;allowed=workspace
        elif p.is_relative_to(LEGACY_WORKSPACE):
            relative=p.relative_to(LEGACY_WORKSPACE)
            if not relative.parts or relative.parts[0] not in COMPONENTS:
                raise ValueError('Unmapped historical path: '+str(p))
            target=workspace/relative;allowed=workspace
        else:
            raise ValueError('Unmapped absolute evidence path: '+str(p))
    else:
        target=(workspace/p) if p.parts and p.parts[0] in COMPONENTS else project/p
        allowed=workspace
    project_runs=project/'runs'
    if target.is_relative_to(project_runs):
        target=evidence/'runs'/target.relative_to(project_runs);allowed=evidence
    resolved=target.resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError('Evidence symlink escapes selected root: '+str(target))
    return resolved


def verified_sha(path, project_root):
    return raw_sha(local_path(path,project_root))


def verify_source(name,expected,project_root,certificate):
    """Only the pinned side-gate parser refactor can differ from legacy bytes.

All robot, control, configuration and audit sources remain exact matches.
The old certificate and old parser are retained byte-for-byte. This exception
is documented as an offline parser migration, never a new physical success.
"""
    root=Path(project_root);current=verified_sha(root/name,root)
    if current==expected:return
    if name!='side_gate.py':raise RuntimeError('Stale teacher source: '+name)
    m=json.loads((root/'portability_migration.json').read_text())
    entry=m['validator_replacements'][name]
    if (raw_sha(root/certificate)!=m['original_certificates'][certificate]
        or expected!=entry['original_sha256'] or current!=entry['replacement_sha256']
        or raw_sha(root/entry['original_snapshot'])!=expected
        or raw_sha(__file__)!=m['resolver_sha256']):
        raise RuntimeError('Invalid or stale parser migration: '+name)
