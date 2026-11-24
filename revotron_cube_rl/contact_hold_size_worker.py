"""One development-only 90 mm transfer of the frozen round-15 from-reset student.

Use the scheduler's device leases. All source substitutions are local to the
native child, and their before/after hashes and diffs accompany the real run.
"""
import argparse
import ast
from contextlib import contextmanager, ExitStack
from difflib import unified_diff
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from unittest.mock import patch

import contact_hold_functional_worker as functional
from contact_hold_grace_worker import ROOT, RUNS, CHECKPOINT, sha

EDGE = .090
BASE_EDGE = .057
MODE = 'from_reset'
BASE = RUNS/'contact_hold_functional_r15_20261111_1945'/MODE
BASE_DIGEST = {21611:'008c776bcb47a1694ad230e7dabd5d6f1601c50fb0c79df7cb38bc83328e284e',
               21612:'b13f9e810ee0770c96862be62d8f35d8f25f573543269618d0f5edf9d00b2041'}
BASE_ASSET = 'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'
FROZEN = {'environment.py':'f42d386e0543df01183831ee10a90be5eab8f04e56472f7a060837bb61479bd7',
          'articulated_cube_env.py':'dfad0023f9e2797d6ea8ea3ac08aa1b6889b7dbb4ff8398cd0ca11fa329bc707',
          'solver_precision.py':'b147e8647ef0199192931926941813e19b099de0be02a76f6df640c3a7567c70',
          'contact_hold_functional_worker.py':'00bdd9213dcb0325da8b11f5ee042c5265eea08c6aaaea3ff46ef6d441d94308',
          'contact_hold_physical_metrics.py':'b4c422bf972bc7d655b7d192a01808cf5e8c8fe3769d2154972f072e34d64370',
          'contact_hold_student_eval.py':functional.HASHES['contact_hold_student_eval.py']}
# Only the pre-reset construction pose, reset Z and the center-to-table-height
# conversion depend on cube size. Preserve the original 1.5/2 mm clearances.
# The final init replacement prevents writing a temporary contract into ROOT.
REPLACEMENTS = {
    ('environment.py','CubePickupEnv','__init__'):[
        ("pos=tuple(c['cube_xyz'])", "pos=(*c['cube_xyz'][:2], c['cube_xyz'][2] + (.090-.057)/2)"),
        ("(ROOT/'asset_contract.json').write_text(","(self.run_dir/'initial_asset_contract.json').write_text(")],
    ('environment.py','CubePickupEnv','reset'):[("self.cfg['table_z']+.0305", "self.cfg['table_z']+.090/2+.002")],
    ('environment.py','CubePickupEnv','step'):[("self.cfg['table_z']+.0285", "self.cfg['table_z']+.090/2")],
    ('contact_hold_student_eval.py','StudentEvaluation','bind'):[
        ('if read(self.source/name) != read(out/name):', 'if not self.matches_size_domain(name, read(self.source/name), read(out/name)):')],
}


def file_sha(path):
    return sha(Path(path).read_bytes())


def method_set():
    methods, diffs = {}, {}
    for (file, cls, name), pairs in REPLACEMENTS.items():
        raw=(ROOT/file).read_text()
        c,=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name==cls]
        f,=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if f.decorator_list:raise ValueError('Unexpected runtime method decorator')
        orig=textwrap.dedent(''.join(raw.splitlines(keepends=True)[f.lineno-1:f.end_lineno]))
        edit=orig
        for before,after in pairs:
            if edit.count(before)!=1:raise ValueError('Different runtime method: '+cls+'.'+name)
            edit=edit.replace(before,after)
        key=cls+'.'+name
        methods[key]=compile(edit,'<size90:'+key+'>','exec')
        diffs[key]=dict(original=sha(orig.encode()), diagnostic=sha(edit.encode()),
                        diff=''.join(unified_diff(orig.splitlines(True),edit.splitlines(True),'reference','size90')))
    return methods,diffs


def matches_domain(name, original, current, asset_digest):
    if name!='cube_asset.json':return current==original
    return (original.get('source_sha256')==BASE_ASSET and asset_digest!=BASE_ASSET and
            current==dict(original,source_sha256=asset_digest))


def reference(seed):
    run=BASE/f'seed_{seed}'
    manifest=run/'artifact_manifest.json'
    if file_sha(manifest)!=BASE_DIGEST[seed]:raise ValueError('Wrong paired 57 mm reference')
    entries=json.loads(manifest.read_text())['sha256']
    def checked(name):
        if file_sha(run/name)!=entries[name]:raise ValueError('Changed baseline artifact: '+name)
        return json.loads((run/name).read_text())
    r=checked('result.json');a=checked('cube_asset.json');f=checked('functional_intervention.json')
    if not (r['seed']==seed and r['functional_mode']==MODE and r['checkpoint_sha256']==CHECKPOINT and
            a['source_sha256']==BASE_ASSET and r['diagnostic_only'] and not r['eligible_training_episode'] and
            r['full_episode'] and not r['execution_interrupted'] and not r['cube_pose_writes_during_control']):
        raise ValueError('Incorrect reference settings')
    return f


def prepare(seed, asset):
    from contact_hold_size_geometry import verify_asset
    for name,digest in FROZEN.items():
        if file_sha(ROOT/name)!=digest:raise ValueError('Changed source: '+name)
    _,_,ref= functional.prepare(MODE,seed)
    if ref!=reference(seed):raise ValueError('Functional contact protocol changed since 57 mm comparison')
    geometry_path=asset.parent/'geometry.json'
    geometry=json.loads(geometry_path.read_bytes())
    verify_asset(asset,geometry)
    if not math.isclose(geometry['edge_m'],EDGE,abs_tol=1e-9) or geometry['source_sha256']!=BASE_ASSET:
        raise ValueError('Expected exact 57 -> 90 mm geometric transfer')
    methods,diffs=method_set()
    rec=dict(kind='round15_fixed_grasp_size_transfer_v1',seed=seed,mode=MODE,source_edge_m=BASE_EDGE,target_edge_m=EDGE,
             paired_reference_run=str(BASE/f'seed_{seed}'), paired_reference_manifest_sha256=BASE_DIGEST[seed],
             asset_sha256=file_sha(asset),geometry_receipt_sha256=file_sha(geometry_path),source_asset_sha256=BASE_ASSET,
             checkpoint_sha256=CHECKPOINT,method_diffs=diffs,original_source_sha256=FROZEN,worker_sha256=file_sha(__file__),
             geometry_helper_sha256=file_sha(ROOT/'contact_hold_size_geometry.py'),diagnostic_only=True,eligible_training_episode=False,
             source_train_config='Unchanged. Original nominal spawn Z is shifted by the half-edge difference only at construction.',
             feedback='Frozen original grasp targets and joint limits; identical from_reset contact gates; measured impulses and poses.',
             table_clearance_at_reset_m=.002)
    return methods,rec


def install(methods, cls, name):
    ns={};exec(methods[cls.__name__+'.'+name],getattr(cls,name).__globals__,ns)
    return ns[name]


def simulate(out,seed,asset):
    methods,rec=prepare(seed,asset)
    import contact_hold_student_eval as adapter
    import solver_precision
    from contact_hold_size_geometry import verify_runtime_mass_properties
    original_cls=adapter.StudentEvaluation
    original_iterations=solver_precision.position_iterations
    actual_bind=install(methods,original_cls,'bind')

    class SizeStudent(original_cls):
        def __init__(self, training, opts, directory):
            super().__init__(training,opts,directory)
            if directory.resolve()!=out or opts.seed!=seed or self.digest!=CHECKPOINT:raise ValueError('Wrong size trial')
            path=directory/'size_intervention.json'
            path.write_text(json.dumps(rec,indent=2,allow_nan=False))
            for source,archive in ((asset,'size_cube_model.usda'),(asset.parent/'geometry.json','size_geometry.json'),
                    (Path(__file__),'size_worker_source.py'),(ROOT/'contact_hold_size_geometry.py','size_asset_helper_source.py')):
                shutil.copyfile(source,directory/archive)
            self.protocol.update(size_diagnostic=True,cube_edge_m=EDGE,size_intervention_sha256=file_sha(path))
        def matches_size_domain(self,name,baseline,actual):
            return matches_domain(name,baseline,actual,rec['asset_sha256'])
        def bind(self,env,directory):
            if env.dynamic_source.resolve()!=asset or file_sha(asset)!=rec['asset_sha256']:
                raise ValueError('Wrong live PhysX asset')
            actual_bind(self,env,directory)
            runtime=verify_runtime_mass_properties(env,edge_m=EDGE)
            (directory/'size_runtime.json').write_text(json.dumps(runtime,indent=2,allow_nan=False))
            self.protocol.update(size_runtime_sha256=file_sha(directory/'size_runtime.json'))
            (directory/'student_evaluation.json').write_text(json.dumps(self.protocol,indent=2))

    with ExitStack() as s:
        installed=False
        @contextmanager
        def size_domain_at_scene_creation(count,module=None):
            # This is the first supported hook after Isaac Kit loads its plugins;
            # keep the exact original solver context around scene construction.
            nonlocal installed
            from environment import CubePickupEnv
            if installed or count!=64 or module is not None:raise ValueError('Unexpected PhysX construction')
            installed=True
            for name in ('__init__','reset','step'):
                s.enter_context(patch.object(CubePickupEnv,name,install(methods,CubePickupEnv,name)))
            with original_iterations(count):yield
        s.enter_context(patch.dict(os.environ,{'ISAAC_MUJOCO_CUBE_ROOT':str(asset.parent)}))
        s.enter_context(patch.object(adapter,'StudentEvaluation',SizeStudent))
        s.enter_context(patch.object(solver_precision,'position_iterations',size_domain_at_scene_creation))
        functional.simulate(out,seed,MODE)


def finalize(out,seed,asset):
    _,rec=prepare(seed,asset)
    m=json.loads((out/'artifact_manifest.json').read_bytes())['sha256']
    def verified(name):
        if file_sha(out/name)!=m[name]:raise ValueError('Incomplete size archive: '+name)
        return json.loads((out/name).read_bytes())
    r=verified('result.json');s=verified('size_intervention.json');cube=verified('cube_asset.json');verified('size_runtime.json')
    if not (r['size_diagnostic'] and s==rec and r['cube_edge_m']==EDGE and r['size_intervention_sha256']==m['size_intervention.json']
            and r['size_runtime_sha256']==m['size_runtime.json'] and s['asset_sha256']==m['size_cube_model.usda']==cube['source_sha256']
            and s['geometry_receipt_sha256']==m['size_geometry.json'] and s['worker_sha256']==m['size_worker_source.py']
            and s['geometry_helper_sha256']==m['size_asset_helper_source.py']):raise ValueError('Different size domain during rollout')
    rows=verified('trajectory.json');cfg=verified('config.json')
    if not rows or any(abs(r['height_m']-(r['cube_position'][2]-cfg['table_z']-EDGE/2))>1e-7 for r in rows):
        raise ValueError('Post-step heights were not measured relative to the 90 mm tabletop zero')
    del rows
    functional.finalize(out,MODE,seed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',required=True,type=int,choices=BASE_DIGEST)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve();asset=a.asset.resolve();job=out.parent.parent
    if (job.parent!=RUNS or not re.fullmatch(r'contact_hold_size_r15_[A-Za-z0-9_]+',job.name) or
            out.parent.name!=MODE or out.name!=f'seed_{a.seed}' or out.exists() or asset!=job/'asset90/cube_model.usda'):
        p.error('Select a fresh runs/contact_hold_size_r15_<id>/from_reset/seed_<seed> and its asset90/cube_model.usda')
    if a.preflight:
        print(json.dumps(prepare(a.seed,asset)[1],indent=2));return
    if a.native_child:
        simulate(out,a.seed,asset);return
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--seed',str(a.seed),'--output',str(out),
                    '--asset',str(asset),'--native-child'],check=True,close_fds=False)
    finalize(out,a.seed,asset)


if __name__=='__main__':main()
