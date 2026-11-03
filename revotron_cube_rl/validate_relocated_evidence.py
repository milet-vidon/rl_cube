"""Copy published evidence, deny old-directory reads, and run negative checks."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
source=Path(__file__).resolve().parent;repo=source.parent
workspace=Path(tempfile.mkdtemp(prefix='rl_cube_relocated_'))
project=workspace/'revotron_cube_rl';project.mkdir()
for p in source.iterdir():
    if p.is_file() and p.suffix in ('.py','.json'):shutil.copy2(p,project/p.name)
shutil.copytree(source/'certificates',project/'certificates')
paths=set()
for n in ['staged_teacher_gate.json','side_teacher_gate.json','opposition_gate_certificate.json']:
    c=json.loads((source/n).read_text())
    for k in ['assets','assets_sha256','evidence_sha256']:
        paths.update(c.get(k,{}))
    if 'grasp_config_path' in c:paths.add(c['grasp_config_path'])
for original in paths:
    relative=Path(original).relative_to('/home/brainco/rl_cube')
    dest=workspace/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(repo/relative,dest)
c=json.loads((source/'palm_direction_gate_certificate.json').read_text())
for trial in c['trials']:
    relative=Path(trial['run']).relative_to('/home/brainco/rl_cube')
    shutil.copytree(repo/relative,workspace/relative,dirs_exist_ok=True)

# Separate process: clean import graph; every Python open into either original
# workspace is rejected before it can read any evidence. Conda runtime is allowed.
program=r'''
import sys,json,os
from pathlib import Path
project=Path(sys.argv[1]);sys.path.insert(0,str(project));denied=[]
repo=Path(sys.argv[2]);forbidden=[repo,Path('/home/brainco/rl_cube/revotron_cube_rl'),Path('/home/brainco/rl_cube/revotron_revo3'),Path('/home/brainco/rl_cube/isaac_mujoco_cube'),Path('/home/brainco/rl_cube/work')]
def no_old_reads(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if any(p.is_relative_to(x) for x in forbidden):
   denied.append(str(p));raise PermissionError('Old workspace inaccessible: '+str(p))
sys.addaudithook(no_old_reads)
from stage_gate import require_staged_gate
from side_gate import require_side_gate
from opposition_gate import require_opposition_gate
from verify_portable_evidence import verify
from palm_direction_gate import build_certificate
from evidence_paths import local_path
results=[]
for f in [require_staged_gate,require_side_gate,require_opposition_gate]:
 assert f()['passed'];results.append({'check':f.__name__,'passed':True})
result=verify();results.append({'check':'six_episodes_reaudit','passed':result['passed'],'accepted':result['accepted']})
cert=json.loads((project/'palm_direction_gate_certificate.json').read_text())
run=local_path(cert['trials'][0]['run'],project)
targets={'source':project/'palm_handover.py','config':run/'config.json',
 'runtime_config':run/'runtime_config.json','trajectory':run/f"trajectory_{cert['trials'][0]['seed']}.json",
 'asset':local_path(next(iter(cert['assets_sha256'])),project),
 'gate_parser':project/'palm_direction_gate.py','old_gate_snapshot':project/'certificates/legacy_source/side_gate.py','resolver':project/'evidence_paths.py','protocol':project/'palm_direction_protocol_v5.json','certificate':project/'palm_direction_gate_certificate.json'}
for label,path in targets.items():
 original=path.read_bytes()
 try:
  path.write_bytes(original+b' ')
  try:verify()
  except (ValueError,RuntimeError):results.append({'check':'tamper_'+label,'passed':True})
  else:raise AssertionError('Tampering passed: '+label)
 finally:path.write_bytes(original)
path=targets['trajectory'];original=path.read_bytes()
try:
 path.unlink()
 try:verify()
 except FileNotFoundError:results.append({'check':'missing_trajectory','passed':True})
 else:raise AssertionError('Missing trajectory passed')
 path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'0'*64+'\nsize 1000\n')
 try:verify()
 except RuntimeError:results.append({'check':'LFS_pointer','passed':True})
 else:raise AssertionError('LFS pointer passed')
finally:path.write_bytes(original)
# Duplicate/incomplete actual runs fail protocol reconciliation, independently
# of protocol-file hashing.
proto=project/'palm_direction_protocol_v5.json';runs=[local_path(t['run'],project) for t in cert['trials']]
assert not build_certificate(proto,runs[:-1])['passed'];results.append({'check':'incomplete_protocol','passed':True})
assert not build_certificate(proto,runs+[runs[0]])['passed'];results.append({'check':'duplicate_seed','passed':True})
assert not denied,denied
print(json.dumps({'passed':True,'workspace':str(project),'old_workspace_reads_attempted':denied,'checks':results,'offline_only':True}))
'''
env=os.environ.copy();env.pop('RL_CUBE_EVIDENCE_ROOT',None)
p=subprocess.run([sys.executable,'-c',program,str(project),str(repo)],env=env,text=True,capture_output=True)
print(p.stdout);print(p.stderr,file=sys.stderr)
args.out.write_text(p.stdout)
raise SystemExit(p.returncode)
