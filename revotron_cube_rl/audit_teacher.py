"""Run preregistered heldout trials and issue a hash-bound staged-teacher gate."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from stage_gate import ROOT,sha256,code_identity
from acceptance import assess_pickup

def main():
    p=argparse.ArgumentParser();p.add_argument('--development-run',required=True);p.add_argument('--name',required=True);p.add_argument('--visual',action='store_true');a=p.parse_args()
    protocol=json.loads((ROOT/'transfer_protocol.json').read_text());cfg=json.loads((ROOT/'config.json').read_text())
    identity=code_identity();dev=ROOT/'runs'/a.development_run
    development=json.loads((dev/'results.json').read_text())
    if {r['seed'] for r in development}!=set(protocol['development_seeds']):raise ValueError('Development seeds differ from preregistration')
    def validate_source(rows):
        for r in rows:
            if r['controller_sha256']!=identity['traditional_controller.py'] or r['config_sha256']!=identity['config.json'] or r['grasp']!='staged_parallel':raise ValueError('Controller/config mismatch')
    validate_source(development)
    if not all(r['acceptance']['accepted'] for r in development):raise ValueError('Development gate not passed')
    seeds=protocol['heldout_seeds']
    if len(set(seeds))<protocol['gates']['minimum_heldout_episodes'] or set(seeds)&set(protocol['development_seeds']):raise ValueError('Invalid heldout split')
    out=ROOT/'runs'/a.name
    if out.exists():raise ValueError('Use a fresh run name; do not overwrite evidence')
    # sweep owns the new run directory; snapshot the launch identity outside it.
    launch={'identity':identity,'development_run':str(dev),'heldout_seeds':seeds,'gpu_count':1,'num_envs':1,'checkpoint':None,'checkpoint_reason':'traditional teacher has no learned weights'}
    asset_root=ROOT.parent
    paths=[Path(os.environ['REVOTRON_USD_PATH']),asset_root/'revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf',Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT'])/'upstream/mujoco_cube-main/cube_3x3x3.xml',Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT'])/'upstream/mujoco_cube-main/assets/sticker.png']
    launch['assets']={str(f):sha256(f) for f in paths}
    (ROOT/'runs'/f'{a.name}_preregistered.json').write_text(json.dumps(launch,indent=2))
    params=protocol['teacher_parameters'];start=time.monotonic()
    cmd=[sys.executable,'-u',str(ROOT/'sweep_controller.py'),'--grasp',params['grasp'],'--widths',str(params['width']),'--pregrasp-scale',str(params['pregrasp_scale']),'--approach-height',str(params['approach_height']),'--seeds',','.join(map(str,seeds)),'--name',a.name]
    if a.visual:cmd.append('--visual')
    subprocess.run(cmd,check=True)
    trials=json.loads((out/'results.json').read_text());validate_source(trials)
    if identity!=code_identity():raise RuntimeError('Code changed during validation')
    evidence={str(dev/'results.json'):sha256(dev/'results.json'),str(out/'results.json'):sha256(out/'results.json')}
    recomputed=[]
    for r in trials:
        path=out/f"trajectory_{r['candidate']}_{r['seed']}.json"
        recomputed.append(assess_pickup(json.loads(path.read_text()),cfg)['accepted'])
        evidence[str(path)]=sha256(path)
    passed=len(trials)==len(seeds) and {r['seed'] for r in trials}==set(seeds) and all(recomputed)
    report={**launch,'passed':passed,'evidence_sha256':evidence,'accepted_episodes':sum(recomputed),'episodes':len(trials),'teacher_success_rate':sum(recomputed)/len(trials),'pure_policy_success_rate':None,'elapsed_s':time.monotonic()-start,'training_elapsed_s':0.,'checkpoint_sha256':None,'soft_limit_version':identity['joint_safety.py'],'scope':'five-seed staged pickup validation only; not a state-policy, turn or vision-policy certificate'}
    for f,h in launch['assets'].items():
        if sha256(f)!=h:raise RuntimeError('Asset changed during evaluation')
    (out/'independent_test.json').write_text(json.dumps(report,indent=2))
    (ROOT/'staged_teacher_gate.json').write_text(json.dumps(report,indent=2))
    print('STAGED_TEACHER_GATE',json.dumps(report),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
