"""Read training progress and the last real worker milestone without importing Isaac or PyTorch."""
import argparse
import json
from collections import Counter
from pathlib import Path

def status(run):
    s=json.loads((run/'state.json').read_text())
    cfg=json.loads((run/'settings.json').read_text())['cfg']
    parent=Path(s['current']); result=json.loads((parent/'result.json').read_text())
    print(f"round validated: {s['round']}/{cfg['rounds']} | model update: {result['global_update']} | validation: {s['last_rate']:.0%}")
    print(f"status: {s['status']} | stage: {s['stage']} | environments/round: {cfg['envs']} | parallel ceiling: {s.get('effective_envs',cfg['envs'])}")
    rd=run/f"round_{s['round']+1:03}"
    waves=sorted(rd.glob('accepted_*.json'))
    if waves:
        n=steps=rows=0; phases=Counter()
        for ptr in waves:
            p=Path(json.loads(ptr.read_text())['path']);r=json.loads((p/'report.json').read_text())
            n+=r['num_envs'];rows+=r['samples']
            for lane in r['phase_steps']:steps+=sum(lane.values());phases.update(lane)
        print(f"accepted environments: {n}/{cfg['envs']}, rows before round cap: {rows}, active environment steps: {steps}")
        print('phase counts across accepted rollouts (active environment steps):',dict(phases))
    logs=list(rd.glob('wave_*/*.log'))
    if logs:
        log=max(logs,key=lambda p:p.stat().st_mtime)
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace').splitlines()
        v=next((v for v in reversed(tail) if v.startswith(('VECTOR_STEP','PLAN_LANES','GPU_COLLECTION_COMPLETE','NATIVE_CLONES')) or '[Error]' in v),None)
        print('latest worker:',v or 'starting Isaac')
        print('log:',log)
    if s.get('backoffs'):
        print('last capacity retry:',s['backoffs'][-1])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('--gpu',action='store_true',help='Read current GPU 0 utilization, memory and power')
    a=ap.parse_args();status(a.run)
    if a.gpu:
        import subprocess
        subprocess.run(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv'],check=True)

