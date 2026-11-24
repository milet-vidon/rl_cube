"""Run the real gated sampler (including per-lane IK) at a capacity-tested size before recommending it."""
import argparse,sys
from pathlib import Path
from contact_hold_wide_capacity import monitored
from contact_hold_vram_probe import slot
from contact_hold_loop_data import ROOT,read,write,checkpoint
from contact_hold_dataset import sha

SAMPLER=('contact_hold_wide_scene.py','contact_hold_wide_sample.py','contact_hold_wide_replay.py','contact_hold_wide_planner.py')
def sampler_code():return {p:sha(ROOT/p) for p in SAMPLER}

def qualify(src,capacity,out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    with slot() as fds:
        trials=read(capacity/'trials.json');code=sampler_code()
        candidates={r['n']:r for r in trials if r['ok'] and r['code']['contact_hold_wide_scene.py']==code['contact_hold_wide_scene.py']
                    and r['code']['contact_hold_wide_scene_probe.py']==sha(ROOT/'contact_hold_wide_scene_probe.py')
                    and r['peak_device_mib']<=r['total_mib']-768 and r['min_host_available_mib']>=6000}
        if not candidates:raise ValueError('No passing real capacity trial within the memory budget')
        parent=Path(read(src/'state.json')['current'])
        for n,r in sorted(candidates.items(),reverse=True):
            d=out/f'n{n}';d.mkdir();data=d/'samples';plan=d/'collect.json'
            write(plan,dict(parent=str(parent),parent_sha=checkpoint(parent),code=code,seeds=list(range(7_000_000,7_000_000+n))))
            print('QUALIFY_REAL_SAMPLER',n,flush=True)
            m=monitored([sys.executable,'-u',str(ROOT/'contact_hold_wide_sample.py'),'--plan',str(plan),'--output',str(data),'--steps','12'],d/'sample.log',fds,deadline_s=2000)
            write(d/'telemetry.json',m)
            if not m['ok']:
                if m['reason']=='device_or_host_memory_headroom':continue
                raise RuntimeError('Sampler trial failed; see '+str(d/'sample.log'))
            rep=read(data/'report.json')
            if m['peak_device_mib']>m['total_mib']-640 or m['min_host_available_mib']<5000:continue
            if rep['schema']!='wide_cuda_wrist_query_v2' or rep['samples']<=0 or rep['num_envs']!=n or rep['cross_lane_contacts']!=0 or rep['data_sha256']!=sha(data/'batch.npz'):raise ValueError('Invalid sampler evidence')
            result=dict(kind='wide_real_sampler_short_probe',n=n,code=code,capacity=str(capacity.resolve()),trials_sha=sha(capacity/'trials.json'),
                        plan_sha=sha(plan),report_sha=sha(data/'report.json'),n_steps=12,telemetry=m,resource_sha=sha(d/'telemetry.json'),
                        samples=rep['samples'],planner_s=rep['planner_init_s'],env_steps_per_s=sum(sum(ph.values()) for ph in rep['phase_steps'])/rep['elapsed_rollout_s'])
            write(out/'qualified.json',result);print('QUALIFIED',n,m['peak_device_mib'],result['env_steps_per_s'],flush=True);return
        raise RuntimeError('No sampler passed the reserved headroom')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('from-run','capacity','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();qualify(a.from_run,a.capacity,a.output)
