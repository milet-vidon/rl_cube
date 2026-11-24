"""Branch a frozen parallel run at its last fully validated round, with a measured fit batch."""
import argparse
from pathlib import Path

import contact_hold_continuous as loop
import contact_hold_gpu_campaign as train
from contact_hold_dataset import sha
from contact_hold_loop_data import ROOT, checkpoint, inspect, read, write


def validated_panel(slot,model):
    p=read(slot/'panel.json')
    if [e['seed'] for e in p['episodes']]!=list(train.VAL):raise ValueError('Validation panel has wrong seeds')
    actual=loop.panel([inspect(Path(e['run']),model,e['seed'],'validation')[0] for e in p['episodes']])
    if actual!=p:raise ValueError('Validation panel disagrees with recorded physics')
    return actual


def trace(src,calibration,envs):
    settings=read(src/'settings.json');source_state=read(src/'state.json')
    if settings['code']!=train.identity() or settings['base']!=checkpoint(train.BASE) or settings['base']!=train.DIGEST:
        raise ValueError('Source campaign or code changed')
    cfg=settings['cfg'].copy();n=source_state['round']
    if not 1<=n<cfg['rounds']:raise ValueError('Need a completed validation round and at least one round remaining')
    if source_state['status'] in ('final_test','confirmed_on_test','heldout_target_not_met'):
        raise ValueError('Do not branch from a campaign that has selected its final model')
    if not cfg['envs']<=envs<=16 or cfg['rounds']*envs>9000:raise ValueError('Invalid lane count or overlapping seed schedule')
    cal=read(calibration);chosen=cal['selected'];pf=Path(cal['source_plan'])
    if not (cal['kind']=='contact_hold_real_fit_memory_calibration' and cal['steps_per_trial']>=3
            and pf.is_relative_to(src) and sha(pf)==cal['source_plan_sha256'] and cal['code']==train.identity()
            and cal['probe_sha256']==sha(ROOT/'contact_hold_vram_probe.py')
            and cal['parent_sha256']==read(pf)['parent_sha'] and chosen in cal['trials']
            and chosen['ok'] and chosen['within_budget'] and chosen['batch']>=8 and chosen['batch']%8==0
            and 512<=cal['headroom_mib'] and chosen['peak_device_mib']<=cal['total_mib']-cal['headroom_mib']):
        raise ValueError('Not a measured batch from this campaign and device budget')
    base=train.BASE;bank=[]
    s=dict(round=0,current=str(base),champion=str(base),stale=0,streak=0,status='running',
           best_score=validated_panel(base.parent/'validation',base)['score'])
    for k in range(1,n+1):
        rd=src/f'round_{k:03}';fit=rd/'fit';fp=fit/'plan.json';p=read(fp)
        if (p['parent']!=s['current'] or p['parent_sha']!=checkpoint(Path(s['current']))
                or p['code']!=train.identity() or p['vector_corpus'][:-1]!=bank or
                read(fit/'experiment.json')['plan_sha']!=sha(fp) or sha(fit/'last.pt')!=read(fit/'result.json')['last_sha256']):
            raise ValueError(f'Source fit lineage changed in round {k}')
        checkpoint(fit)
        entry=p['vector_corpus'][-1];_,r=train.verify_batch(entry)
        if r['checkpoint']!=p['parent_sha'] or Path(entry['path']).parent!=rd:
            raise ValueError('Collection does not belong to this parent and round')
        bank.append(entry);v=validated_panel(rd/'validation',fit);loop.advance(s,v,fit,cfg);s['last_rate']=v['rate']
    if bank!=source_state['vector_corpus'] or any(s[k]!=source_state[k] for k in
            ('round','current','champion','best_score','last_rate','streak','stale')):
        raise ValueError('Saved state disagrees with completed rounds')
    loop.ensure_unseen(train.TEST)
    cfg.update(batch=chosen['batch'],envs=envs)
    s.update(vector_corpus=bank,stage='branch_from_completed_validation',status='running',streak=0,stale=0)
    lineage=dict(kind='high_vram_branch',source=str(src),round=n,current=s['current'],parent_sha=checkpoint(Path(s['current'])),
                 champion=s['champion'],calibration=str(calibration),calibration_sha256=sha(calibration),
                 source_settings_sha256=sha(src/'settings.json'),previous_batch=settings['cfg']['batch'],
                 branch_source_sha256=sha(Path(__file__)),branch_launcher_sha256=sha(ROOT/'run_contact_hold_high_vram.sh'),
                 batch=cfg['batch'],envs=envs,remaining_rounds=cfg['rounds']-n,target_device_mib=cal['target_device_mib'],
                 measured_device_mib=chosen['peak_device_mib'])
    return cfg,s,lineage


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--preview',action='store_true',help='read evidence without stopping or starting a campaign')
    a=p.parse_args();src=a.from_run.resolve();dest=a.run_dir.resolve();cal=a.calibration.resolve()
    if dest.exists() or src.is_relative_to(dest) or dest.is_relative_to(src):p.error('choose a new sibling run directory')
    if a.preview:
        print(trace(src,cal,a.num_envs)[2],flush=True);return
    try:guard=loop.lock(src/'campaign.lock')
    except BlockingIOError:p.exit(2,'Source run is active. Wait for a completed PANEL, then stop its terminal with Ctrl+C; retry this command.\n')
    with guard:
        cfg,s,link=trace(src,cal,a.num_envs)
        from contact_hold_vram_probe import gpu
        name,total,_,_=gpu();receipt=read(cal)
        if name!=receipt['gpu'] or total!=receipt['total_mib']:raise ValueError('Rerun calibration for this GPU')
        dest.mkdir(parents=True,exist_ok=False)
        write(dest/'lineage.json',link);write(dest/'settings.json',dict(cfg=cfg,code=train.identity(),base=train.DIGEST));write(dest/'state.json',s)
        print('CONTINUE_FROM',s['current'],'round',s['round']+1,'batch',cfg['batch'],'envs',cfg['envs'],flush=True)
        train.campaign(dest,cfg,True)


if __name__=='__main__':main()
