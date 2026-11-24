"""Additive remote telemetry source. Never writes local runs or launches jobs."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import os
import re
import subprocess
import threading
import time
from .remote_reader import NAME

CONFIG = Path('/home/brainco/rl_cube/work/remote_training_access/source.json')
LOCK = threading.Lock()
CACHE = {}


def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def config():
    p=Path(os.environ.get('CUBE_DASH_REMOTE_CONFIG',str(CONFIG)))
    if not p.is_file():return None
    d=json.loads(p.read_text())
    if (not re.fullmatch(r'[A-Za-z0-9.-]{1,200}',d['host']) or
        not re.fullmatch(r'[a-z_][a-z0-9_-]{0,63}',d['user'])):
        raise ValueError('Invalid remote source configuration')
    for key in ('identity_file','known_hosts_file'):
        if not Path(d[key]).is_absolute() or not Path(d[key]).is_file():raise ValueError('Remote connection files unavailable')
    return d


def fetch_remote(cfg,command):
    args=['ssh','-F','/dev/null','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
          '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ControlMaster=no',
          '-o','ConnectTimeout=4','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=1',
          '-o','UserKnownHostsFile='+cfg['known_hosts_file'],'-o','GlobalKnownHostsFile=/dev/null',
          '-i',cfg['identity_file'],cfg['user']+'@'+cfg['host'],command]
    result=subprocess.run(args,capture_output=True,text=True,timeout=9,check=True)
    if len(result.stdout)>2*1024*1024:raise ValueError('Remote response exceeded size limit')
    data=json.loads(result.stdout)
    if data.get('schema')!='cube_remote_monitor_v1' or not isinstance(data.get('data'),dict):
        raise ValueError('Unexpected remote telemetry format')
    return data


def remote_snapshot(run=None):
    if run is not None and (not isinstance(run,str) or not NAME.fullmatch(run)):
        raise ValueError('无效的远端实验名称')
    command='catalog' if run is None else 'snapshot '+run
    cfg=config()
    if cfg is None:
        return dict(source=None,connection=dict(ok=False,stale=False,error='尚未配置远端训练连接'),runs=[])
    source={k:cfg[k] for k in ('id','label','host')}
    key=(cfg['id'],cfg['host'],command)
    with LOCK:
        entry=CACHE.get(key,{})
        ttl=12 if not entry.get('ok',True) else 10 if run is None else 5
        if not entry or time.monotonic()-entry['checked_mono']>=ttl:
            try:
                payload=fetch_remote(cfg,command)
                entry=dict(payload=payload,ok=True,last_success_at=now(),success_mono=time.monotonic())
            except (OSError,subprocess.SubprocessError,ValueError,TypeError,KeyError):
                entry=dict(entry,ok=False,error='远端暂时无法连接；请检查主机和网络状态。')
            entry.update(checked_mono=time.monotonic(),checked_at=now())
            CACHE[key]=entry
            if len(CACHE)>32:
                oldest=min(CACHE,key=lambda k:CACHE[k]['checked_mono'])
                if oldest!=key:CACHE.pop(oldest,None)
        payload=entry.get('payload',{})
        result=copy.deepcopy(payload.get('data',{}))
        result.update(source=source,remote_host=payload.get('host'),remote_sampled_at=payload.get('sampled_at'),
            connection=dict(ok=entry['ok'],stale=not entry['ok'] and bool(payload),
                error=entry.get('error') if not entry['ok'] else None,last_success_at=entry.get('last_success_at'),
                checked_at=entry['checked_at'],age_seconds=round(time.monotonic()-entry['success_mono'],1) if payload else None))
        if run is None:result.setdefault('runs',[])
        return result
