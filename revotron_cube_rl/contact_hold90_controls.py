"""Validated round-boundary configuration for the isolated 90 mm tunable engine."""
import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ENGINE='contact_hold90_tunable_v1'
def _field(kind,default,lo,hi,step=1):
    return dict(type=kind,default=default,min=lo,max=hi,step=step)
FIELDS={
    'rounds':_field('int',100,1,2000),'envs':_field('int',1536,4,2048,4),
    'val_envs':_field('int',32,4,256,4),'updates':_field('int',2000,1,100000),
    'batch':_field('int',32768,16,262144),'lr':_field('float',3e-5,1e-7,1e-3,1e-7),
    'replay_rounds':_field('int',3,1,16),'horizon':_field('int',3500,16,3500),
    **{k:_field('float',1.,.1,10.,.1) for k in ('grasp_weight','lift_weight','turn_weight')},
    **{k:_field('float',0.,0.,10.,.1) for k in ('hold_score','ending_score','turn_score')}
}
DEFAULTS={k:d['default'] for k,d in FIELDS.items()}
IMMUTABLE={'envs','val_envs','horizon'}
OBJECTIVE=('grasp_weight','lift_weight','turn_weight','hold_score','ending_score','turn_score')

def validate_params(params):
    if not isinstance(params,dict) or set(params)-set(FIELDS):raise ValueError('包含不支持的训练参数')
    p=DEFAULTS|params
    for k,v in p.items():
        f=FIELDS[k]
        if type(v) not in (float,int) or (f['type']=='int' and type(v)!=int) or not f['min']<=v<=f['max']:
            raise ValueError(f'{k} 超出范围或类型不正确：{f["min"]} .. {f["max"]}')
        if k in ('envs','val_envs') and v%4:raise ValueError(k+' 必须是 4 的倍数')
        if f['type']=='float':p[k]=float(v)
    return p

def _read(path):
    if path.is_symlink():raise ValueError('配置路径不能为符号链接')
    return json.loads(path.read_bytes())

def _save(path,data):
    tmp=path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with tmp.open('x') as f:
            json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

@contextmanager
def _guard(home):
    if home.is_symlink():raise ValueError('不接受符号链接实验')
    fd=os.open(home/'control.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX);yield
    finally:os.close(fd)

def _snapshot(home,settings):
    f=home/'control_state.json';head=_read(f) if f.is_file() else None
    if not head:raise ValueError('训练参数尚未初始化')
    latest=head['pending'] or head
    return dict(supported=True,effective=head['params'],effective_revision=head['revision'],
                revision=latest['revision'],pending=head.get('pending'),frozen_round=head['frozen_round'],completed=head['completed'])

def control_snapshot(home):
    home=Path(home);p=home/'settings.json'
    settings=_read(p) if p.is_file() else {}
    if settings.get('engine')!=ENGINE:
        return dict(supported=False,effective=None,effective_revision=None,revision=None,pending=None,frozen_round=0,completed=False)
    return _snapshot(home,settings)  # control_state is replaced atomically

def initialize(home,params):
    with _guard(home):
        p=home/'control_state.json'
        if p.exists():raise ValueError('此实验已有参数')
        _save(p,dict(params=validate_params(params),revision=uuid4().hex,pending=None,frozen_round=0,completed=False))

def queue_params(home,params,*,expected_revision=None):
    home=Path(home)
    if (home/'settings.json').is_symlink() or _read(home/'settings.json').get('engine')!=ENGINE:raise ValueError('旧版训练不支持在线修改')
    with _guard(home):
        p=home/'control_state.json';h=_read(p);old=h['pending'] or h
        if h['completed']:raise ValueError('已结束的实验不能修改')
        if expected_revision is not None and old['revision']!=expected_revision:raise ValueError('参数版本已变，请刷新再提交')
        if not isinstance(params,dict):raise ValueError('配置必须是对象')
        v=validate_params(old['params']|params)
        if any(h['params'][k]!=v[k] for k in IMMUTABLE):raise ValueError('环境数、验证数和控制步长仅能在新建实验时设置')
        if v['rounds']<max(1,h['frozen_round']):raise ValueError('总轮数不能小于已经开始的轮次')
        h['pending']=dict(revision=uuid4().hex,params=v,earliest_round=h['frozen_round']+1)
        _save(p,h);return _snapshot(home,None)

def freeze_round(home,k):
    """Atomically consume only at a new boundary. A resumed round always keeps its receipt."""
    with _guard(home):
        head=home/'control_state.json';h=_read(head);d=home/f'round_{k:03d}';receipt=d/'control.json'
        if d.is_symlink():raise ValueError('非法轮次目录')
        if receipt.exists():
            r=_read(receipt)
            if r['round']!=k or validate_params(r['params'])!=r['params']:raise ValueError('轮次配置错误')
            if h['frozen_round']==k:return r
            # Crash after writing the receipt but before updating control_state.
            if h['frozen_round']!=k-1:raise ValueError('轮次状态不一致')
            consumed=(h['pending'] or h)['revision']==r['revision']
            h.update(params=r['params'],revision=r['revision'],frozen_round=k,pending=None if consumed else h['pending'])
            _save(head,h);return r
        if h['frozen_round']!=k-1:raise ValueError('轮次配置或状态缺失')
        params=h['pending'] or h
        if k>params['params']['rounds']:
            h.update(params=params['params'],revision=params['revision'],completed=True,pending=None)
            _save(head,h);return None
        if h['completed']:raise ValueError('实验已结束')
        r=dict(round=k,revision=params['revision'],params=validate_params(params['params']))
        d.mkdir(exist_ok=True);_save(receipt,r)
        h.update(params=r['params'],revision=r['revision'],pending=None,frozen_round=k)
        _save(head,h);return r
