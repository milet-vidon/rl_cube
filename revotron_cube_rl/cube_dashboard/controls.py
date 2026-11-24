"""User-triggered controls for the 90 mm configurable trainer. GET never creates files or jobs."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import fcntl
import os
import re
import shutil
import stat
import subprocess
import threading
import uuid

from . import inference, rollouts as ro, training
import contact_hold90_controls as engine


class ControlConflict(RuntimeError):
    """Busy resource, incompatible run, or stale revision."""


class ControlUnavailable(RuntimeError):
    """Cannot safely inspect or launch on this host."""


_RUN = re.compile(r"contact_hold90_[A-Za-z0-9][A-Za-z0-9_-]{0,56}\Z")
_NEW = re.compile(r"contact_hold90_ui_[A-Za-z0-9][A-Za-z0-9_-]{0,53}\Z")
_ID = re.compile(r"[a-f0-9]{32}\Z")
_FIXED = engine.IMMUTABLE
# UI descriptions distinguish phase mass, real physical episode outcomes, and independent metrics.
_META = {
    "rounds":("schedule","目标总轮数","从第 1 轮开始累计；续训时填写希望达到的总轮数。"),
    "updates":("schedule","每轮更新次数","每轮拟合中优化器更新的次数。"),
    "envs":("sampling","采样并行环境","每轮采样环境数，4 的倍数；显存不足时引擎可缩小单波宽度。"),
    "val_envs":("sampling","独立验证环境","每轮固定验证环境数，4 的倍数；与训练采样分开。"),
    "horizon":("sampling","每条轨迹最大控制步","每个采样或验证环境运行的最大步数。"),
    "batch":("optimizer","训练批次大小","每次更新抽取的训练样本行数。"),
    "lr":("optimizer","学习率","Adam 对每次梯度更新使用的步长。"),
    "replay_rounds":("optimizer","回放轮数","拟合时使用最近多少轮已采集的训练数据。"),
    "grasp_weight":("phase","抓取阶段权重","相对其它阶段，提高或降低抓取示教行在批次中的采样占比。"),
    "lift_weight":("phase","抬升阶段权重","相对其它阶段，调整抬升示教行的采样占比。"),
    "turn_weight":("phase","转动阶段权重","相对其它阶段，调整转动示教行的采样占比。"),
    "hold_score":("reward","曾经拿稳奖励","有过拿稳的采样轨迹，其模仿损失额外加权；0 关闭该项。"),
    "ending_score":("reward","结束稳持奖励","结束时仍稳持的采样轨迹，其模仿损失额外加权；0 关闭该项。"),
    "turn_score":("reward","整颗转动奖励","训练采样中连续稳持 1 秒且中段整颗净转达到 15°，额外加权该种子的模仿损失；不是拧动单层。0 关闭。"),
}


def _schema():
    out=[]
    for key, field in engine.FIELDS.items():
        group,label,help_text=_META[key]
        out.append({k:field[k] for k in ("type","min","max","step")} |
                   dict(key=key,group=group,label=label,help=help_text,default=engine.DEFAULTS[key],mutable=key not in _FIXED))
    return out


def _name(value):
    if not isinstance(value,str) or not _RUN.fullmatch(value):
        raise ValueError("名称须以 contact_hold90_ 开头，后接 1–57 位英文字母、数字、下划线或连字符，首位须为字母或数字。")
    return value


def _new_name(value):
    if not isinstance(value,str) or not _NEW.fullmatch(value):
        raise ValueError("新实验名须以 contact_hold90_ui_ 开头，后接 1–54 位英文字母、数字、下划线或连字符，首位须为字母或数字。")
    return value


def _known_home(root,name):
    _name(name)
    p=root/"runs"/name
    if not ro._safe(p) or not p.is_dir():raise FileNotFoundError("没有这个训练实验")
    if not ro._safe(p/"settings.json") or not ro._safe(p/"state.json"):raise ControlUnavailable("训练目录来源不可信")
    return p


def _store(root):
    p=root/"runs"/"cube_dashboard_controls"
    if not ro._safe(p) or not p.parent.is_dir():raise ControlUnavailable("未找到可信的面板记录目录")
    return p


def _read(path):
    try:return ro._obj(path,262144)
    except FileNotFoundError:return None
    except (ValueError,TypeError,ro.RolloutUnavailable) as e:raise ControlUnavailable("面板配置文件损坏或超出大小限制") from e


def _child(folder,name,create=False):
    p=folder/name
    if not ro._safe(p):raise ControlUnavailable("不能使用链接作为面板数据目录")
    if create:p.mkdir(exist_ok=True)
    return p


def _draft(root,run,write=False):
    folder=_store(root)
    if write:folder.mkdir(exist_ok=True)
    path=_child(folder,"drafts",write)/((run or "_new")+".json")
    if not ro._safe(path):raise ControlUnavailable("草稿来源无效")
    return path


def _params(v):
    if not isinstance(v,dict) or set(v)!=set(engine.DEFAULTS):raise ValueError("请提交全部训练参数，不能增加或遗漏字段。")
    return engine.validate_params(v)


def _models(root):
    # This API is bound to the project whose dashboard is running, never another project by name.
    if root/"runs"!=inference.RUNS:return {}
    return dict(inference._model_cards())


def _model(cards,mid,verify=False):
    if mid is None:mid=inference.ARCHIVE_MODEL
    if not isinstance(mid,str) or mid not in cards:raise ValueError("请选择已完成且来源可核实的起始模型")
    card,path=cards[mid]
    if verify:
        try:inference._verified(path,card["sha256"],64*1024**2)
        except inference.InferenceUnavailable as e:raise ControlUnavailable("起始权重校验失败："+str(e)) from e
    return card,path


def _compat_snapshot(root,run):
    home=_known_home(root,run)
    settings,_,entry=training._parts(home)
    values=dict(engine.DEFAULTS)
    values.update({k:settings[k] for k in values if k in settings})
    if settings.get("engine")!=engine.ENGINE:
        return values, None, None, dict(**entry,online=False,can_apply=False,can_resume=False,
            reason="原版长训没有在线调参入口。可复制这些参数和已完成轮的模型，另开可调实验。")
    s=engine.control_snapshot(home)
    if not s.get("supported"):raise ControlUnavailable("无法核实可调引擎的配置")
    pend=s.get("pending")
    if pend:pend=dict(values=pend["params"],revision=pend["revision"],earliest_round=pend["earliest_round"])
    completed=s["completed"]
    return (s["effective"],pend,s.get("effective_revision"),dict(**entry,online=True,
            can_apply=not completed,can_resume=entry["lock_held"] is False and not completed,
            reason=("该实验已结束，可复制参数并选取已完成的权重，新开训练。" if completed else None),
            frozen_round=s.get("frozen_round"),engine=engine.ENGINE))


def _availability(root):
    try:ro._availability(root,ro._locks())
    except ro.RolloutBusy as e:raise ControlConflict(str(e)) from e
    except ro.RolloutUnavailable as e:raise ControlUnavailable(str(e)) from e


def _job(d):
    q=_read(d/"request.json")
    if not q or not _ID.fullmatch(d.name) or q.get("id")!=d.name or not _RUN.fullmatch(str(q.get("run",""))):return None
    done=_read(d/"exit.json")
    state=training._load(d.parents[2]/q["run"]/"state.json",d.parents[2]/q["run"])
    p=_read(d/"process.json")
    live=bool(p and ro._identity(p.get("pid"))==p)
    if done and done.get("code")!=0:status="failed";reason=done.get("reason")
    elif live:status="running" if state else "starting";reason=None
    elif state.get("status")=="completed":status="completed";reason=None
    elif done:status="interrupted";reason="进程已退出，可在实验配置中查看状态。"
    else:status="interrupted";reason="启动进程已退出，没有完成记录。"
    return dict(id=d.name,run=q["run"],status=status,reason=reason,created_at=q["created_at"],action=q["action"])


def _jobs(root):
    folder=_child(_store(root),"jobs")
    if not folder.is_dir():return []
    jobs=[]
    ds=sorted((d for d in folder.iterdir() if _ID.fullmatch(d.name) and ro._safe(d) and d.is_dir()),
              key=lambda d:d.stat().st_mtime_ns,reverse=True)
    for d in ds[:40]:
        try:
            row=_job(d)
            if row:jobs.append(row)
        except (OSError,KeyError,TypeError,ValueError,ControlUnavailable):pass
    return jobs


def control_status(project_root, run=None):
    root=Path(project_root).resolve()
    _store(root)
    if run:
        values,pend,rev,entry=_compat_snapshot(root,run)
    else:values,pend,rev,entry=dict(engine.DEFAULTS),None,None,None
    reason=None
    try:
        _availability(root)
        ro._check_gpu()  # A non-project CUDA process may be present without a campaign lock.
    except (ControlConflict,ControlUnavailable,ro.RolloutBusy,ro.RolloutUnavailable) as e:reason=str(e)
    jobs=_jobs(root)
    active=next((j for j in jobs if j["status"] in ("starting","running")),None)
    if active:reason="训练正在运行（"+active["run"]+"）。"
    cards=_models(root)
    if entry and reason:entry["can_resume"]=False
    return dict(schema=_schema(),defaults=dict(engine.DEFAULTS),effective=values,pending=pend,effective_revision=rev,
                draft=_read(_draft(root,run)),run=entry,available=reason is None,reason=reason,
                jobs=jobs,parents=[card for card,_ in cards.values()],
                suggested_name="contact_hold90_ui_"+datetime.now().strftime("%Y%m%d_%H%M%S"))


def _queue(home,values,rev):
    try:return engine.queue_params(home,values,expected_revision=rev)
    except (RuntimeError,BlockingIOError) as e:raise ControlConflict(str(e)) from e
    except ValueError as e:
        if "版本" in str(e) or "已结束" in str(e):raise ControlConflict(str(e)) from e
        raise


def _spawn(root,argv,fd,log):
    # The child holds this dispatch guard through conda activation and trainer initialization.
    return subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            pass_fds=(fd,),close_fds=True,start_new_session=True)


def _watch(proc,d):
    code=proc.wait();message=None
    if code:
        lines=[s.decode(errors="replace")[:240] for s in ro._last_lines(d/"launcher.log") if s.strip()]
        message=f"训练启动器退出码 {code}"+("："+lines[-1] if lines else "")
    ro._atomic(d/"exit.json",dict(code=code,reason=message,finished_at=ro._now()))


def _launch(root,action,name,values,mid,rev):
    folder=_store(root);home=root/"runs"/name
    with ExitStack() as stack:
        _availability(root)
        # Same nonblocking global lock pair used by independent rollouts / normal GPU stages.
        # Hold until after child inherits the campaign dispatch guard; never queue another trainer via UI.
        try:
            for n in ro._NAMES:ro._reserve(stack,ro._global(root)/n)
            ro._availability(root,ro._locks(),check_global=False)
            ro._check_gpu()
        except ro.RolloutBusy as e:raise ControlConflict(str(e)) from e
        except ro.RolloutUnavailable as e:raise ControlUnavailable(str(e)) from e
        if shutil.disk_usage(root/"runs").free<2*1024**3:raise ControlUnavailable("可用空间不足 2 GiB。")
        folder.mkdir(exist_ok=True)
        guard=folder/"campaign.lock"
        if not ro._safe(guard):raise ControlUnavailable("无效的启动锁")
        gf=stack.enter_context(os.fdopen(os.open(guard,os.O_NOFOLLOW|os.O_CREAT|os.O_RDWR,0o600),"r+b"))
        if not stat.S_ISREG(os.fstat(gf.fileno()).st_mode):raise ControlUnavailable("无效的启动锁")
        try:fcntl.flock(gf,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise ControlConflict("另一个训练启动请求已受理。") from e
        if not ro._safe(home):raise ValueError("训练目录不能是符号链接")
        pcard=None;parent=None
        if action=="start":
            if home.exists():raise ControlConflict("同名实验已存在；请选择新名称，或到原实验续训。")
            pcard,parent=_model(_models(root),mid,verify=True)
        else:
            effective,pending,effective_revision,entry=_compat_snapshot(root,name)
            if not entry["online"]:raise ControlConflict(entry["reason"])
            if not entry["can_resume"]:raise ControlConflict("当前无法独占该实验，不能续训。")
            if values["rounds"]<=entry["completed_round"]:raise ValueError("续训的目标总轮数须大于已完成轮数。")
            if rev is not None and rev!=(pending["revision"] if pending else effective_revision):
                raise ControlConflict("配置已被修改，请刷新后重试。")
            if values!=(pending["values"] if pending else effective):_queue(home,values,rev)
        jobs=_child(folder,"jobs",True);uid=uuid.uuid4().hex;d=jobs/uid;d.mkdir(mode=0o700)
        q=dict(id=uid,run=name,action=action,created_at=ro._now(),parent=pcard and pcard["id"],parent_sha256=pcard and pcard["sha256"])
        ro._atomic(d/"request.json",q)
        script=root/"run_contact_hold90_tunable.sh"
        argv=["/bin/bash",str(script),"--run-dir",str(home)]
        if action=="start":
            ro._atomic(d/"config.json",values);argv+=["--control-config",str(d/"config.json"),"--parent-sha",pcard["sha256"]]
            if parent and pcard["id"]!=inference.ARCHIVE_MODEL:argv+=["--parent",str(parent.parent)]
        else:argv+=["--resume"]
        try:
            with (d/"launcher.log").open("xb") as log:proc=_spawn(root,argv,gf.fileno(),log)
        except OSError as e:
            ro._atomic(d/"exit.json",dict(code=1,reason="无法启动训练进程",finished_at=ro._now()))
            raise ControlUnavailable("无法启动训练进程") from e
        ident=ro._identity(proc.pid)
        if ident:ro._atomic(d/"process.json",ident)
        threading.Thread(target=_watch,args=(proc,d),daemon=True).start()
    return dict(message="已发起训练，最新阶段可在训练监控查看。",run=name,id=uid,status="starting")


def control_action(project_root,action,data):
    root=Path(project_root).resolve()
    if action not in {"draft","apply","start","resume"}:raise ValueError("无效操作")
    optional={"draft":{"name","parent"},"apply":{"expected_revision"},"start":{"parent"},"resume":{"expected_revision"}}[action]
    if not isinstance(data,dict) or not {"run","values"}<=set(data) or set(data)-{"run","values"}-optional:
        raise ValueError("无效的配置请求")
    run=data["run"];values=_params(data["values"]);parent=data.get("parent");rev=data.get("expected_revision")
    if run is not None:_name(run)
    if run is None and action!="draft":raise ValueError("请选择实验")
    if rev is not None and (not isinstance(rev,str) or not 1<=len(rev)<=128):raise ValueError("配置版本无效")
    if action=="draft":
        if run is not None:_known_home(root,run)
        name=data.get("name")
        if name:_new_name(name)
        elif name is not None and name!="":raise ValueError("草稿名称无效")
        if parent is not None:_model(_models(root),parent)
        rec=dict(values=values,updated_at=ro._now(),parent=parent,name=name)
        ro._atomic(_draft(root,run,True),rec)
        return dict(message="草稿已保存；训练参数尚未更改。",run=run,draft=rec)
    if action=="apply":
        _,_,_,entry=_compat_snapshot(root,run)
        if not entry["can_apply"]:raise ControlConflict(entry["reason"] or "不能更新此实验")
        snap=_queue(_known_home(root,run),values,rev)
        p=snap.get("pending")
        return dict(message="参数已提交，当前已冻结轮次保持原配置。" if p else "配置已是所选值。",
                    run=run,revision=(p or {}).get("revision",snap.get("effective_revision")),
                    earliest_round=(p or {}).get("earliest_round"))
    if action=="start":_new_name(run)
    return _launch(root,action,run,values,parent,rev)
