"""Local, user-triggered PhysX jobs. Status is read-only; starts never queue for the GPU."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import uuid

from . import inference

_JOB = re.compile(r"[0-9a-f]{32}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SEEDS = (20_000_000,20_000_001,20_000_002,20_000_003)
_NAMES = ("contact_hold_gpu_training.lock","isaac_gui_evaluation.lock")
_CMD = """set -euo pipefail
source /home/brainco/rl_cube/activate_6_2.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python -B -u -m cube_dashboard.rollout_worker --model "$1" --output "$2"
"""


class RolloutBusy(RuntimeError):
    """Known contention, no job has been queued."""


class RolloutUnavailable(RuntimeError):
    """Could not establish exclusive ownership or validate a durable result."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strict(text):
    def forbidden(_):
        raise ValueError("无效的非有限值")
    return json.loads(text,parse_constant=forbidden)


def _safe(path):
    # All paths here are generated internally from an allowlisted model / opaque job ID.
    try: return path.resolve() == path
    except (OSError,RuntimeError): return False


def _read(path, limit):
    if not _safe(path): raise RolloutUnavailable("任务文件不能是链接")
    with path.open("rb") as src:
        st=os.fstat(src.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size>limit: raise RolloutUnavailable("任务文件超出预期")
        data=src.read(limit+1)
    if len(data)>limit: raise RolloutUnavailable("任务文件过大")
    return data


def _obj(path,limit=8192):
    obj=_strict(_read(path,limit))
    if not isinstance(obj,dict): raise RolloutUnavailable("任务记录格式无效")
    return obj


def _atomic(path,obj):
    tmp=path.parent/("."+uuid.uuid4().hex)
    try:
        with tmp.open("x") as f:
            json.dump(obj,f,ensure_ascii=False,allow_nan=False);f.write("\n")
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def _store(root):
    runs=root/"runs"
    if not _safe(runs) or not runs.is_dir():raise RolloutUnavailable("未找到可信的项目运行目录")
    folder=runs/"cube_dashboard_rollouts"
    if not _safe(folder):raise RolloutUnavailable("闭环记录目录不能是链接")
    return folder


def _locate(root,job_id):
    if not isinstance(job_id,str) or not _JOB.fullmatch(job_id):raise KeyError("未知的闭环实验")
    d=_store(root)/job_id
    if not d.is_dir() or not _safe(d):raise KeyError("未知的闭环实验")
    return d


def _locks():
    """(dev major,dev minor,ino) -> flock holder; absence reliable only on the host."""
    rows={}
    try:
        with open("/proc/locks",encoding="ascii") as src:
            for line in src:
                col=line.split()
                if len(col)<8 or col[1:3]!=["FLOCK","ADVISORY"]: continue
                hi,lo,num=col[5].split(":")
                rows[(int(hi,16),int(lo,16),int(num))]=int(col[4])
        complete=Path("/proc/1/comm").read_text().strip() in ("systemd","init")
    except (OSError,ValueError):
        complete=False
    return rows,complete


def _held(path,locked):
    if not _safe(path):raise RolloutUnavailable("检测到不可信锁路径")
    s=path.stat()
    if not stat.S_ISREG(s.st_mode): raise RolloutUnavailable("检测到无效的锁文件")
    return (os.major(s.st_dev),os.minor(s.st_dev),s.st_ino) in locked


def _global(root):
    return root.parent/"work/teacher_optimization_20261103"


def _campaign_check(root,table):
    rows,complete=table
    for lock in (root/"runs").glob("*/campaign.lock"):
        try:
            if _held(lock,rows):raise RolloutBusy(f"训练正在运行（{lock.parent.name}），本次不会启动或排队。")
        except FileNotFoundError: continue
    for other in _global(root).glob("*training*.lock"):
        if other.name not in _NAMES and _held(other,rows):raise RolloutBusy("检测到其他训练锁，本次不会启动或排队。")
    if not complete:raise RolloutUnavailable("当前进程无法完整查看主机训练锁，已禁用实时仿真。")


def _availability(root,table,check_global=True):
    try:
        _campaign_check(root,table)
        if check_global:
            for name in _NAMES:
                if _held(_global(root)/name,table[0]):
                    raise RolloutBusy("GPU 或 Isaac 正被其他任务占用，本次不会启动或排队。")
    except OSError as exc:
        raise RolloutUnavailable("无法核实主机的训练或 GPU 锁") from exc


def _reserve(stack,path):
    if not _safe(path):raise RolloutUnavailable("GPU 锁路径无效")
    try:
        f=stack.enter_context(os.fdopen(os.open(path,os.O_NOFOLLOW|os.O_RDONLY),"rb"))
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise RolloutUnavailable("GPU 锁文件无效")
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return f.fileno()
    except BlockingIOError as exc:
        raise RolloutBusy("另一项训练或仿真已取得 GPU，本次不会启动或排队。") from exc
    except OSError as exc:
        raise RolloutUnavailable("无法取得现有训练锁") from exc


def _check_gpu():
    try:
        r=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],
                         check=True,capture_output=True,text=True,timeout=4)
    except (OSError,subprocess.SubprocessError) as exc:
        raise RolloutUnavailable("无法核实 GPU 计算进程，本次不启动。") from exc
    if r.stdout.strip():raise RolloutBusy("GPU 上已有计算进程，本次不会启动或排队。")


def _last_lines(path,size=32768):
    if not _safe(path):return []
    try:
        with path.open("rb") as f:
            s=os.fstat(f.fileno())
            if not stat.S_ISREG(s.st_mode):return []
            offset=max(0,s.st_size-size);f.seek(offset)
            lines=f.read(size).splitlines()
            return lines[1 if offset else 0:]
    except OSError:return []


def _frames_so_far(d):
    for line in reversed(_last_lines(d/"frames.jsonl")):
        try:
            i=_strict(line)["step"]
            if type(i)==int and 0<=i<3500:return i+1
        except (KeyError,ValueError,TypeError):continue
    return 0


def _load_receipt(d,request):
    receipt=_obj(d/"worker_result.json",3_000_000)
    count=receipt.get("step_count")
    if (receipt.get("checkpoint_sha256")!=request["sha256"] or receipt.get("seed")!=_SEEDS[0] or
        receipt.get("panel_seeds")!=list(_SEEDS) or not isinstance(receipt.get("trajectory_sha256"),str) or
        not _SHA.fullmatch(receipt["trajectory_sha256"]) or type(count)!=int or not 0<=count<=3500):
        raise RolloutUnavailable("闭环结果与请求的模型、种子或长度不一致")
    return receipt


def _identity(pid):
    """Linux PID and start time including reboot identity; a re-used PID is not a live job."""
    if type(pid)!=int or pid<1:return None
    try:
        raw=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        if raw[0] in ("X","Z"):return None
        return {"pid":pid,"start_ticks":int(raw[19]),
                "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError,ValueError,IndexError):return None


def _summarize(d,table):
    req=_obj(d/"request.json")
    if (req.get("schema")!=1 or req.get("id")!=d.name or not _JOB.fullmatch(d.name) or
        type(req.get("model")) is not str or not isinstance(req.get("sha256"),str) or
        not _SHA.fullmatch(req["sha256"]) or req.get("seed")!=_SEEDS[0]):
        raise RolloutUnavailable("任务记录不完整")
    try: end=_obj(d/"exit.json")
    except FileNotFoundError: end={}
    n=_frames_so_far(d); reason=None; complete=(d/"worker_result.json").exists()
    if end and end.get("code")!=0:
        status="failed";reason=end.get("reason") or "仿真进程未正常退出"
    elif complete:
        try: r=_load_receipt(d,req); n=r["step_count"];status="completed"
        except (ValueError,KeyError,TypeError,RolloutUnavailable,OSError) as exc:
            status="failed";reason="无法核实返回的闭环结果："+str(exc)[:200]
    else:
        try:live=_held(d/"job.lock",table[0])
        except FileNotFoundError:live=False
        if not live and not end:
            try:
                process=_obj(d/"process.json")
                live=process==_identity(process.get("pid"))
            except (OSError,ValueError,RolloutUnavailable):pass
        if live:status="running"
        elif end:status="failed";reason="进程已退出，但没有完整轨迹报告。"
        elif table[1]:status="interrupted";reason="任务进程已退出，未产生完整轨迹。"
        else:status="unknown";reason="无法核实主机进程。"
    return {k:req.get(k) for k in ("id","model","label","seed","sha256","created_at")}|{
        "status":status,"frames":n,"reason":reason,"finished_at":end.get("finished_at")}


def rollout_status(project_root):
    root=Path(project_root).resolve()
    folder=_store(root);table=_locks()
    jobs=[]
    if folder.is_dir():
        # IDs are server-generated hex only; logs or malformed job dirs are never published.
        dirs=sorted((d for d in folder.iterdir() if _JOB.fullmatch(d.name) and _safe(d) and d.is_dir()),
                    key=lambda d:d.stat().st_mtime_ns,reverse=True)[:120]
        for d in dirs:
            try:jobs.append(_summarize(d,table))
            except (OSError,ValueError,TypeError,KeyError,RolloutUnavailable): continue
    jobs.sort(key=lambda x:x["created_at"] or "",reverse=True)
    active=next((x["id"] for x in jobs if x["status"]=="running"),None)
    reason=None
    try:_availability(root,table)
    except (RolloutBusy,RolloutUnavailable) as exc:reason=str(exc)
    if active:reason="已有闭环仿真在运行。"
    return dict(available=reason is None,reason=reason,active_job=active,jobs=jobs,
                seed=_SEEDS[0],panel_seeds=list(_SEEDS))


def _model(model_id):
    if not isinstance(model_id,str) or len(model_id)>160:raise ValueError("无效模型")
    choices={m["id"]:m for m in inference.catalog()["models"]}
    if model_id not in choices:raise KeyError("未知或未完成的模型")
    return choices[model_id]


def _spawn(root,model_id,d,fds,log):
    env=dict(os.environ,CUBE_DASH_GPU_FD=str(fds[0]),CUBE_DASH_ISAAC_FD=str(fds[1]))
    return subprocess.Popen(["/bin/bash","-c",_CMD,"cube-dashboard",model_id,str(d)],cwd=root,env=env,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            close_fds=True,pass_fds=tuple(fds),start_new_session=True)


def _watch(proc,d):
    rc=proc.wait()
    reason=None
    if rc:
        lines=[s.decode(errors="replace")[:250] for s in _last_lines(d/"worker.log") if s.strip()]
        reason=next((s for s in reversed(lines) if re.match(r"^\w*(?:Error|Exception):",s)),None)
        reason=f"仿真退出码 {rc}"+("："+reason if reason else "")
    _atomic(d/"exit.json",dict(code=rc,finished_at=_now(),reason=reason))


def start_rollout(project_root,model_id):
    root=Path(project_root).resolve()
    model=_model(model_id)
    folder=_store(root)
    with ExitStack() as stack:
        _availability(root,_locks())
        fds=[_reserve(stack,_global(root)/name) for name in _NAMES]
        # Close the check/acquire race; a subsequently started campaign will wait at gpu_slot.
        _availability(root,_locks(),check_global=False)
        _check_gpu()
        if shutil.disk_usage(root/"runs").free<2*1024**3:raise RolloutUnavailable("可用空间不足 2 GiB。")
        folder.mkdir(exist_ok=True)
        ident=uuid.uuid4().hex;d=folder/ident;d.mkdir(mode=0o700)
        req=dict(schema=1,id=ident,model=model_id,label=model["label"],sha256=model["sha256"],
                 seed=_SEEDS[0],created_at=_now())
        jf=stack.enter_context((d/"job.lock").open("xb"))
        fcntl.flock(jf,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(jf.fileno())
        _atomic(d/"request.json",req)
        try:
            with (d/"worker.log").open("xb") as log:proc=_spawn(root,model_id,d,fds,log)
        except OSError as exc:
            _atomic(d/"exit.json",dict(code=1,finished_at=_now(),reason="无法启动仿真进程"))
            raise RolloutUnavailable("无法启动仿真进程") from exc
        # Closing our copies on return leaves the flock open file descriptions with the child,
        # even if the web server or browser exits. No manager thread owns/renews those locks.
        threading.Thread(target=_watch,args=(proc,d),daemon=True).start()
        identity=_identity(proc.pid)
        if identity:
            try:_atomic(d/"process.json",identity)
            except OSError:pass
    return {k:req[k] for k in ("id","model","label","sha256","seed","created_at")}|dict(
        status="running",frames=0,reason=None,finished_at=None)


def rollout_frames(project_root,job_id):
    d=_locate(Path(project_root).resolve(),job_id)
    job=_summarize(d,_locks())
    if job["status"]!="completed":
        raise RolloutBusy(job["reason"] or "闭环尚未结束，完成后可回放。")
    r=_load_receipt(d,job)
    raw=_read(d/"frames.jsonl",32*1024**2)
    if sha256(raw).hexdigest()!=r["trajectory_sha256"]:
        raise RolloutUnavailable("回放文件与已完成仿真的轨迹哈希不符")
    frames=[]
    for i,line in enumerate(raw.splitlines()):
        f=_strict(line)
        if not isinstance(f,dict) or type(f.get("step"))!=int or f["step"]!=i or i>=3500:
            raise RolloutUnavailable("闭环轨迹存在缺帧或错序")
        frames.append(f)
    if len(frames)!=r["step_count"]:raise RolloutUnavailable("轨迹和报告长度不一致")
    return r|dict(job=job,frames=frames)
