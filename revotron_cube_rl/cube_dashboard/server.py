"""Loopback-only UI. Training setup, launch and closed-loop evaluation require separate explicit actions."""
import argparse
import csv
import json
from datetime import datetime, timezone
import logging
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PROJECT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("static")
ASSETS = {"index.html": "text/html", "app.js": "text/javascript", "quality.js": "text/javascript", "compare.js": "text/javascript", "app.css": "text/css", "controls.js": "text/javascript", "controls.css": "text/css",
          "remote.js":"text/javascript", "remote.css":"text/css",
          "inference.js": "text/javascript", "inference.css": "text/css", "rollouts.js":"text/javascript", "rollouts.css":"text/css", "cube.svg": "image/svg+xml"}
HW_LOCK = threading.Lock()
GPU_CACHE = (0., None)
PREDICT_SLOT = threading.BoundedSemaphore(1)
LOG = logging.getLogger("cube_dashboard")

def hardware():
    global GPU_CACHE
    with HW_LOCK:
        now = time.monotonic()
        if GPU_CACHE[1] is not None and now-GPU_CACHE[0] < 5:
            return GPU_CACHE[1]
        result = {"available": False}
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True,
                               timeout=2, check=True)
            name, util, used, total = (v.strip() for v in next(csv.reader(p.stdout.splitlines())))
            result = dict(available=True, name=name, util_percent=int(util), used_mib=int(used), total_mib=int(total))
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
            pass
        result.update(scope="device", device_index=0,
                      sampled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if not result["available"]:
            result.update(util_percent=None, used_mib=None, total_mib=None)
        GPU_CACHE = (now, result)
        return result

class APIError(Exception):
    def __init__(self, status, message):
        self.status,self.message = status,message

class Handler(BaseHTTPRequestHandler):
    server_version = "CubeLab/1"
    def setup(self):
        super().setup()
        self.connection.settimeout(10)
    def log_message(self, fmt, *args):
        if getattr(self, "path", "").split("?")[0] not in ("/api/hardware","/api/training","/api/quality","/api/compare","/api/runs","/api/remote/runs","/api/remote/training"):
            super().log_message(fmt, *args)
    def send_data(self, code, typ, data):
        self.send_response(code)
        self.send_header("Content-Type",typ+"; charset=utf-8")
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Referrer-Policy","no-referrer")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command!="HEAD":
            self.wfile.write(data)
    def send_json(self,code,obj):
        self.send_data(code,"application/json",json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode())
    def local(self):
        port=self.server.server_port
        allowed={"localhost:"+str(port),"127.0.0.1:"+str(port)}
        if port==80: allowed.update({"localhost","127.0.0.1"})
        if self.headers.get("Host","").lower() not in allowed:
            raise APIError(403,"只接受本机浏览器访问。")
        origin=self.headers.get("Origin")
        if origin and origin.lower() not in {"http://"+v for v in allowed}:
            raise APIError(403,"不接受跨站请求。")
    def dispatch(self):
        self.local()
        u=urlsplit(self.path)
        q=parse_qs(u.query, max_num_fields=8)
        def one(name):return q.get(name,[""])[0]
        route=u.path
        if self.command=="POST":
            actions={"/api/training/control/"+a:a for a in ("draft","apply","start","resume")}
            if route not in actions and route not in ("/api/inference/predict","/api/rollouts"):raise APIError(404,"没有这个接口。")
            if self.headers.get_content_type()!="application/json":raise APIError(415,"需要 JSON 请求。")
            try:n=int(self.headers.get("Content-Length","0"))
            except ValueError:raise APIError(400,"请求长度无效。")
            if not 0<n<=(16384 if route in actions else 4096):raise APIError(400,"请求长度无效。")
            try:data=json.loads(self.rfile.read(n))
            except (UnicodeError,ValueError):raise APIError(400,"无效的 JSON。")
            if route in actions:
                from .controls import control_action, ControlConflict, ControlUnavailable
                if not isinstance(data,dict):raise APIError(400,"需要 JSON 对象。")
                action=actions[route]
                try:result=control_action(PROJECT,action,data)
                except ControlConflict as e:raise APIError(409,str(e))
                except ControlUnavailable as e:raise APIError(503,str(e))
                self.send_json(202 if action in ("start","resume") else 200,result);return
            if route=="/api/rollouts":
                if not isinstance(data,dict) or set(data)!={"model"} or not isinstance(data["model"],str) or not 0<len(data["model"])<=180:
                    raise APIError(400,"请选择一个有效的模型。")
                from .rollouts import start_rollout,RolloutBusy,RolloutUnavailable
                try:job=start_rollout(PROJECT,data["model"])
                except RolloutBusy as e:raise APIError(409,str(e))
                except RolloutUnavailable as e:raise APIError(503,str(e))
                self.send_json(202,job);return
            if not isinstance(data,dict) or set(data)!={"model","episode","step"} or not all(isinstance(data[k],str) and 0<len(data[k])<=180 for k in ("model","episode")) or type(data.get("step"))!=int or not 0<=data["step"]<=1_000_000:
                raise APIError(400,"需要有效的模型、轨迹和非负整数步数。")
            from . import inference
            if not PREDICT_SLOT.acquire(blocking=False):raise APIError(429,"正在计算另一条动作，请稍后重试。")
            try:result=inference.predict(data["model"],data["episode"],data["step"])
            finally:PREDICT_SLOT.release()
            self.send_json(200,result);return
        if route.startswith("/api/"):
            if route in ("/api/remote/runs","/api/remote/training"):
                from .remote import remote_snapshot
                if route=="/api/remote/training" and not one('run'):raise APIError(400,'请选择远端实验。')
                result=remote_snapshot(one('run') if route=="/api/remote/training" else None)
            elif route=="/api/runs":
                from .training import runs_catalog
                result={"runs":runs_catalog(PROJECT)}
            elif route=="/api/training":
                from .training import training_snapshot
                result=training_snapshot(PROJECT,one("run") or None)
            elif route=="/api/quality":
                from .training import quality_snapshot
                result=quality_snapshot(PROJECT,one("run") or None)
            elif route=="/api/compare":
                from .training import comparison_snapshot
                result=comparison_snapshot(PROJECT,one("run"),one("base"),one("target"))
            elif route=="/api/training/control":
                from .controls import control_status
                result=control_status(PROJECT,one("run") or None)
            elif route=="/api/hardware":result=hardware()
            elif route=="/api/rollouts":
                from .rollouts import rollout_status
                result=rollout_status(PROJECT)
            elif route=="/api/rollouts/frames":
                from .rollouts import rollout_frames,RolloutBusy,RolloutUnavailable
                try:result=rollout_frames(PROJECT,one("job"))
                except RolloutBusy as e:raise APIError(409,str(e))
                except RolloutUnavailable as e:raise APIError(503,str(e))
            elif route=="/api/inference/catalog":
                from .inference import catalog
                result=catalog()
            elif route=="/api/inference/trajectory":
                from .inference import trajectory
                result=trajectory(one("episode"))
            elif route=="/api/health":result={"service":"cube-lab","ok":True}
            else:raise APIError(404,"没有这个接口。")
            self.send_json(200,result);return
        name="index.html" if route=="/" else route.removeprefix("/static/") if route.startswith("/static/") else None
        if name not in ASSETS:raise APIError(404,"找不到这个页面。")
        self.send_data(200,ASSETS[name],(STATIC/name).read_bytes())
    def handle_request(self):
        try:self.dispatch()
        except (BrokenPipeError,ConnectionResetError,socket.timeout):return
        except APIError as e:self.send_json(e.status,{"error":e.message})
        except (KeyError,FileNotFoundError) as e:self.send_json(404,{"error":str(e)[:250]})
        except (ValueError,IndexError) as e:self.send_json(400,{"error":str(e)[:250]})
        except Exception as e:
            LOG.exception("Dashboard request failed")
            self.send_json(500,{"error":"读取实验数据失败："+str(e)[:240]})
    do_GET=do_HEAD=do_POST=handle_request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=8765)
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error("端口必须在 1 到 65535 之间")
    try:server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler)
    except OSError as exc:p.exit(1,f"无法监听本地端口 {a.port}：{exc}；若面板已打开，直接访问原地址，或使用 --port 指定其他端口。\n")
    print(f"魔方操作工作台：http://127.0.0.1:{a.port}/  （Ctrl+C 只关闭面板）",flush=True)
    with server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass
if __name__=="__main__":main()
