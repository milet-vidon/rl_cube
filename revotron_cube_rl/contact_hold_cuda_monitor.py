"""Stop a corrupt native physics process on its first error; retry only capacity errors."""
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from contact_hold_wide_capacity import available_mib
from contact_hold_vram_probe import gpu

PRESSURE = re.compile(r'PxgCudaDeviceMemoryAllocator failed to allocate memory|out of memory|MemoryConfig::|Contact buffer exhausted|buffer overflow|capacity exceeded|capacity of .* insufficient|increase .*capacity', re.I)

def first_error(line):
    native = '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower())
    explicit = re.search(r'^(?:torch\.(?:cuda\.)?OutOfMemoryError|RuntimeError: (?:CUDA out of memory|Contact buffer exhausted))', line)
    if native or explicit:
        return ('memory_pressure' if PRESSURE.search(line) else 'native_error'), line.strip()[:800]

def monitor(cmd, log, fds, *, gpu_floor=448, ram_floor=4096, deadline_s=None):
    peak=rss=0; low=available_mib(); start=time.monotonic(); fault=None; pending=''
    name,total,_,_=gpu()
    def scan(reader, final=False):
        nonlocal pending
        data=(pending+reader.read()).split('\n')
        pending=data.pop()
        if final and pending:data.append(pending)
        return next((err for line in data if (err:=first_error(line))),None)
    with log.open('x') as out, log.open(errors='replace') as reader, subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, pass_fds=fds, start_new_session=True) as proc:
        try:
            while proc.poll() is None:
                fault=scan(reader)
                if fault:break
                _,_,used,_=gpu(); avail=available_mib(); peak=max(used,peak); low=min(low,avail)
                try:
                    rss=max(rss,int(next(l.split()[1] for l in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))//1024)
                except (FileNotFoundError, StopIteration):pass
                if total-used<gpu_floor or avail<ram_floor:fault=('memory_pressure','device_or_host_memory_headroom');break
                if deadline_s and time.monotonic()-start>deadline_s:fault=('timeout','trial deadline');break
                time.sleep(.35)
        finally:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
        fault=fault or scan(reader,True)
    if not fault and proc.returncode:fault=('worker_exit',f'worker returned {proc.returncode}')
    return dict(ok=not fault,kind=fault[0] if fault else None,reason=fault[1] if fault else None,
                peak_device_mib=peak,min_host_available_mib=low,peak_worker_rss_mib=rss,gpu=name,total_mib=total,
                elapsed_s=time.monotonic()-start,exit_code=proc.returncode)
