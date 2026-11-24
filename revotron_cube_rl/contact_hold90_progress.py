"""Show simulator and optimizer progress on the campaign terminal; keep the full logs."""
from contextlib import contextmanager
from threading import Event, Thread

@contextmanager
def live_progress(path):
    initial = path.stat().st_size if path.exists() else 0
    stop = Event()
    milestones = ('NATIVE_CLONES', 'GPU_SCENE', 'GPU90_', 'FIT90 ')
    def read_log():
        while not path.exists():
            if stop.wait(0.25) and not path.exists():
                return
        with path.open(errors='replace') as stream:
            stream.seek(initial)
            while True:
                line = stream.readline()
                if line:
                    if line.startswith(milestones):
                        print(line.rstrip(), flush=True)
                elif stop.is_set():
                    return
                else:
                    stop.wait(0.5)
    worker = Thread(target=read_log, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2)
