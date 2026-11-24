"""Flush cProfile before IsaacLab native app.close() exits Python."""
from pathlib import Path
import argparse,cProfile,hashlib,json,runpy,sys,time

def run_profile(target,output,arguments):
    target=Path(target).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(target.parent))
    import isaac_compat
    original=isaac_compat.close_application;profile=cProfile.Profile();flushed=False
    def flush():
        nonlocal flushed
        if not flushed:
            profile.disable();profile.dump_stats(str(output));flushed=True
    def close(app):
        try:flush()
        finally:original(app)
    isaac_compat.close_application=close
    metadata={'target':str(target),'arguments':arguments,'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'parent Python process; child host preflight CPU excluded','start_unix':time.time()}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    sys.argv=[str(target)]+arguments
    try:
        profile.enable();runpy.run_path(str(target),run_name='__main__')
    finally:
        flush();isaac_compat.close_application=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('target');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    run_profile(a.target,a.output,a.arguments)
