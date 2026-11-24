"""Concrete original-baseline teacher-preserve diagnostic, pickup to wrist only."""
from pathlib import Path
import argparse
import json
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/center_j56_dev_v3_20261103'))
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=9501)
    parser.add_argument('--selection-seed',type=int,default=20261104)
    parser.add_argument('--teacher-probability',type=float,choices=[1.],default=1.)
    parser.add_argument('--max-steps',type=int,default=2000)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    from left_dagger_baseline_runtime_v1 import source_preflight,checkpoint_preflight,create_runtime
    from collect_left_dagger_draft_v1 import collect,v4_actor
    preflight=source_preflight(args.source)
    preflight['checkpoint_validation']=checkpoint_preflight(args.checkpoint,preflight)
    actor,checkpoint_hash=v4_actor(args.checkpoint,preflight['left'])
    if args.preflight_only:
        print(json.dumps({k:v for k,v in preflight.items() if k not in ('rows','experiment')},indent=2));return
    if not args.output:parser.error('--output required for actual diagnostic')
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=None
    try:
        runtime=create_runtime(args)
        with (args.output/'transitions.jsonl').open('w') as journal:
            result=collect(runtime,actor,journal,checkpoint_hash,max_steps=args.max_steps,
                teacher_probability=args.teacher_probability,seed=args.selection_seed)
        result=runtime.finalize(result)
        print('RESULT',json.dumps(result),flush=True)
    except BaseException as exc:
        (args.output/'error.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2))
        raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':main()
