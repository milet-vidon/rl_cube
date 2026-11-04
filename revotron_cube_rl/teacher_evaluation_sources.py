"""Resolve accepted, seed-matched right-hand schedules for hybrid evaluation."""
from pathlib import Path
from teacher_bootstrap import dataset,sha

def resolve_schedule(source,metadata,seed,explicit=None,allow_mismatch=False):
    source=Path(source);schedule=Path(explicit) if explicit is not None else None
    if schedule is None:
        if metadata['source_seed']==seed:schedule=source
        else:
            match=next((s for s in metadata.get('additional_sources',[]) if s['seed']==seed),None)
            if match is not None:
                # Resolve inside the explicitly selected local run root, never a stale training workspace.
                schedule=source.parent/Path(match['source']).name
                if sha(schedule/'artifact_manifest.json')!=match['manifest_sha256']:raise ValueError('Right schedule evidence mismatch')
            elif allow_mismatch:schedule=source
            else:raise ValueError('No accepted right-hand schedule for this seed; refusing a confounded evaluation')
    _,_,names,_,result=dataset(schedule)
    if names!=metadata['joint_names']:raise ValueError('Right schedule joint schema mismatch')
    matches=result['seed']==seed
    if not matches and not allow_mismatch:raise ValueError('Right schedule seed does not match evaluation seed')
    return schedule,result,matches
