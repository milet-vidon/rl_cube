"""Stream verified world poses from a completed demonstration; no physics runtime required."""
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import time


def _poses(values, names):
    if not isinstance(values, dict) or set(values) != names:
        raise ValueError('Missing measured cube or robot poses in replay')
    for p in values.values():
        if (not isinstance(p,list) or len(p)!=7
                or any(type(v) not in (float,int) or not math.isfinite(v) for v in p)
                or abs(math.hypot(*p[3:])-1.)>.003):
            raise ValueError('Invalid world pose; expected finite xyz and unit XYZW quaternion')


def _index(stream, meta, summary):
    """Only keep byte offsets. Full 30 Hz robot and cube readbacks stay on disk."""
    offsets = array('Q')
    digest = hashlib.sha256()
    links = set(meta['robot_link_names'])
    cubes = {'Core', *meta['cubelet_names']}
    if not links or len(links) != len(meta['robot_link_names']) or len(cubes)!=27:
        raise ValueError('Missing physical robot links or cubelets in replay metadata')
    first_manual, last_step, manual = None, None, 0
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line:
            break
        digest.update(line)
        row = json.loads(line)
        step, phase = row['step'], row['phase']
        if (type(step) is not int or step < 0 or (last_step is not None and step != last_step+1)
                or not isinstance(phase,str)):
            raise ValueError('Nonconsecutive recorded steps')
        _poses(row.get('bodies'),links)
        _poses(row.get('cube'),cubes)
        if not all(type(row[k]) in (int,float) and math.isfinite(row[k])
                   for k in ('angle_deg','thumb_Ns','index_Ns','other_Ns')):
            raise ValueError('Invalid angle or contact readback')
        if phase=='manual':
            if first_manual is None:
                first_manual=len(offsets)
            manual+=1
        offsets.append(offset)
        last_step=step
    if (not offsets or len(offsets)!=summary['recorded_frames'] or last_step!=summary['last_step']
            or manual!=summary['manual_frames'] or digest.hexdigest()!=summary['frames_sha256']):
        raise ValueError('Demo frames differ from the completed recording or have no replayable frames')
    return offsets, first_manual or 0


def _snapshot(row, index, count, summary, source):
    return dict(phase='replay', recorded_phase=row['phase'], step=row['step'], frame_index=index, frame_count=count,
                bodies=row['bodies'], cube=row['cube'], angle_deg=row['angle_deg'], thumb_Ns=row['thumb_Ns'],
                index_Ns=row['index_Ns'], other_Ns=row['other_Ns'], structure_ok=row['structure_ok'],
                thumb_index_same_substeps=row['thumb_index_same_substeps'], safe=row['safe'],
                reason=f"{source} · {summary['reason']}")


def replay(ui, path):
    """Loop from the manual handoff, or show the prefix for a failed/no-action recording."""
    path = Path(path).expanduser()
    if path.is_dir():
        frames = path/'demo_frames.jsonl'
    elif path.name=='demo_frames.jsonl':
        frames = path
    else:
        raise ValueError('Pass a demonstration directory or its demo_frames.jsonl')
    folder = frames.parent
    summary = json.loads((folder/'demo_summary.json').read_bytes())
    raw_meta = (folder/'demo_metadata.json').read_bytes()
    meta = json.loads(raw_meta)
    schema = 'r27_human_two_finger_demo_v2'
    if meta.get('schema')!=schema or summary.get('schema')!=schema:
        raise ValueError('Replay requires a v2 demonstration with measured link poses; older files lack these poses')
    if (summary.get('trajectory')!=frames.name or summary.get('status') not in ('stopped','failed')
            or meta.get('control_hz')!=30 or hashlib.sha256(raw_meta).hexdigest()!=summary.get('metadata_sha256')):
        raise ValueError('Invalid demonstration metadata, clock or completion receipt')

    with frames.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        offsets, start = _index(stream,meta,summary)
        print(f"Verified {len(offsets)} frames; replaying {frames} from frame {start+1}. Ctrl+C to finish.",flush=True)
        while True:
            stream.seek(offsets[start])
            deadline = time.monotonic()
            for i in range(start,len(offsets)):
                now_stat = os.fstat(stream.fileno())
                if now_stat.st_size!=stat.st_size or now_stat.st_mtime_ns!=stat.st_mtime_ns:
                    raise ValueError('Recording changed during replay')
                row = json.loads(stream.readline())
                ui.publish(_snapshot(row,i-start+1,len(offsets)-start,summary,folder.name))
                deadline+=1./meta['control_hz']
                time.sleep(max(0.,deadline-time.monotonic()))
