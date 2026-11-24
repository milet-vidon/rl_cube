"""Frozen R70 comparison of a collision-geometry-based index-pad orientation at +7°."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType

import natural_face90_approach_anchor_probe as anchor
import natural_face90_pad_orientation_policy as policy


points, v7 = anchor.points, anchor.v7
SCHEMA = 'r70_plus7_pad_orientation_v1'
SCRIPT = 'natural_face90_pad_orientation_probe.py'
GEOMETRY_SCRIPT = 'natural_face90_pad_geometry.py'
FILES = (SCRIPT, 'run_natural_face90_pad_orientation_probe.sh', 'natural_face90_pad_orientation_policy.py',
         GEOMETRY_SCRIPT, 'natural_face90_geometry_probe_surfaces.py', 'natural_face90_first_touch_audit.py')
DEFAULT_HOME = points.RUNS / 'r27_recipe_pad_orientation_plus7'
SUMMARY = 'pad_orientation_summary.json'


def source_pins():
    return {name: v7.digest(points.HERE / name) for name in FILES}


def geometry_hash(geometry):
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_probe(req):
    anchor.validate_probe(req)
    plan = req['pad_orientation_plan']
    if (plan['schema'] != SCHEMA or plan['source_sha256'] != source_pins()
            or plan['geometry_sha256'] != geometry_hash(req['pad_orientation_geometry'])
            or v7.digest(Path(plan['input_summary'])) != plan['input_summary_sha256']
            or v7.digest(Path(plan['scene_file'])) != plan['scene_sha256']):
        raise ValueError('Pad orientation source, scene, frozen approach summary or geometry changed')


def verify_report(report, req):
    rows, learning = points.verify_report(report, req)
    if v7.read(report)['domain_receipt']['scene_sha256'] != req['pad_orientation_plan']['scene_sha256']:
        raise ValueError('Candidate used a different scene from the archived +7° diagnostic')
    policy.verify_receipt(learning['pad_orientation'], report.parent / 'turn_contact_evidence.jsonl', req)
    return rows, learning


_run = FunctionType(anchor._run.__code__, dict(anchor._run.__globals__, PostPhase=policy.PadOrientationPhase))
_worker = FunctionType(anchor._worker.__code__, dict(anchor._worker.__globals__, run=_run, validate_request=validate_probe))
_wave = FunctionType(anchor._wave.__code__, dict(anchor._wave.__globals__, SCRIPT=SCRIPT,
                                                verify_report=verify_report, validate_request=validate_probe))


def prepare(archive, check_weights=False):
    saved = v7.read(archive)
    sweep = Path(saved['plan']['sweep_dir'])
    source, sweep_plan, previous_req, previous_plan = anchor.baseline(sweep)
    if (saved['schema'] != anchor.SCHEMA or saved['updated'] is not False or saved['independent'] is not False
            or previous_plan != saved['plan'] or previous_plan != v7.read(archive.parent / 'plan.json')):
        raise ValueError('Completed approach-anchor plan does not match current frozen inputs')
    previous_report = v7.accepted_report(archive.parent / 'probe', previous_req)
    if str(previous_report) != saved['candidate_report'] or v7.digest(previous_report) != saved['candidate_report_sha256']:
        raise ValueError('Previous approach summary references a different accepted report')
    original_req = points.candidate(source, sweep_plan, 7)
    baseline = anchor.arm_summary(Path(previous_plan['baseline_report']), original_req, False)
    previous = anchor.arm_summary(previous_report, previous_req, True)
    if saved['baseline'] != baseline or saved['candidate'] != previous:
        raise ValueError('Archived +7° arms differ from their physical evidence')
    if check_weights:
        points.check_frozen_weights(source, sweep_plan)
    receipt = v7.read(previous_report)['domain_receipt']
    scene = Path(receipt['scene_source']) / 'dynamic_training_scene.usda'
    # A separate process reads stage meshes with pxr. The worker consumes only
    # the frozen arrays, keeping its Isaac/Kit import order intact.
    raw = subprocess.check_output([sys.executable, '-B', str(points.HERE / GEOMETRY_SCRIPT),
                                   '--summary', str(archive)])
    geometry = json.loads(raw)
    plan = dict(schema=SCHEMA, input_summary=str(archive), input_summary_sha256=v7.digest(archive),
                scene_file=str(scene), scene_sha256=receipt['scene_sha256'], geometry_sha256=geometry_hash(geometry),
                checkpoint_sha=previous_req['checkpoint_sha'], seeds=previous_req['seeds'], source_sha256=source_pins())
    req = dict(previous_req, pad_orientation_geometry=geometry, pad_orientation_plan=plan)
    validate_probe(req)
    return plan, req, baseline, previous


def paired_result(candidate, original, previous):
    fixed = set(anchor.metrics.FIXED_SEEDS)
    qualified = [r['seed'] for r in candidate['per_seed'] if r['seed'] in fixed
                 and r['safe_qualified_index_first'] and r['safe_drive_2deg']]
    unsafe = set(candidate['unsafe_seeds'])
    new = sorted(unsafe - set(original['unsafe_seeds']))
    return dict(qualifying_fixed_seeds=qualified, new_unsafe_seeds=new,
                new_unsafe_vs_previous=sorted(unsafe - set(previous['unsafe_seeds'])),
                paired_contact_gate_met=len(qualified) >= 2 and not new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-summary', type=Path, default=anchor.DEFAULT_HOME / anchor.SUMMARY)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--check', action='store_true', help='CPU-only verification of archived rollouts, R70 and mesh geometry')
    args = parser.parse_args()
    home, archive = args.run_dir.resolve(), args.input_summary.resolve()
    if (home.parent != points.RUNS or home == archive.parent or args.run_dir.is_symlink()
            or archive.name != anchor.SUMMARY):
        parser.error('Use a separate run directory under runs/ and the completed approach anchor summary')
    plan, req, original, previous = prepare(archive, args.check)
    saved_plan = home / 'plan.json'
    if home.exists() and (not saved_plan.is_file() or v7.read(saved_plan) != plan):
        parser.error('Run directory has a different or missing plan')
    if args.check:
        print('CHECK OK — frozen R70, 16 paired lanes, scene and pad geometry', plan['geometry_sha256'])
        return
    home.mkdir(exist_ok=True)
    with v7.exclusive(home / 'campaign.lock') as fd:
        if saved_plan.exists() and v7.read(saved_plan) != plan:
            raise ValueError('Pad orientation plan changed')
        v7.save(saved_plan, plan)
        slot = home / 'probe'
        _, learning = _wave(slot, req, fd)
        report = v7.accepted_report(slot, req)
        candidate = anchor.arm_summary(report, req, True)
        result = dict(schema=SCHEMA, plan=plan, updated=False, independent=False, baseline=original,
                      previous=previous, candidate=candidate, candidate_report=str(report),
                      candidate_report_sha256=v7.digest(report), pad_orientation=learning['pad_orientation'],
                      **paired_result(candidate, original, previous))
        output = home / SUMMARY
        if output.exists() and v7.read(output) != result:
            raise ValueError('Saved orientation comparison differs from verified evidence')
        v7.save(output, result)
        print('Frozen pad orientation complete', output)
        print(json.dumps(dict(fixed_11=candidate['fixed_11'], all_16=candidate['all_16'],
                              new_unsafe_seeds=result['new_unsafe_seeds'],
                              paired_contact_gate_met=result['paired_contact_gate_met'])))


if __name__ == '__main__':
    if '--request' in sys.argv[1:]:
        _worker()
    else:
        main()
