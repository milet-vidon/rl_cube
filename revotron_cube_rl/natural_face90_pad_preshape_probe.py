"""Frozen R70 comparison of unloaded index preshaping followed by pad orientation at +7°."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

import natural_face90_pad_orientation_probe as wrist
import natural_face90_pad_preshape_check as geometry_check
import natural_face90_pad_preshape_policy as policy


anchor, points, v7 = wrist.anchor, wrist.points, wrist.v7
SCHEMA = 'r70_plus7_pad_preshape_v1'
SCRIPT = 'natural_face90_pad_preshape_probe.py'
DEFAULT_HOME = points.RUNS / 'r27_recipe_pad_preshape_plus7'
SUMMARY = 'pad_preshape_summary.json'
VARIANT = 'index_-6'
OFFSET_DEG = -6
FILES = (SCRIPT, 'run_natural_face90_pad_preshape_probe.sh', 'natural_face90_pad_preshape_policy.py',
         'natural_face90_pad_preshape_check.py', 'kinematics.py', 'natural_face90_turn_guide.py', 'config.json',
         '../revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf')


def source_pins():
    return {name: v7.digest(points.HERE / name) for name in FILES}


def handoff_geometry(archive, geometry, previous):
    rows, fixed = geometry_check.handoffs(archive)
    seeds = [r['seed'] for r in rows]
    if (seeds != [r['seed'] for r in previous['per_seed'] if r['learned_frames']]
            or tuple(fixed) != anchor.metrics.FIXED_SEEDS
            or geometry_check.VARIANTS[VARIANT] != {'index': OFFSET_DEG}):
        raise ValueError('Static screen differs from the measured cohort or preshape')
    check = geometry_check.ShapeCheck(geometry)
    per_seed = {}
    for row in rows:
        variants = check.check(row)
        per_seed[str(row['seed'])] = {name: variants[name] for name in ('measured', VARIANT)}
    counts = {}
    for name in ('measured', VARIANT):
        usable = [s for s in seeds if per_seed[str(s)][name]['tilt_deg'] is not None]
        counts[name] = dict(all=len(usable), fixed=len(set(usable).intersection(fixed)))
    if len(seeds) != 13 or counts[VARIANT] != dict(all=len(seeds), fixed=len(fixed)):
        raise ValueError(f'Frozen -6° preshape is missing a feasible sampled direction at an archived handoff: {counts}')
    return dict(handoffs=len(seeds), fixed_handoffs=len(fixed), counts=counts, per_seed=per_seed)


def validate_probe(req):
    wrist.validate_probe(req)
    plan = req['pad_preshape_plan']
    if (plan['schema'] != SCHEMA or plan['source_sha256'] != source_pins()
            or plan['orientation_plan'] != req['pad_orientation_plan']
            or req['pad_preshape_offset_deg'] != OFFSET_DEG or plan['index_flexion_offset_deg'] != OFFSET_DEG
            or plan['handoff_geometry']['counts'][VARIANT] != dict(all=13, fixed=11)):
        raise ValueError('Unexpected frozen index preshape, source or geometric coverage')


def verify_report(report, req):
    rows, learning = points.verify_report(report, req)
    if v7.read(report)['domain_receipt']['scene_sha256'] != req['pad_orientation_plan']['scene_sha256']:
        raise ValueError('Candidate used a different scene from the archived +7° diagnostic')
    receipt = learning['pad_preshape']
    if receipt['orientation'] != learning['pad_orientation']:
        raise ValueError('Preshape handoff and archived pad orientation receipts differ')
    policy.verify_receipt(receipt, report.parent / 'turn_contact_evidence.jsonl', req)
    return rows, learning


_run = FunctionType(wrist._run.__code__, dict(wrist._run.__globals__, PostPhase=policy.PadPreshapePhase))
_worker = FunctionType(wrist._worker.__code__, dict(wrist._worker.__globals__, run=_run, validate_request=validate_probe))
_wave = FunctionType(wrist._wave.__code__, dict(wrist._wave.__globals__, SCRIPT=SCRIPT,
                                                verify_report=verify_report, validate_request=validate_probe))


def prepare(archive, check_weights):
    orientation, req, original, previous = wrist.prepare(archive, check_weights)
    screen = handoff_geometry(archive, req['pad_orientation_geometry'], previous)
    plan = dict(schema=SCHEMA, seeds=req['seeds'], checkpoint_sha=req['checkpoint_sha'],
                input_summary=orientation['input_summary'], input_summary_sha256=orientation['input_summary_sha256'],
                index_flexion_offset_deg=OFFSET_DEG, source_sha256=source_pins(),
                orientation_plan=orientation, handoff_geometry=screen)
    req = dict(req, pad_preshape_offset_deg=OFFSET_DEG, pad_preshape_plan=plan)
    validate_probe(req)
    return plan, req, original, previous


def paired_result(candidate, original, previous, receipt):
    result = wrist.paired_result(candidate, original, previous)
    physical = result['qualifying_fixed_seeds']
    result['rollout_qualifying_fixed_seeds'] = physical
    result['rollout_paired_contact_gate_met'] = result['paired_contact_gate_met']
    orientation = receipt['orientation']
    attributed = []
    for row in candidate['per_seed']:
        i = row['lane']
        if (row['seed'] in physical and receipt['ready_step_by_lane'][i] is not None
                and receipt['handoff_error_rad_by_lane'][i] <= policy.ACTUAL_TOL_RAD
                and orientation['feasible_by_lane'][i]):
            attributed.append(row['seed'])
    result['qualifying_fixed_seeds'] = attributed
    result['paired_contact_gate_met'] = len(attributed) >= 2 and not result['new_unsafe_seeds']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-summary', type=Path, default=anchor.DEFAULT_HOME / anchor.SUMMARY)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--check', action='store_true', help='Verify the frozen inputs and screen measured versus extended index geometry on CPU')
    args = parser.parse_args()
    home, archive = args.run_dir.resolve(), args.input_summary.resolve()
    if (home.parent != points.RUNS or home == archive.parent or args.run_dir.is_symlink()
            or archive.name != anchor.SUMMARY):
        parser.error('Use a separate run directory under runs/ and the completed approach anchor summary')
    plan, req, original, previous = prepare(archive, args.check)
    saved_plan = home / 'plan.json'
    if home.exists() and (not saved_plan.is_file() or v7.read(saved_plan) != plan):
        parser.error('Run directory has a different or missing plan')
    counts = plan['handoff_geometry']['counts']
    print('CPU handoff geometry (13 learned, 11 fixed)', json.dumps(counts), flush=True)
    if args.check:
        print('CHECK OK — frozen R70, 16 paired lanes, index MCP/PIP/DIP -6°')
        return
    home.mkdir(exist_ok=True)
    with v7.exclusive(home / 'campaign.lock') as fd:
        if saved_plan.exists() and v7.read(saved_plan) != plan:
            raise ValueError('Pad preshape plan changed')
        v7.save(saved_plan, plan)
        slot = home / 'probe'
        _, learning = _wave(slot, req, fd)
        report = v7.accepted_report(slot, req)
        candidate = anchor.arm_summary(report, req, True)
        result = dict(schema=SCHEMA, plan=plan, updated=False, independent=False, baseline=original, previous=previous,
                      candidate=candidate, candidate_report=str(report), candidate_report_sha256=v7.digest(report),
                      pad_orientation=learning['pad_orientation'], pad_preshape=learning['pad_preshape'],
                      **paired_result(candidate, original, previous, learning['pad_preshape']))
        output = home / SUMMARY
        if output.exists() and v7.read(output) != result:
            raise ValueError('Saved preshape comparison differs from verified evidence')
        v7.save(output, result)
        print('Frozen pad preshape complete', output)
        print(json.dumps(dict(fixed_11=candidate['fixed_11'], all_16=candidate['all_16'],
                              new_unsafe_seeds=result['new_unsafe_seeds'],
                              paired_contact_gate_met=result['paired_contact_gate_met'])))


if __name__ == '__main__':
    if '--request' in sys.argv[1:]:
        _worker()
    else:
        main()
