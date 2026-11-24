"""Inspect the first index-only contacts in an archived +7 degree approach-anchor probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT = Path(__file__).resolve().parent / 'runs/r27_recipe_approach_anchor_plus7/approach_anchor_summary.json'


def checked_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')
    return json.loads(data)


def checked_frames(path, expected, seeds):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            frame = json.loads(line)
            lane = frame['lane']
            if not 0 <= lane < len(seeds) or seeds[lane] != frame['seed']:
                raise ValueError(f'Lane/seed mismatch in {path}')
            yield frame
    if digest.hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')


def touch_geometry(turn, post, point, radial, radius):
    # Reconstruct the worker's post-step nZ material marker, in Core coordinates.
    angle = math.radians(turn['face_deg'])
    c, s = math.cos(angle), math.sin(angle)
    x, y = radial
    mx, my = radius * (x*c + y*s), radius * (y*c - x*s)
    gap = -.045 - turn['pad_z_m']  # positive outside the selected layer
    lateral = math.hypot(turn['pad_x_m'] - mx, turn['pad_y_m'] - my)
    near = abs(gap) <= .006 and lateral <= .012
    if (near != bool(turn['pad_near_marker']) or bool(turn['safe']) != bool(post['safe'])
            or not math.isclose(turn['face_deg'], post['face_deg'], abs_tol=1e-8)):
        raise ValueError(f'Inconsistent post-step geometry at seed {turn["seed"]}, step {turn["step"]}')
    failures = [name for name, passed in (
        ('pad_axial', abs(gap) <= .006), ('pad_lateral', lateral <= .012),
        ('safe_readback', turn['safe']), ('natural_ready', post['natural_ready']),
        ('index_impulse', turn['index_distal_impulse_Ns'] > 1e-5),
        ('static_impulse', turn['static_support_impulse_Ns'] > 1e-5),
        ('structure', post['structure_ok']), ('allowed_layer', post['allowed_layer_state']),
        ('wrong_contact', not post['wrong_contact']), ('retract', not post['retract']),
    ) if not passed]
    qualified = bool(turn['qualified_index_frame'])
    if qualified and failures:
        raise ValueError('Recorded qualification conflicts with observed gates')
    if not qualified and not failures:
        failures.append('unresolved_support_gate')
    # The solver's contact point and marker are from its own substep.
    return dict(pad_gap_mm=1000*gap, pad_lateral_mm=1000*lateral,
                native_lateral_mm=1000*point['contact_marker_lateral_m'],
                native_to_pad_mm=1000*point['anchor_contact_distance_m'],
                native_cubelet=point['cubelet'], native_substep=point['substep'],
                frame_qualified=qualified, observed_failed_gates=failures)


def audit(path, include_native=False):
    summary = json.loads(path.read_bytes())
    if summary['schema'] != 'r70_plus7_approach_anchor_v1' or summary['updated']:
        raise ValueError('Expected the frozen +7 degree approach-anchor summary')
    report_path = Path(summary['candidate_report'])
    report = checked_json(report_path, summary['candidate_report_sha256'])
    plan = summary['plan']
    sweep = checked_json(Path(plan['sweep_dir']) / 'contact_point_summary.json', plan['sweep_summary_sha256'])
    seeds = report['seeds']
    episodes = summary['candidate']['per_seed']
    if (report['role'] != 'validate' or seeds != plan['seeds']
            or [r['seed'] for r in episodes] != seeds or [r['seed'] for r in report['rows']] != seeds):
        raise ValueError('Inconsistent candidate report or seed set')
    fixed = sweep['fixed_C_no_contact_horizon_seeds']
    by_seed = {r['seed']: r for r in episodes}
    first = {(s, by_seed[s]['first_contact']['step']) for s in fixed
             if by_seed[s]['first_contact']['kind'] == 'index_only'}
    learning = report['expert_learning']
    material, native = learning['contact_point'], learning['geometry_probe']
    if material['offset_deg'] != 7 or material['radius_m'] != .030:
        raise ValueError('Unexpected material-point geometry')
    folder = report_path.parent
    turns, posts, points, last = {}, {}, {}, {}
    for frame in checked_frames(folder / 'turn_contact_evidence.jsonl', learning['turn_evidence_sha256'], seeds):
        key = frame['seed'], frame['step']
        if key in first:
            turns[key] = frame
    for frame in checked_frames(folder / 'post_measurements.jsonl', report['post_evidence_sha256'], seeds):
        seed, step = frame['seed'], frame['step']
        last[seed] = max(last.get(seed, step), step)
        if (seed, step) in first:
            posts[seed, step] = frame
    for point in checked_frames(folder / native['file'], native['sha256'], seeds):
        key = point['seed'], point['step']
        if key in first:
            previous = points.get(key)
            if previous is None or (point['substep'], -point['impulse_Ns']) < (previous['substep'], -previous['impulse_Ns']):
                points[key] = point
    if first != turns.keys() or first != posts.keys() or first != points.keys():
        raise ValueError('Missing turn, post-step or native evidence for first index contact')
    rows = []
    for seed in fixed:
        episode = by_seed[seed]
        contact = episode['first_contact']
        step = contact['step']
        row = dict(seed=seed, kind=contact['kind'], first_step=step, unsafe=episode['unsafe'],
                   end_reason=episode['end_reason'], safe_qualified_first=episode['safe_qualified_index_first'],
                   steps_to_unsafe_end=last[seed]-step if episode['unsafe'] and step is not None else None)
        if (seed, step) in first:
            turn = turns[seed, step]
            radial = material['selected_radial_xy_by_lane'][turn['lane']]
            row.update(touch_geometry(turn, posts[seed, step], points[seed, step], radial, material['radius_m']))
            if include_native:
                row['native_point'] = points[seed, step]
            if (turn['index_distal_impulse_Ns'] <= 1e-5 or turn['other_digit_target_impulse_Ns'] > 1e-5
                    or contact['qualified'] != bool(row['frame_qualified'] and turn['safe'])):
                raise ValueError(f'First index contact disagrees with archived readback for {seed}')
        rows.append(row)
    return dict(all_unsafe_seeds=[r['seed'] for r in report['rows'] if r['unsafe']],
                new_unsafe_seeds=summary['new_unsafe_seeds'], fixed=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = audit(args.summary)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print('seed     first_step pad_gap_mm pad_lat_mm native_lat_mm native_to_pad_mm frame_Q'
          '  failed_gates       end_reason  steps_touch_to_unsafe_end')
    for row in result['fixed']:
        if 'frame_qualified' in row:
            print(f'{row["seed"]} {row["first_step"]:10d} {row["pad_gap_mm"]:10.2f} {row["pad_lateral_mm"]:10.2f}'
                  f' {row["native_lateral_mm"]:13.2f} {row["native_to_pad_mm"]:16.2f} {int(row["frame_qualified"]):7d}',
                  ','.join(row['observed_failed_gates']) or 'pass', row['end_reason'], row['steps_to_unsafe_end'])
        else:
            print(row['seed'], row['kind'], row['end_reason'])
    print('All unsafe seeds', result['all_unsafe_seeds'], 'new vs frozen +7 baseline', result['new_unsafe_seeds'])


if __name__ == '__main__':
    main()
