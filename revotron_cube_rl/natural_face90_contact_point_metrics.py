"""Score material-point offsets against the archived C rollout on the same R70 lanes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from natural_face90_hover_ab import R70_SHA, SEEDS
from natural_face90_hover_d_metrics import CONTACT_IMPULSE, fixed_panel, group, read_episodes


FIXED_SEEDS = (66002208, 66002211, 66002212, 66002213, 66002215, 66002217,
               66002218, 66002220, 66002221, 66002222, 66002223)
OFFSETS = (7, -7)


def _rotated_xy(vector, degrees):
    """Positive angle is about the outward nZ axis in the Core frame."""
    x, y = vector
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    return (x * c + y * s, y * c - x * s)


def _verify_xy(actual, expected):
    if len(actual) != 2 or not all(math.isclose(a, e, abs_tol=1e-8, rel_tol=0)
                                   for a, e in zip(actual, expected)):
        raise ValueError('Material-point coordinates do not match the selected nZ rotation')


def _geometry(folder, offset, episodes, seeds, learning):
    material = learning['contact_point']
    radius = material['radius_m']
    selected = material['selected_radial_xy_by_lane']
    base = material['base_radial_xy_by_lane']
    active = material['selected_by_lane']
    if (material['offset_deg'] != offset or radius != .030
            or any(len(seq) != len(seeds) for seq in (selected, base, active))):
        raise ValueError('Wrong material-point offset, radius or lane count')
    for lane, row in enumerate(episodes):
        if type(active[lane]) is not bool or (row['learned_frames'] and not active[lane]):
            raise ValueError('Missing material-point selection on a learned lane')
        if active[lane]:
            if not math.isclose(math.hypot(*base[lane]), 1., rel_tol=0, abs_tol=1e-8):
                raise ValueError('Invalid original unit radial')
            _verify_xy(selected[lane], _rotated_xy(base[lane], offset))

    first_distal = {}
    expected_impulses = {}
    with (folder / 'turn_contact_evidence.jsonl').open() as frames:
        for line in frames:
            r = json.loads(line)
            radial = selected[r['lane']]
            unit = _rotated_xy(radial, r['pre_material_face_deg'])
            _verify_xy((r['pre_material_marker_x_m'], r['pre_material_marker_y_m']),
                       (radius * unit[0], radius * unit[1]))
            if r['index_distal_impulse_Ns'] > 0:
                expected_impulses[r['seed'], r['step']] = r['index_distal_impulse_Ns']
            if r['index_distal_impulse_Ns'] > CONTACT_IMPULSE:
                first_distal.setdefault(r['seed'], r['step'])

    native = learning['geometry_probe']
    if (native['schema'] != 'native_dip_contact_geometry_v1' or native['file'] != 'native_contact_geometry.jsonl'
            or native['selected_seeds'] != seeds
            or native['active_frames_by_seed'] != {str(r['seed']): r['learned_frames'] for r in episodes}):
        raise ValueError('Incomplete native contact observation of the paired rollout')
    # The first threshold-crossing control interval can contain multiple contacting
    # points. Use the strongest point of the earliest substep, as in the native probe.
    first_points = {}
    measured_impulses = Counter()
    digest = hashlib.sha256()
    count = 0
    with (folder / native['file']).open('rb') as points:
        for line in points:
            digest.update(line)
            p = json.loads(line)
            lane, seed = p['lane'], p['seed']
            if (not 0 <= lane < len(seeds) or seeds[lane] != seed
                    or not math.isfinite(p['impulse_Ns']) or p['impulse_Ns'] <= 0
                    or 'nZ' not in p['cubelet'].split('_')):
                raise ValueError('Unexpected native contact lane, impulse or cubelet')
            unit = _rotated_xy(selected[lane], p['face_deg'])
            _verify_xy(p['marker_core_m'][:2], (radius * unit[0], radius * unit[1]))
            measured_impulses[seed, p['step']] += p['impulse_Ns']
            count += 1
            if p['step'] == first_distal.get(seed):
                previous = first_points.get(seed)
                if previous is None or (p['substep'], -p['impulse_Ns']) < (previous['substep'], -previous['impulse_Ns']):
                    first_points[seed] = p
    if (digest.hexdigest() != native['sha256'] or native['points'] != count
            or first_points.keys() != first_distal.keys()):
        raise ValueError('Missing or changed native contact evidence')
    contact_frames = Counter(seed for seed, step in measured_impulses)
    if (native['frames_by_seed'] != {str(seed): contact_frames[seed] for seed in seeds}
            or measured_impulses.keys() != expected_impulses.keys()
            or any(not math.isclose(value, expected_impulses[key], rel_tol=1e-8, abs_tol=1e-12)
                   for key, value in measured_impulses.items())):
        raise ValueError('Native DIP impulses differ from the commanded control interval readback')
    for r in episodes:
        p = first_points.get(r['seed'])
        r['first_measured_distal_native_contact'] = None if p is None else {
            key: p[key] for key in ('step', 'substep', 'cubelet', 'point_core_m', 'marker_core_m',
                                    'point_cubelet_m', 'contact_marker_lateral_m', 'impulse_Ns')
        }


def _read(path, offset, seeds):
    path = Path(path)
    report = json.loads(path.read_bytes())
    learning = report['expert_learning']
    if (report['role'] != 'validate' or learning['actor_updated'] is not False
            or learning['deterministic'] is not True or learning['input_checkpoint_sha'] != R70_SHA):
        raise ValueError('Comparison requires all three frozen R70 validation reports')
    with (path.parent / 'turn_contact_evidence.jsonl').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != learning['turn_evidence_sha256']:
            raise ValueError('Changed turn-contact trajectory')
    episodes = read_episodes(path, seeds, treatment=False)
    for row, outcome in zip(episodes, report['rows'], strict=True):
        if not math.isfinite(outcome['final_face_deg']) or outcome['target_deg'] != 15:
            raise ValueError('Invalid signed final angle or unexpected stage target')
        # The raw report uses zero for a lane that never produced any post measurements.
        row['final_face_deg'] = outcome['final_face_deg'] if outcome['post_samples'] else None
    if offset:
        _geometry(path.parent, offset, episodes, seeds, learning)
    return episodes


def _scores(control, candidate, fixed):
    pairs = [dict(C=c, D=r, seed=r['seed']) for c, r in zip(control, candidate, strict=True)
             if r['seed'] in fixed]
    panel = fixed_panel(pairs)
    unsafe = [r['seed'] for r in candidate if r['unsafe']]
    new_unsafe = [r['seed'] for c, r in zip(control, candidate, strict=True) if r['unsafe'] and not c['unsafe']]
    both = [r['seed'] for r in candidate if r['seed'] in fixed and r['safe_qualified_index_first']
            and r['safe_drive_2deg']]
    gaps = panel['safe_paired_min_approach_gap_mm']
    return dict(all_16=group(candidate), fixed_C_no_contact_horizon=panel['D'],
                safe_paired_min_approach_gap_mm=dict(pairs=gaps['pairs'], baseline_median=gaps['C_median'],
                    candidate_median=gaps['D_median'], median_candidate_minus_baseline=gaps['median_D_minus_C'],
                    smaller_in_candidate=gaps['smaller_in_D']),
                unsafe_seeds=unsafe, new_unsafe_seeds=new_unsafe, qualifying_fixed_seeds=both,
                meets_training_gate=len(both) >= 2 and not new_unsafe, per_seed=candidate)


def _selection(angles):
    """Rank eligible candidates. Exact ties prefer the prespecified positive offset."""
    ranks = {}
    for angle in OFFSETS:
        score = angles[str(angle)]
        if score['meets_training_gate']:
            ranks[angle] = (len(score['qualifying_fixed_seeds']), -len(score['unsafe_seeds']),
                            score['fixed_C_no_contact_horizon']['safe_qualified_index_first'])
    best = [a for a in ranks if ranks[a] == max(ranks.values())]
    return dict(selected_offset_deg=best[0] if best else None,
                decision='selected' if best else 'no_candidate', tied_offsets_deg=best if len(best) > 1 else [])


def compare(c_report, plus_report, minus_report, seeds):
    """Read the three physical reports and recompute contact and isolated drive from traces."""
    seeds = list(seeds)
    if seeds != SEEDS:
        raise ValueError('Expected the 16 original paired R70 seeds in lane order')
    control = _read(c_report, 0, seeds)
    fixed = [r['seed'] for r in control if r['first_contact']['phase'] == 'no_measured_contact'
             and r['standoff_frames'] and r['end_reason'] == 'post_horizon' and not r['unsafe']]
    if tuple(fixed) != FIXED_SEEDS:
        raise ValueError('Baseline no-contact cohort differs from the archived 11 seeds')
    angles = {'0': _scores(control, control, fixed)}
    for offset, path in zip(OFFSETS, (plus_report, minus_report), strict=True):
        angles[str(offset)] = _scores(control, _read(path, offset, seeds), fixed)
    return dict(fixed_C_no_contact_horizon_seeds=fixed, angles=angles, **_selection(angles))
