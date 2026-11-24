"""Paired C/D readbacks. Each first contact is attributed to its actual command interval."""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from natural_face90_hover_ab_metrics import CONTACT_IMPULSE, HALF, hover_command, validate_residual


def close(a, b):
    return math.isclose(a, b, rel_tol=0., abs_tol=1e-8)


def check_recenter(row, previous, seen):
    hold = row['pre_approach_recenter_hold']
    was_held = previous is not None and previous['pre_approach_recenter_hold']
    entry = row['pre_approach_recenter_entry']
    measured = row['pre_measured_pad_z_m']
    cap = row['pre_recenter_depth_cap_m']
    origin_z = row['pre_c_requested_z_m']
    request_z = row['requested_z_m']
    original_hover = hover_command(dict(row, requested_z_m=origin_z), seen)
    raw_contact = row['pre_recenter_any_right_contact']
    seen_right = row['pre_recenter_seen_right_contact']
    previously_seen = previous is not None and previous['pre_recenter_seen_right_contact']
    previous_approach = (previous is not None and previous['pre_aligned'] and previous['pre_support_ok']
                         and previous['pre_orientation_ok'] and not previous['pre_recovering'] and not previously_seen)
    expected = (original_hover and not seen_right and row['pre_orientation_ok'] and (previous_approach or was_held)
                and .004 < row['pre_lateral_error_m'] <= .007 and row['pre_normal_gap_m'] > 0)
    if (raw_contact not in (0, 1) or seen_right != int(previously_seen or raw_contact)
            or (seen and not seen_right) or hold != int(expected) or entry != int(hold and not was_held)
            or not close(measured, -HALF - row['pre_normal_gap_m'])):
        raise ValueError('Invalid D hold gate, contact history or pre-command pad depth')
    expected_cap = min(previous['pre_recenter_depth_cap_m'], measured) if hold and was_held else measured
    if not close(cap, expected_cap) or not close(request_z, cap if hold else origin_z):
        raise ValueError('D normal target disagrees with its measured depth cap or C command')
    return original_hover, bool(hold)


def episode(frames, outcome, treatment):
    first = None
    count = Counter()
    min_gap = None
    qualified_drive = False
    last_isolated = False
    drive_start = None
    peak = 0.
    for i, row in enumerate(frames):
        pre_contact = row['pre_contact'] or row['pre_index_contact'] or row['pre_other_digit_contact']
        if first is None and pre_contact:
            first = dict(phase='at_handoff' if i == 0 else 'already_in_contact',
                         kind='unattributed', step=None, qualified=False)
        seen = first is not None
        if treatment:
            anchor, hold = check_recenter(row, frames[i-1] if i else None, seen)
        else:
            anchor, hold = hover_command(row, seen), False
        validate_residual(row, anchor)
        if row['pre_hover_measured_anchor'] != int(anchor):
            raise ValueError('Measured joint anchor disagrees with the original C gate')
        approach = (not seen and row['pre_aligned'] and row['pre_orientation_ok']
                    and row['pre_support_ok'] and not row['pre_recovering'] and not row['drive_command'])
        count['anchor'] += int(anchor)
        count['standoff'] += int(anchor and not hold)
        count['hold'] += int(hold)
        count['approach'] += int(approach)
        count['hold_entries'] += int(treatment and row['pre_approach_recenter_entry'])
        if (approach or hold) and row['safe']:
            gap = 1000 * row['pre_normal_gap_m']
            min_gap = gap if min_gap is None else min(min_gap, gap)
        index = row['index_distal_impulse_Ns'] > CONTACT_IMPULSE
        other = row['other_digit_target_impulse_Ns'] > CONTACT_IMPULSE
        any_digit = row['right_target_impulse_Ns'] > CONTACT_IMPULSE or index or other
        qualified = bool(row['qualified_index_frame'])
        drive = bool(row['drive_command'])
        count['distal_contact_frames'] += int(index)
        count['qualified_index_frames'] += int(qualified)
        count['drive_frames'] += int(drive)
        isolated = qualified and index and not other and bool(row['safe'])
        qualified_drive |= drive and isolated
        if drive and isolated and last_isolated:
            if drive_start is None:
                drive_start = frames[i-1]['face_deg']
            peak = max(peak, (row['face_deg'] - drive_start + 180.) % 360. - 180.)
        else:
            drive_start = None
        last_isolated = isolated
        if not close(row['peak_drive_response_deg'], peak):
            raise ValueError('Drive response disagrees with isolated two-endpoint signed face motion')
        if not seen and any_digit:
            phase = 'recenter' if hold else 'standoff' if anchor else 'approach' if approach else 'other'
            if index and other:
                kind = 'mixed'
            elif index:
                kind = 'index_only'
            elif other:
                kind = 'other_only'
            else:
                kind = 'aggregate_only'
            first = dict(phase=phase, kind=kind, step=row['step'], qualified=qualified and bool(row['safe']))
    if any(count[k] != outcome[k] for k in ('distal_contact_frames', 'qualified_index_frames', 'drive_frames')):
        raise ValueError('Physical contact or drive frame counts differ from the saved episode')
    if not close(peak, outcome['peak_drive_response_deg']):
        raise ValueError('Measured drive response differs from the saved episode')
    if first is None:
        first = dict(phase='no_measured_contact' if frames else 'no_learned_phase',
                     kind='none', step=None, qualified=False)
    safe = not outcome['unsafe']
    clean = safe and first['kind'] == 'index_only' and first['qualified']
    return dict(seed=outcome['seed'], lane=outcome['lane'], learned_frames=len(frames),
                anchor_frames=count['anchor'], standoff_frames=count['standoff'], hold_frames=count['hold'],
                hold_entries=count['hold_entries'], unloaded_approach_frames=count['approach'], first_contact=first,
                min_safe_precontact_approach_gap_mm=min_gap, unsafe=outcome['unsafe'], end_reason=outcome['end_reason'],
                success=outcome['success'], turn_started=outcome['turn_started'], drive_frames=count['drive_frames'],
                safe_qualified_index_first=clean, safe_qualified_approach_index_first=clean and first['phase']=='approach',
                safe_qualified_drive=safe and qualified_drive, safe_drive_2deg=safe and qualified_drive and peak>=2.,
                peak_drive_response_deg=peak)


def read_episodes(path, seeds, treatment):
    path = Path(path)
    report = json.loads(path.read_bytes())
    rows = report['rows']
    if (len(set(seeds)) != len(seeds) or report['seeds'] != seeds or [r['seed'] for r in rows] != seeds
            or [r['lane'] for r in rows] != list(range(len(seeds)))):
        raise ValueError('Mismatched seed or lane order')
    lanes = [[] for _ in seeds]
    with (path.parent / 'turn_contact_evidence.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            lane, step = row['lane'], row['step']
            if (type(lane) is not int or not 0 <= lane < len(lanes) or seeds[lane] != row['seed']
                    or type(step) is not int or step < 0 or not all(math.isfinite(v) for v in row.values())):
                raise ValueError('Invalid lane, step or control measurement')
            if lanes[lane] and step != lanes[lane][-1]['step'] + 1:
                raise ValueError('Missing or duplicated control step')
            lanes[lane].append(row)
    learning = report['expert_learning']
    if (sum(map(len, lanes)) != learning['learned_transitions']
            or sum(map(bool, lanes)) != learning['learned_lanes']):
        raise ValueError('Incomplete learned frame coverage')
    return [episode(frames, row, treatment) for frames, row in zip(lanes, rows)]


def group(rows):
    result = dict(episodes=len(rows), reached_learned_phase=sum(bool(r['learned_frames']) for r in rows),
                  safe_unloaded_approach_episodes=sum(bool(r['unloaded_approach_frames']) and not r['unsafe'] for r in rows),
                  recenter_episodes=sum(bool(r['hold_frames']) for r in rows),
                  drive_episodes=sum(bool(r['drive_frames']) for r in rows),
                  first_contact_phase=dict(Counter(r['first_contact']['phase'] for r in rows)),
                  first_contact_kind=dict(Counter(r['first_contact']['kind'] for r in rows)),
                  successes=sum(r['success'] for r in rows))
    for key in ('unsafe', 'turn_started', 'safe_qualified_index_first', 'safe_qualified_approach_index_first',
                'safe_qualified_drive', 'safe_drive_2deg'):
        result[key] = sum(r[key] for r in rows)
    return result


def fixed_panel(pairs):
    gaps = [(p['C']['min_safe_precontact_approach_gap_mm'], p['D']['min_safe_precontact_approach_gap_mm'])
            for p in pairs if not p['C']['unsafe'] and not p['D']['unsafe']]
    gaps = [(c, d) for c, d in gaps if c is not None and d is not None]
    return dict(seeds=[p['seed'] for p in pairs], C=group([p['C'] for p in pairs]),
                D=group([p['D'] for p in pairs]), safe_paired_min_approach_gap_mm=dict(
                    pairs=len(gaps), C_median=statistics.median(c for c, d in gaps) if gaps else None,
                    D_median=statistics.median(d for c, d in gaps) if gaps else None,
                    median_D_minus_C=statistics.median(d-c for c, d in gaps) if gaps else None,
                    smaller_in_D=sum(d < c for c, d in gaps)))


def compare(baseline_report, treatment_report, seeds):
    seeds = list(seeds)
    control = read_episodes(baseline_report, seeds, False)
    treatment = read_episodes(treatment_report, seeds, True)
    pairs = [dict(seed=c['seed'], lane=c['lane'], C=c, D=d) for c, d in zip(control, treatment)]
    grouped = dict(C=group(control), D=group(treatment))
    cohorts = {'fixed_C_no_contact_horizon': [], 'fixed_C_early_other_digit': [], 'fixed_C_no_learned_phase': [],
               'fixed_C_remaining': []}
    for p in pairs:
        c = p['C']
        first = c['first_contact']
        if first['phase'] == 'no_measured_contact' and c['standoff_frames'] and c['end_reason'] == 'post_horizon' and not c['unsafe']:
            cohort = 'fixed_C_no_contact_horizon'
        elif first['phase'] == 'standoff' and first['kind'] == 'other_only':
            cohort = 'fixed_C_early_other_digit'
        elif not c['learned_frames']:
            cohort = 'fixed_C_no_learned_phase'
        else:
            cohort = 'fixed_C_remaining'
        p['cohort'] = cohort
        cohorts[cohort].append(p)
    grouped.update({name: fixed_panel(panel) for name, panel in cohorts.items() if panel or name != 'fixed_C_remaining'})
    return dict(group=grouped, per_seed=pairs)
