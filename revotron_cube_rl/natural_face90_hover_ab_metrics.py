"""Paired readback of frozen R70 runs. Pure CPU, with no simulator startup."""
import json
import math
import statistics
from pathlib import Path

from contact_hold90_cuda_domain import CONTACT_IMPULSE, HALF


def hover_command(r, seen_contact):
    """Identify the original unloaded standoff branch from its pre-step decisions."""
    standoff_z = -HALF + min(-.030, r['index_lead_m'] - .014)
    return (not seen_contact and r['pre_support_ok'] and not r['pre_aligned']
            and not r['pre_contact'] and not r['pre_index_contact'] and not r['pre_other_digit_contact']
            and not r['pre_recovering'] and not r['pre_centering'] and not r['pre_settling']
            and not r['pre_capture_remaining'] and not r['drive_command']
            and math.isclose(r['requested_z_m'], standoff_z, rel_tol=0., abs_tol=1e-8))


def tracking(samples):
    """A sample is (command_z, post-step pad_z) in mm on the same control interval."""
    vals = [pad-goal for goal,pad in samples.values()]
    return dict(samples=len(vals),
                rms_mm=math.sqrt(statistics.mean(v*v for v in vals)) if vals else None,
                peak_inward_mm=max(0., max(vals)) if vals else None,
                peak_to_peak_mm=max(vals)-min(vals) if len(vals) >= 2 else None,
                pad_peak_to_peak_mm=(max(pad for _,pad in samples.values()) -
                                     min(pad for _,pad in samples.values())) if len(vals) >= 2 else None)


def validate_residual(r, hover):
    names = ('pre_arm_residual_baseline_max_rad', 'pre_arm_residual_applied_max_rad',
             'pre_arm_residual_change_max_rad')
    original, applied, change = (r[name] for name in names)
    if (any(v < 0 or not math.isfinite(v) for v in (original, applied, change))
            or r['pre_hover_residual_off'] != int(hover)
            or (hover and (applied != 0. or not math.isclose(change, original, abs_tol=1e-10)))
            or (not hover and (change != 0. or original != applied))):
        raise ValueError(f"Residual intervention mismatch at seed {r['seed']} step {r['step']}")


def first_contact(first, on_entry, impulse, hover, r):
    if first is None and on_entry:
        return dict(status='already_in_contact_at_handoff', step=None)
    if first is None and impulse > CONTACT_IMPULSE:
        return dict(status='early_hover_contact' if hover else 'contact_outside_hover',
                    step=r['step'], impulse_Ns=impulse,
                    goal_z_mm=1000*r['requested_z_m'], post_pad_z_mm=1000*r['pad_z_m'])
    return first


def episode(rows, outcome, check_intervention):
    seen_contact = False
    right, index = None, None
    samples = {}
    active_changes = 0
    for i, r in enumerate(rows):
        right = first_contact(right, i == 0 and r['pre_contact'], 0., False, r)
        index = first_contact(index, i == 0 and r['pre_index_contact'], 0., False, r)
        seen_contact |= bool(r['pre_contact'] or r['pre_index_contact'] or r['pre_other_digit_contact'])
        hover = hover_command(r, seen_contact)
        if check_intervention:
            validate_residual(r, hover)
            active_changes += int(r['pre_arm_residual_change_max_rad'] > 0)
        if hover:
            samples[r['step']] = (1000*r['requested_z_m'], 1000*r['pad_z_m'])
        right_impulse = r['right_target_impulse_Ns']
        index_impulse = r['index_distal_impulse_Ns']
        right = first_contact(right, False, right_impulse, hover, r)
        index = first_contact(index, False, index_impulse, hover, r)
        seen_contact |= max(right_impulse, index_impulse, r['other_digit_target_impulse_Ns']) > CONTACT_IMPULSE
    missing = dict(status='no_measured_contact' if rows else 'did_not_reach_learned_phase', step=None)
    result = dict(seed=outcome['seed'], lane=outcome['lane'], learned_frames=len(rows),
                  entered_hover=bool(samples), tracking=tracking(samples),
                  first_right_digit_target_contact=right or missing, first_index_target_contact=index or missing,
                  unsafe=outcome['unsafe'], end_reason=outcome['end_reason'], success=outcome['success'],
                  turn_started=outcome['turn_started'], drive_frames=outcome['drive_frames'],
                  peak_drive_response_deg=outcome['peak_drive_response_deg'])
    if check_intervention:
        result['changed_residual_frames'] = active_changes
    return result, samples


def _read(report_path, seeds, check_intervention):
    path = Path(report_path)
    report = json.loads(path.read_bytes())
    if (len(set(seeds)) != len(seeds) or report['seeds'] != seeds
            or [r['seed'] for r in report['rows']] != seeds
            or [r['lane'] for r in report['rows']] != list(range(len(seeds)))):
        raise ValueError('Report seeds or lane order differ from the requested panel')
    groups = {seed: [] for seed in seeds}
    with (path.parent/'turn_contact_evidence.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            seed, lane, step = r['seed'], r['lane'], r['step']
            if not (0 <= lane < len(seeds) and seeds[lane] == seed):
                raise ValueError('Trace seed/lane mismatch')
            rows = groups[seed]
            if rows and step != rows[-1]['step']+1:
                raise ValueError(f'Missing or duplicated learned control step for seed {seed}')
            rows.append(r)
    learning = report['expert_learning']
    if (sum(map(len, groups.values())) != learning['learned_transitions']
            or sum(bool(g) for g in groups.values()) != learning['learned_lanes']):
        raise ValueError('Learned-frame coverage does not match the accepted report')
    episodes, tracks = [], {}
    for outcome in report['rows']:
        seed = outcome['seed']
        rows = groups[seed]
        if sum(r['index_distal_impulse_Ns'] > CONTACT_IMPULSE for r in rows) != outcome['distal_contact_frames']:
            raise ValueError(f'Native contact trace disagrees with report for seed {seed}')
        stats, tracks[seed] = episode(rows, outcome, check_intervention)
        episodes.append(stats)
    return episodes, tracks


def median_present(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def precontact_window(contact, a_track, b_track=None):
    """Last 60 intervals ending at A's first early right-digit contact. Include its post readback."""
    if contact['status'] != 'early_hover_contact':
        return dict(complete=False, status='no_early_right_digit_anchor_in_A')
    end = contact['step']
    steps = range(end-59,end+1)
    window = dict(end_step=end, start_step=end-59,
                  A_eligible_frames=sum(k in a_track for k in steps))
    if b_track is not None:
        window['B_eligible_frames'] = sum(k in b_track for k in steps)
    if window['A_eligible_frames'] != 60 or (b_track is not None and window['B_eligible_frames'] != 60):
        return dict(window, complete=False, status='incomplete_contiguous_hover_window')
    window.update(complete=True, status='complete', A=tracking({k:a_track[k] for k in steps}))
    if b_track is not None:
        window['B'] = tracking({k:b_track[k] for k in steps})
        window['delta_pad_peak_to_peak_mm'] = window['B']['pad_peak_to_peak_mm'] - window['A']['pad_peak_to_peak_mm']
    return window


def group(episodes):
    summary = dict(
        episodes=len(episodes), reached_learned_phase=sum(bool(r['learned_frames']) for r in episodes),
        entered_hover=sum(r['entered_hover'] for r in episodes),
        hover_frames=sum(r['tracking']['samples'] for r in episodes),
        unsafe=sum(r['unsafe'] for r in episodes), successes=sum(r['success'] for r in episodes),
        turn_started=sum(r['turn_started'] for r in episodes), drive_episodes=sum(r['drive_frames'] > 0 for r in episodes),
        safe_drive_2deg=sum(not r['unsafe'] and r['peak_drive_response_deg'] >= 2 for r in episodes))
    for label, key in (('right_digit','first_right_digit_target_contact'), ('index','first_index_target_contact')):
        counts = {s:sum(r[key]['status']==s for r in episodes)
                  for s in ('early_hover_contact','contact_outside_hover','no_measured_contact',
                            'already_in_contact_at_handoff','did_not_reach_learned_phase')}
        summary[f'early_{label}_contacts'] = counts['early_hover_contact']
        summary[f'{label}_first_contact_status'] = counts
        summary[f'hover_without_early_{label}_contact'] = sum(
            r['entered_hover'] and r[key]['status'] in ('contact_outside_hover','no_measured_contact')
            for r in episodes)
    for metric in ('rms_mm', 'peak_inward_mm', 'peak_to_peak_mm'):
        summary[f'median_hover_{metric}'] = median_present(r['tracking'][metric] for r in episodes)
    return summary


def summarize(report, seeds):
    """Accepted baseline A, in the original seed and lane order."""
    episodes, tracks = _read(report, list(seeds), False)
    for a in episodes:
        a['precontact_2s'] = precontact_window(a['first_right_digit_target_contact'], tracks[a['seed']])
    complete = [r['precontact_2s']['A'] for r in episodes if r['precontact_2s']['complete']]
    return dict(group=dict(group(episodes), complete_precontact_2s_windows=len(complete),
                           median_precontact_2s_pad_peak_to_peak_mm=median_present(
                               r['pad_peak_to_peak_mm'] for r in complete)), per_seed=episodes)


def compare(control_report, treatment_report, seeds):
    """B minus A paired at identical absolute control steps where both issued a free-hover command."""
    seeds = list(seeds)
    control, a_tracks = _read(control_report, seeds, False)
    treatment, b_tracks = _read(treatment_report, seeds, True)
    pairs = []
    for a,b in zip(control,treatment):
        seed = a['seed']
        shared = sorted(a_tracks[seed].keys() & b_tracks[seed].keys())
        ta = tracking({k:a_tracks[seed][k] for k in shared})
        tb = tracking({k:b_tracks[seed][k] for k in shared})
        delta = {m:tb[m]-ta[m] if ta[m] is not None else None
                 for m in ('rms_mm', 'peak_inward_mm', 'peak_to_peak_mm')}
        window = precontact_window(a['first_right_digit_target_contact'], a_tracks[seed], b_tracks[seed])
        pairs.append(dict(seed=seed, lane=a['lane'], A=a, B=b, common_hover_frames=len(shared), precontact_2s=window,
                          common_hover_A=ta, common_hover_B=tb, common_hover_delta_B_minus_A=delta))
    windows = [p['precontact_2s'] for p in pairs if p['precontact_2s']['complete']]
    grouped = dict(A=group(control), B=group(treatment),
                   seeds_with_common_hover=sum(p['common_hover_frames']>0 for p in pairs),
                   seeds_with_common_swing=sum(p['common_hover_frames']>=2 for p in pairs),
                   paired_precontact_2s=dict(
                       complete_seeds=len(windows),
                       A_median_pad_peak_to_peak_mm=median_present(w['A']['pad_peak_to_peak_mm'] for w in windows),
                       B_median_pad_peak_to_peak_mm=median_present(w['B']['pad_peak_to_peak_mm'] for w in windows),
                       median_paired_delta_pad_peak_to_peak_mm=median_present(
                           w['delta_pad_peak_to_peak_mm'] for w in windows)),
                   median_paired_delta_B_minus_A={
                       key:median_present(p['common_hover_delta_B_minus_A'][key] for p in pairs)
                       for key in ('rms_mm','peak_inward_mm','peak_to_peak_mm')})
    return dict(group=grouped, per_seed=pairs)
