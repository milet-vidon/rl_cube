"""Frozen +7° diagnostic: measured right-arm IK origin on unloaded approach commands."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
import sys
from types import FunctionType

import torch

from contact_hold90_cuda_domain import _specialize
import natural_face90_contact_point as points
import natural_face90_contact_point_metrics as metrics
import natural_face90_contact_point_worker as material
from natural_face90_hover_ab_metrics import CONTACT_IMPULSE, HALF
from natural_face90_hover_d_metrics import group

v7 = points.v7
SCHEMA = 'r70_plus7_approach_anchor_v1'
SCRIPT = 'natural_face90_approach_anchor_probe.py'
FILES = (SCRIPT, 'run_natural_face90_approach_anchor_probe.sh')
DEFAULT_HOME = points.RUNS / 'r27_recipe_approach_anchor_plus7'
SUMMARY = 'approach_anchor_summary.json'


def probe_pins():
    return {name: v7.digest(points.HERE / name) for name in FILES}


class ApproachGuide(material.MaterialGuide):
    def __init__(self, controller):
        super().__init__(controller)
        self.raw_seen = torch.zeros_like(self.aligned)
        self.approach_count = torch.zeros_like(self.anchor_frames)

    def filter_residual(self, residual, fresh, learned, obs):
        applied = super().filter_residual(residual, fresh, learned, obs)
        state = self.control_evidence()
        impulse = obs['right_total_impulse']
        any_touch = (impulse != 0.) | state['pre_contact'] | state['pre_index_contact'] | state['pre_other_digit_contact'] | obs['wrong_contact'].bool()
        self.raw_seen = (self.raw_seen & ~fresh) | (learned & any_touch)
        measured_z = -HALF - state['pre_normal_gap_m']
        engage = (learned & ~self.raw_seen & state['pre_aligned'] & state['pre_orientation_ok']
                  & state['pre_support_ok'] & ~state['pre_recovering'] & ~self.drive_contact
                  & ~obs['contact_goal_reached'].bool() & (state['pre_normal_gap_m'] > 0.)
                  & (self.goal_local[:, 2] > measured_z))
        state.update(pre_approach_anchor=engage, pre_raw_right_impulse_Ns=impulse,
                     pre_approach_any_touch=any_touch, pre_approach_seen_touch=self.raw_seen,
                     pre_approach_wrong=obs['wrong_contact'].bool(),
                     pre_approach_goal_reached=obs['contact_goal_reached'].bool())
        self.approach_count += engage.long()
        return applied

    def select_anchor(self, measured, previous_target):
        old = super().select_anchor(measured, previous_target)
        state = self.control_evidence()
        measured = measured.to(old.dtype)
        anchor = torch.where(state['pre_approach_anchor'][:, None], measured, old)
        state['pre_ik_anchor_error_max_rad'] = (anchor - measured).abs().amax(dim=1)
        return anchor


class ProbePhase(material.PostPhase):
    def __init__(self, e, req, out):
        super().__init__(e, req, out)
        self.guide = ApproachGuide(self.controller)

    def finish_learning(self):
        validate_probe(self.req)
        result = super().finish_learning()
        result['approach_anchor'] = dict(schema=SCHEMA, frames_by_lane=self.guide.approach_count.tolist(),
                                         source_sha256=probe_pins())
        return result


def validate_probe(req):
    points.validate_request(req)
    if (req.get('paired_validation') is not True or req['role'] != 'validate' or req['learn']
            or req['iteration'] != 70 or req['seeds'] != metrics.SEEDS or req['target_deg'] != 15
            or req['contact_point_offset_deg'] != 7 or req['checkpoint_sha'] != metrics.R70_SHA
            or req.get('approach_anchor_source_sha256') != probe_pins()
            or req.get('approach_anchor_schema') != SCHEMA):
        raise ValueError('Expected frozen +7° R70, the original lanes, and current diagnostic pins')


_run = FunctionType(material.run.__code__, dict(material.run.__globals__, PostPhase=ProbePhase))
_worker = _specialize(material.recipe.main, [
    ('from natural_face90_recipe_long import fingerprint, worker_software, validate_request',
     'from natural_face90_contact_point import fingerprint, worker_software'),
], dict(run=_run, validate_request=validate_probe))
_wave = _specialize(points.wave, [
    ("HERE / 'natural_face90_contact_point_worker.py'", 'HERE / SCRIPT'),
], dict(SCRIPT=SCRIPT, validate_request=validate_probe))


def baseline(sweep):
    c, source = points.baseline()
    plan = points.sweep_plan(c, source)
    if v7.read(sweep / 'plan.json') != plan:
        raise ValueError('The referenced sweep differs from the archived C/R70 plan')
    summary = points.summarize(sweep, source, plan)
    if (v7.read(sweep / points.SUMMARY) != summary or summary['decision'] != 'no_candidate'
            or summary['selected_offset_deg'] is not None):
        raise ValueError('Expected the completed no-candidate sweep')
    report = Path(summary['reports']['7']['path'])
    req = points.candidate(source, plan, 7)
    next_req = dict(req, approach_anchor_schema=SCHEMA, approach_anchor_source_sha256=probe_pins())
    validate_probe(next_req)
    probe_plan = dict(schema=SCHEMA, seeds=req['seeds'], checkpoint=req['checkpoint'], checkpoint_sha=req['checkpoint_sha'],
                      sweep_dir=str(sweep), sweep_summary_sha256=v7.digest(sweep / points.SUMMARY),
                      baseline_report=str(report), baseline_report_sha256=v7.digest(report),
                      source_sha256=probe_pins(), original_software=req['software'])
    return source, plan, next_req, probe_plan


def measured_approach(row, prior_touch):
    return (not prior_touch and row['pre_aligned'] and row['pre_orientation_ok'] and row['pre_support_ok']
            and not row['pre_recovering'] and not row['drive_command'] and row['pre_normal_gap_m'] > 0
            and row['requested_z_m'] > -HALF - row['pre_normal_gap_m'])


def approach_stats(rows, seeds):
    frames = [r for seed in seeds for r in rows[seed]]
    def med(i):
        return median(frame[i] for frame in frames) if frames else None
    return dict(frames=len(frames), requested_inward_median_mm=med(0), measured_inward_median_mm=med(1),
                measured_inward_frames=sum(r[1] > 0 for r in frames),
                returns_to_standoff=sum(r[2] for r in frames))


def phase_motion(path, seeds, candidate):
    frames = {s: [] for s in seeds}
    contact, seen_raw, previous = set(), set(), {}
    hits = Counter()
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            s = r['seed']
            pre_contact = r['pre_contact'] or r['pre_index_contact'] or r['pre_other_digit_contact']
            if pre_contact:
                contact.add(s)
            measured_z = -HALF - r['pre_normal_gap_m']
            if candidate:
                raw = r['pre_raw_right_impulse_Ns'] != 0 or pre_contact or r['pre_approach_wrong']
                if raw:
                    seen_raw.add(s)
                engage = bool(measured_approach(r, s in seen_raw) and not r['pre_approach_goal_reached'])
                if (r['pre_approach_any_touch'] != bool(raw) or r['pre_approach_seen_touch'] != (s in seen_raw)
                        or r['pre_approach_anchor'] != engage or (engage and r['pre_hover_residual_off'])):
                    raise ValueError(f'Invalid unloaded approach attribution at seed {s}, step {r["step"]}')
                actual = 0. if engage or r['pre_hover_measured_anchor'] else r['pre_previous_reference_error_max_rad']
                if not math.isclose(actual, r['pre_ik_anchor_error_max_rad'], rel_tol=0, abs_tol=1e-8):
                    raise ValueError('Measured/reference IK anchor disagrees with its physical receipt')
                hits[s] += int(engage)
            if previous.get(s) and r['pre_hover_measured_anchor']:
                frames[s][-1][2] = True
            previous[s] = bool(r['safe'] and measured_approach(r, s in contact))
            if previous[s]:
                frames[s].append([1000 * (r['requested_z_m'] - measured_z),
                                  1000 * (r['pad_z_m'] - measured_z), False])
            if max(r['right_target_impulse_Ns'], r['index_distal_impulse_Ns'], r['other_digit_target_impulse_Ns']) > CONTACT_IMPULSE:
                contact.add(s)
    return frames, [hits[s] for s in seeds]


def arm_summary(report, req, candidate):
    rows, learning = points.verify_report(report, req)
    seeds = req['seeds']
    episodes = metrics._read(report, 7, seeds)
    frames, counts = phase_motion(report.parent / 'turn_contact_evidence.jsonl', seeds, candidate)
    if candidate and learning.get('approach_anchor') != dict(schema=SCHEMA, frames_by_lane=counts, source_sha256=probe_pins()):
        raise ValueError('Approach frame counts or diagnostic source hashes differ from the measured trace')
    fixed = [row for row in episodes if row['seed'] in metrics.FIXED_SEEDS]
    for row in episodes:
        row['approach'] = approach_stats(frames, [row['seed']])
        if candidate:
            row['measured_approach_anchor_frames'] = counts[row['lane']]
    return dict(all_16=group(episodes), fixed_11=group(fixed),
                unsafe_seeds=[row['seed'] for row in rows if row['unsafe']],
                approach=dict(all_16=approach_stats(frames, seeds), fixed_11=approach_stats(frames, metrics.FIXED_SEEDS)),
                per_seed=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir', type=Path, default=points.DEFAULT_SWEEP)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--check', action='store_true', help='Verify source, baseline and R70 on CPU')
    args = parser.parse_args()
    home, sweep = args.run_dir.resolve(), args.sweep_dir.resolve()
    if home.parent != points.RUNS or sweep.parent != points.RUNS or home == sweep or args.run_dir.is_symlink():
        parser.error('Use separate experiment and completed sweep directories directly under runs/')
    source, sweep_plan, req, plan = baseline(sweep)
    if home.exists() and (not (home / 'plan.json').is_file() or v7.read(home / 'plan.json') != plan):
        parser.error('This output directory belongs to a different experiment')
    before = arm_summary(Path(plan['baseline_report']), points.candidate(source, sweep_plan, 7), False)
    if args.check:
        points.check_frozen_weights(source, sweep_plan)
        print('CHECK OK — archived +7°, R70 and 16 paired lanes; approach', before['approach']['fixed_11'])
        return
    home.mkdir(exist_ok=True)
    with v7.exclusive(home / 'campaign.lock') as fd:
        if (home / 'plan.json').exists() and v7.read(home / 'plan.json') != plan:
            raise ValueError('Diagnostic plan changed')
        v7.save(home / 'plan.json', plan)
        slot = home / 'probe'
        _wave(slot, req, fd)
        report = v7.accepted_report(slot, req)
        after = arm_summary(report, req, True)
        result = dict(schema=SCHEMA, plan=plan, updated=False, independent=False,
                      baseline=before, candidate=after, candidate_report=str(report), candidate_report_sha256=v7.digest(report),
                      new_unsafe_seeds=sorted(set(after['unsafe_seeds']) - set(before['unsafe_seeds'])),
                      qualifying_fixed_seeds=[r['seed'] for r in after['per_seed'] if r['seed'] in metrics.FIXED_SEEDS
                                              and r['safe_qualified_index_first'] and r['safe_drive_2deg']])
        output = home / SUMMARY
        if output.exists() and v7.read(output) != result:
            raise ValueError('Existing diagnostic summary differs from the verified evidence')
        v7.save(output, result)
        print('Frozen approach-anchor complete:', output, '\n', json.dumps(after['fixed_11']), flush=True)


if __name__ == '__main__':
    if '--request' in sys.argv[1:]:
        _worker()
    else:
        main()
