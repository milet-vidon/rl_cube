"""Passive native contact attribution. Never changes collisions, targets or physics."""
import re
import numpy as np
import torch

LANE = re.compile(r'^/World/envs/env_(\d+)(/.*)$')


def identity(path):
    match = LANE.fullmatch(str(path))
    if not match:
        raise ValueError('Unmapped physical contact body: ' + str(path))
    return int(match[1]), match[2]


def pair_rows(paths, counts, starts, actors, impulses, lookup):
    """Host reduction of the original sensor packet, including non-contiguous spans."""
    if len(paths) != len(counts) or len(paths) != len(starts):
        raise ValueError('Incomplete contact sensor coverage')
    rows = {}
    for sensor, count, start in zip(paths, counts, starts):
        lane, first = identity(sensor)
        if count < 0 or start < 0 or start + count > min(len(impulses),len(actors)):
            raise ValueError('Invalid native contact span')
        for i in range(start, start + count):
            other_lane, second = identity(lookup[int(actors[i])])
            impulse = float(abs(impulses[i]))
            if not np.isfinite(impulse):
                raise ValueError('Nonfinite contact impulse')
            if lane != other_lane:
                raise ValueError('Cross-lane contact')
            if impulse and first.startswith('/Robot/') and second.startswith('/Robot/'):
                if first == second:
                    raise ValueError('Self actor identity in native contact packet')
                a, b = sorted((first, second))
                key = (lane, a, b)
                # Original GPU reducer also halves the sum of two directed sensors.
                rows[key] = rows.get(key, 0.) + .5 * impulse
    return rows


class PairRecorder:
    def __init__(self, scene):
        self.base = scene.reducer
        self.views = [(view, list(view.sensor_paths)) for kind, view, _ in scene.contact_views if kind == 'robot']
        if not self.views:
            raise ValueError('Missing native robot sensors')
        self.lookup = {}
        self.clear_pairs()
        self.totals = torch.zeros(scene.n, device=scene.home.device, dtype=torch.float64)

    def clear_pairs(self):
        self.pairs = {}
        self.substep = 0

    def clear(self, scene):
        self.base.clear(scene)
        self.clear_pairs()
        self.totals.zero_()

    def update(self, scene):
        self.base.update(scene)
        self.substep += 1
        # Only an increase in original self-contact impulse needs host attribution.
        if not bool((scene.contacts[:, 10] > self.totals).any()):
            return
        import warp as wp
        for view, paths in self.views:
            raw = view.get_raw_contact_data(1.)
            counts = raw[4].reshape(-1).long().cpu().numpy()
            starts = raw[5].reshape(-1).long().cpu().numpy()
            if len(counts) != len(paths) or counts.sum() >= view.max_contact_data_count:
                raise ValueError('Invalid robot contact coverage/capacity')
            if not counts.sum():
                continue
            spans = np.concatenate([np.arange(s, s+c) for c, s in zip(counts, starts) if c])
            actors = raw[6].reshape(-1).view(torch.int64).cpu().numpy()
            impulses = raw[0].reshape(-1).cpu().numpy()
            if spans.min() < 0 or spans.max() >= min(len(actors),len(impulses)):
                raise ValueError('Invalid native contact offset')
            unknown = sorted(set(int(a) for a in actors[spans]) - self.lookup.keys())
            if unknown:
                ids = np.asarray(unknown,dtype=np.int64).view(np.uint64)
                names = scene.sv_warp_contact.get_other_actor_paths_from_ids(wp.array(ids,dtype=wp.uint64,device='cpu'))
                if len(names) != len(ids):
                    raise ValueError('Incomplete actor path lookup')
                self.lookup.update(zip(unknown, map(str,names)))
            for key, val in pair_rows(paths, counts, starts, actors, impulses, self.lookup).items():
                rec = self.pairs.setdefault(key, dict(impulse_Ns=0., first_substep=self.substep, last_substep=self.substep))
                rec['impulse_Ns'] += val
                rec['last_substep'] = self.substep
        self.totals.copy_(scene.contacts[:,10])

    def validate(self):
        self.base.validate()

    def lane(self, i, expected):
        result = [dict(body_a=a,body_b=b,**v) for (lane,a,b),v in self.pairs.items() if lane == i]
        total = sum(r['impulse_Ns'] for r in result)
        if abs(total-expected) > max(1e-9,abs(expected)*1e-5):
            raise ValueError(f'Contact attribution mismatch: {total} vs native {expected}')
        return sorted(result,key=lambda r:-r['impulse_Ns'])
