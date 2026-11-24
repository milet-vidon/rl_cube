"""Native, lane-local right thumb/index and left support impulses at each 240 Hz substep."""
import torch

from natural_face90_turn_contacts import _actor_paths
from natural_face90_v5_metrics import DIGIT, LANE

IMPULSE = 1e-5
QUIET = 1e-7
SIDE_BITS = ('pX', 'nX', 'pY', 'nY')


def opposed(thumb, index):
    """A quarter-turn carries the cubelet material labels with the physical layer."""
    opposite = ((thumb & 5) << 1) | ((thumb & 10) >> 1)
    return (opposite & index) != 0


class GripContacts:
    def __init__(self, scene, monitor):
        self.original = scene.reducer
        self.raw = monitor.r
        if self.original.base is not self.raw or scene.dec != 8:
            raise ValueError('Expected the original target/contact observer at 240 Hz')
        self.scene = scene
        self.sources = self.raw.target.nonzero(as_tuple=True)[0]
        if not (torch.bincount(self.raw.lanes[self.sources], minlength=scene.n) == 9).all():
            raise ValueError('Missing native target cubelets in grip observer')
        labels = [str(p).rsplit('/', 1)[-1].split('_') for p in self.raw.v.sensor_paths]
        self.sides = self.raw.acc.new_tensor([[side in label for side in SIDE_BITS] for label in labels])
        self.weights = torch.tensor([1,2,4,8], dtype=torch.int64, device=self.raw.acc.device)
        self.class_by_id = {}
        self.channels = torch.empty(0, dtype=torch.long, device=scene.home.device)
        # Per control interval, eight rows of [right thumb on nZ, right index on nZ, left on rest].
        self.samples = self.raw.acc.new_zeros(scene.n, scene.dec, 3)
        self.side_masks = torch.zeros(scene.n, scene.dec, 2, dtype=torch.int64, device=self.raw.acc.device)
        self.last = self.raw.acc.new_zeros(scene.n, 2)
        self.count = 0
        scene.reducer = self

    def clear(self, scene):
        self.original.clear(scene)
        self.samples.zero_()
        self.side_masks.zero_()
        self.last.zero_()
        self.count = 0

    def classify(self, scene):
        raw = self.raw
        if len(raw.cache) == len(self.class_by_id):
            return
        missing = sorted(raw.cache.keys() - self.class_by_id.keys())
        fingers = [k for k in missing if raw.cache[k][1] == 1]
        self.class_by_id.update(dict.fromkeys(missing, 2))
        if fingers:
            for k, path in zip(fingers, _actor_paths(scene, fingers), strict=True):
                m = LANE.fullmatch(str(path))
                if not m or int(m[1]) != raw.cache[k][0] or not DIGIT.fullmatch(m[2]):
                    raise ValueError('Invalid native grip actor or lane')
                if m[2].startswith('/Robot/right_thumb_'):
                    self.class_by_id[k] = 0
                elif m[2].startswith('/Robot/right_index_'):
                    self.class_by_id[k] = 1
        keys = sorted(self.class_by_id)
        self.channels = torch.tensor([self.class_by_id[k] for k in keys],
                                     dtype=torch.long, device=raw.acc.device)

    def update(self, scene):
        self.original.update(scene)
        raw = self.raw
        totals = raw.acc[:, [0, 3]]
        delta = totals - self.last
        self.last.copy_(totals)
        sample = self.samples[:, self.count]
        self.count += 1
        sample[:, 2] = delta[:, 1]
        if not (delta[:, 0] > 0.).any():
            return
        self.classify(scene)
        force, _, _, _, counts, starts, other = raw.v.get_raw_contact_data(1.)
        counts = counts.reshape(-1).long()[self.sources]
        total = int(counts.sum())
        if total == 0 or total >= raw.capacity:
            raise ValueError('Invalid native target contacts at a loaded substep')
        sensors = torch.repeat_interleave(self.sources, counts, output_size=total)
        offsets = torch.repeat_interleave(counts.cumsum(0) - counts, counts, output_size=total)
        i = starts.reshape(-1).long()[sensors] + torch.arange(total, device=counts.device) - offsets
        forces = force.reshape(-1)
        actors = other.reshape(-1).view(torch.int64)
        if (i < 0).any() or (i >= min(len(forces), len(actors))).any():
            raise ValueError('Invalid native target contact spans')
        actor = actors[i]
        where = torch.searchsorted(raw.keys, actor)
        if (where >= len(self.channels)).any() or not torch.equal(raw.keys[where], actor):
            raise ValueError('Unknown target-contact body')
        lane = raw.lanes[sensors]
        if not torch.equal(raw.code[where, 0], lane):
            raise ValueError('Cross-lane grip attribution')
        col = self.channels[where]
        impulse = forces[i].double().abs()
        sample[:, 0].scatter_add_(0, lane, impulse * (col == 0))
        sample[:, 1].scatter_add_(0, lane, impulse * (col == 1))
        # Each corner can legitimately belong to two outer material columns.
        # Retain all memberships. The central nZ cubelet belongs to none.
        side_load = impulse.new_zeros(scene.n * 3, 4)
        side_load.index_add_(0, lane*3 + col, impulse[:,None] * self.sides[sensors])
        loaded = side_load.view(scene.n, 3, 4)[:, :2] > IMPULSE/scene.dec
        self.side_masks[:,self.count-1] = (loaded * self.weights).sum(2)

    def validate(self):
        self.original.validate()
        both = self.samples[:, :, :2].sum((1, 2))
        if (self.count != self.scene.dec or self.raw.count != self.count
                or not self.samples.isfinite().all() or (self.samples < 0).any()
                or (both > self.raw.acc[:, 0] + 1e-10*(1.+self.raw.acc[:, 0])).any()):
            raise ValueError('Incomplete or inconsistent native bimanual contact sample')

    def read(self, measurements):
        thumb, index, left = self.samples.sum(1).unbind(1)
        overlap = self.samples.gt(0).all(2).sum(1)
        t, i = self.side_masks.unbind(2)
        opposed_steps = (opposed(t, i) & (self.samples[:,:,2] > IMPULSE/self.scene.dec)).sum(1)
        m = measurements
        other = (self.raw.acc[:, 0] - thumb - index).clamp_min(0.)
        wrong = self.raw.acc[:, 1]
        left_target = self.raw.acc[:, 4]
        support = ((left > IMPULSE) & m['support'] & m['natural_ready'] & m['structure_ok']
                   & m['allowed_layer_state'] & ~m['wrong_contact'])
        bilateral = (support & (thumb > IMPULSE) & (index > IMPULSE) & (opposed_steps > 0)
                     & (left_target <= QUIET) & (other <= QUIET) & (wrong <= QUIET))
        return dict(thumb_Ns=thumb, index_Ns=index, left_static_Ns=left, left_target_Ns=left_target,
                    other_target_Ns=other, wrong_Ns=wrong, overlap=overlap, opposed_substeps=opposed_steps,
                    supported=support,
                    thumb=thumb > IMPULSE, index=index > IMPULSE, left_static=left > IMPULSE,
                    bilateral=bilateral)
