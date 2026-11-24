"""Additional native DIP-to-nZ contact attribution for the right-hand turn.

PhysX reports the opposing rigid body for every cubelet sensor contact. This
identifies the index distal link. The calibrated point on that body is an
independent kinematic measurement; the native contact need not be at that point.
"""
import numpy as np
import torch

from natural_face90_recipe_frames import PAD_BODY
from natural_face90_v5_metrics import DIGIT, LANE, rotate, unit


def _actor_paths(scene, native_ids):
    import warp as wp
    ids = np.asarray(native_ids, dtype=np.int64).view(np.uint64)
    return scene.sv_warp_contact.get_other_actor_paths_from_ids(
        wp.array(ids, dtype=wp.uint64, device='cpu'))


class IndexContact:
    def __init__(self, scene, monitor, pad_body, pad_offset):
        self.scene = scene
        self.original = scene.reducer
        self.raw = monitor.r
        if getattr(self.original, 'base', None) is not self.raw:
            raise ValueError('Expected the native PairRecorder above the target-layer reducer')
        names = scene.robot.shared_metatype.link_names
        if not (0 <= pad_body < len(names) and names[pad_body] == PAD_BODY):
            raise ValueError('Calibrated index distal body does not match articulation')
        self.pad_body = pad_body
        self.pad_offset = torch.as_tensor(pad_offset, dtype=torch.float64, device=scene.home.device)
        if self.pad_offset.shape != (3,) or not self.pad_offset.isfinite().all():
            raise ValueError('Invalid calibrated index-pad offset')
        target_sensors = self.raw.target.nonzero(as_tuple=True)[0]
        coverage = torch.bincount(self.raw.lanes[target_sensors], minlength=scene.n)
        if coverage.shape != (scene.n,) or not (coverage == 9).all():
            raise ValueError('Expected nine target-layer native sensors per lane')
        self.sensors = target_sensors
        self.index_impulse = torch.zeros(scene.n, dtype=torch.float64, device=scene.home.device)
        self._previous_total = torch.zeros_like(self.index_impulse)
        self._known = set()
        self._distal_ids = set()
        self._distal_tensor = torch.empty(0, dtype=torch.long, device=self.index_impulse.device)
        self.substeps = 0

    @property
    def other_digit_impulse(self):
        # Includes index links proximal to the distal rigid body.
        return (self.raw.acc[:, 0] - self.index_impulse).clamp_min(0.)

    def pad_position_world(self):
        body = self.scene.robot.get_link_transforms()[:, self.pad_body].double()
        return body[:, :3] + rotate(unit(body[:, 3:]), self.pad_offset)

    def clear(self, scene):
        self.original.clear(scene)
        self.index_impulse.zero_()
        self._previous_total.zero_()
        self.substeps = 0

    def _resolve_new(self, scene):
        # The target reducer has already classified and verified the native IDs
        # this substep. Resolve full paths only for newly seen right-digit IDs.
        if len(self.raw.cache) == len(self._known):
            return
        new = self.raw.cache.keys() - self._known
        fingers = sorted(k for k in new if self.raw.cache[k][1] == 1)
        if fingers:
            paths = _actor_paths(scene, fingers)
            if len(paths) != len(fingers):
                raise ValueError('Missing native right-finger body identities')
            for k, path in zip(fingers, paths):
                m = LANE.fullmatch(str(path))
                if not m or int(m[1]) != self.raw.cache[k][0] or not DIGIT.fullmatch(m[2]):
                    raise ValueError('Native right-finger identity or lane changed')
                if m[2] == '/Robot/' + PAD_BODY:
                    self._distal_ids.add(k)
            self._distal_tensor = torch.tensor(sorted(self._distal_ids), dtype=torch.long,
                                               device=self.index_impulse.device)
        self._known.update(new)

    def update(self, scene):
        self.original.update(scene)
        self.substeps += 1
        total = self.raw.acc[:, 0]
        # With zero additional impulse in the existing all-right-digit / nZ
        # channel there cannot be any positive distal / nZ impulse this substep.
        if torch.equal(total, self._previous_total):
            return
        self._previous_total.copy_(total)
        self._resolve_new(scene)
        if not len(self._distal_tensor):
            return
        forces, _, _, _, counts, starts, other = self.raw.v.get_raw_contact_data(1.)
        count = counts.reshape(-1).long()
        start = starts.reshape(-1).long()
        if count.numel() != len(self.raw.sid) or count.shape != start.shape:
            raise ValueError('Invalid native cube sensor coverage')
        if (count < 0).any() or count.sum() >= self.raw.capacity:
            raise ValueError('Native cube contact capacity exceeded')
        selected = count[self.sensors]
        n = int(selected.sum())
        if n == 0:
            raise ValueError('Positive target impulse without target sensor records')
        sensor = torch.repeat_interleave(self.sensors, selected, output_size=n)
        offsets = torch.repeat_interleave(selected.cumsum(0) - selected, selected, output_size=n)
        index = start[sensor] + torch.arange(n, device=sensor.device) - offsets
        force = forces.reshape(-1)
        actor = other.reshape(-1).view(torch.int64)
        if (index < 0).any() or (index >= min(len(force), len(actor))).any():
            raise ValueError('Invalid target-layer native contact span')
        impulse = force[index].double().abs()
        if not impulse.isfinite().all():
            raise ValueError('Nonfinite target-layer contact impulse')
        # The wrapped reducer already rejects unknown IDs and cross-lane pairs
        # for every native record. Here only the sensor subset and DIP change.
        distal = torch.isin(actor[index], self._distal_tensor)
        self.index_impulse.scatter_add_(0, self.raw.lanes[sensor], impulse * distal)

    def validate(self):
        self.original.validate()
        if self.substeps != self.scene.dec or self.raw.count != self.substeps:
            raise ValueError('Incomplete per-frame native index contact sample')
        total = self.raw.acc[:, 0]
        if (self.index_impulse > total + 1e-10 * (1. + total)).any():
            raise ValueError('Distal target impulse exceeds all right-finger target impulses')


def install(scene, monitor, pad_body, pad_offset):
    tracker = IndexContact(scene, monitor, pad_body, pad_offset)
    scene.reducer = tracker
    return tracker
