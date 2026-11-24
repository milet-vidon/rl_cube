"""Combine per-wave training/validation reservoirs into one uniform reservoir per stratum."""
from operator import index

import numpy as np


def merge(waves, seed, cap=8192):
    """Return the sampled columns for one logical round; lane IDs must already be global.

    Each (arrays, report) provides unbiased samples *without replacement* and the
    BEFORE-subsampling counts in eligible_per_stratum. Reduced wave widths change
    the particular draw but not the per-stratum sampling law.
    """
    # Wave zero's reservoir also uses the round seed. Don't reuse its RNG stream:
    # the merge draws must be independent of the within-wave reservoir draws.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x57415645,)))
    return _merge(waves, rng, cap)


def _merge(waves, rng, cap):
    cap = index(cap)
    if cap < 1:
        raise ValueError('Reservoir capacity must be positive')
    prior = [0] * 4
    bank = [None] * 4
    columns = None
    for a, report in waves:
        if columns is None:
            columns = tuple(a)
        if set(a) != set(columns) or a['val'].ndim != 1 or a['val'].dtype != np.bool_:
            raise ValueError('Invalid wave columns or validation split')
        rows = len(a['val'])
        if a['phase'].shape != (rows,) or any(len(a[k]) != rows for k in columns):
            raise ValueError('Wave column lengths disagree')
        raw = report['eligible_per_stratum']
        if len(raw) != 4 or any(isinstance(n,(bool,np.bool_)) for n in raw):
            raise ValueError('Expected four population counts')
        population = [index(n) for n in raw]
        group = 2 * a['val'].astype(np.int8) + np.isin(a['phase'], (4,5))
        for g, n in enumerate(population):
            ids = np.flatnonzero(group == g)
            if n < 0 or not min(cap, n) <= len(ids) <= n:
                raise ValueError(f'Wave reservoir/count mismatch for stratum {g}')
            if not n:
                continue
            old = prior[g]
            total = old + n
            size = min(cap, total)
            # Number of new-wave rows in a uniform size-k subset of the union.
            take_new = (size if not old else n if size == total else
                        int(rng.hypergeometric(n, old, size)))
            take_old = size - take_new
            if take_new:
                selected = ids if take_new == len(ids) else rng.choice(ids, take_new, replace=False)
            if take_old:
                old_rows = min(old, cap)
                keep = slice(None) if take_old == old_rows else rng.choice(old_rows, take_old, replace=False)
            # Uniform sub-sampling of a uniform local reservoir is itself uniform
            # on that wave's entire eligible population. Retain at most cap rows.
            if take_new and take_old:
                bank[g] = {k: np.concatenate((bank[g][k][keep], a[k][selected])) for k in columns}
            elif take_new:
                bank[g] = {k: a[k][selected] for k in columns}
            elif take_old < min(old, cap):
                bank[g] = {k: bank[g][k][keep] for k in columns}
            prior[g] = total
    if not any(prior):
        raise ValueError('No eligible supervised transitions')
    return {k: np.concatenate([b[k] for b in bank if b is not None]) for k in columns}
