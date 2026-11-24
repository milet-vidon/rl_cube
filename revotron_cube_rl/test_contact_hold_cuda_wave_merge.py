"""Exact small-population law, recovery determinism and split integrity for wave backoff."""
import itertools
import math
import unittest
from collections import defaultdict
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from contact_hold_cuda_wave_merge import merge


class Branch(Exception):
    pass


class ExactRng:
    """Enumerate the draws the actual merger requests, with exact probabilities."""
    def __init__(self, trace):
        self.trace = iter(trace)

    def draw(self, outcomes):
        try:
            return next(self.trace)
        except StopIteration:
            raise Branch(outcomes)

    def hypergeometric(self, good, bad, k):
        denom = math.comb(good + bad, k)
        return self.draw([(v,Fraction(math.comb(good,v)*math.comb(bad,k-v), denom))
                          for v in range(max(0,k-bad),min(k,good)+1)])

    def choice(self, ids, n, replace):
        assert not replace
        pool = list(range(ids)) if isinstance(ids,int) else list(ids)
        weight = Fraction(1,math.comb(len(pool),n))
        return self.draw([(np.array(ix),weight) for ix in itertools.combinations(pool,n)])


def wave(values, counts, group=0):
    n = len(values)
    a = dict(lane=np.array(values), val=np.full(n,group>=2), phase=np.full(n,4 if group%2 else 0),
             x=np.array(values)[:,None])
    return a,dict(eligible_per_stratum=counts)


class WaveMergeTest(unittest.TestCase):
    def test_equal_probability_of_every_pair_despite_unequal_saturated_waves(self):
        # Uniform local reservoirs (2 of 3, 2 of 5); pooling them unweighted
        # cannot yield the correct 1/C(8,2) probability for every global pair.
        masses = defaultdict(Fraction)
        for a in itertools.combinations(range(3),2):
            for b in itertools.combinations(range(3,8),2):
                waves = [wave(a,[3,0,0,0]), wave(b,[5,0,0,0])]
                stack = [([], Fraction(1,30))]
                while stack:
                    trace, mass = stack.pop()
                    try:
                        with patch('contact_hold_cuda_wave_merge.np.random.default_rng',return_value=ExactRng(trace)):
                            out = merge(waves, 101, cap=2)
                    except Branch as exc:
                        stack.extend((trace+[v],mass*w) for v,w in exc.args[0])
                    else:
                        pair = tuple(sorted(out['lane']))
                        np.testing.assert_array_equal(out['x'][:,0], out['lane'])
                        masses[pair] += mass
        self.assertEqual(dict(masses), {p:Fraction(1,28) for p in itertools.combinations(range(8),2)})

    def test_strata_stay_independent_and_resume_replays_same_draw(self):
        waves = [wave([10,11,12],[30,0,0,0]),wave([13,14,15],[70,0,0,0]),
                 wave([20,21],[0,2,0,0],1),wave([30],[0,0,1,0],2),wave([40,41],[0,0,0,2],3)]
        first = merge(iter(waves), 27, cap=3)
        second = merge(iter(waves), 27, cap=3)
        for k in first:np.testing.assert_array_equal(first[k],second[k])
        np.testing.assert_array_equal(first['x'][:,0],first['lane'])
        self.assertEqual(len(set(first['lane'])),8)
        self.assertEqual(set(first['lane'][first['val']]),{30,40,41})
        self.assertEqual(set(first['lane'][first['phase']==4]),{20,21,40,41})

    def test_reject_mislabeled_saturated_stratum(self):
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            merge([wave([1,2],[100,0,0,0])], 12, cap=3)


if __name__ == '__main__':
    unittest.main()
