"""Bounded, deterministic priority sampling for student corrections; validation is never mixed into training."""
import numpy as np

class Bank:
    def __init__(self,capacity,seed):
        self.cap=capacity;self.rng=np.random.default_rng(seed)
        self.parts=[[] for _ in range(4)];self.count=[0]*4;self.seen=[0]*4

    def _compact(self,g):
        parts=self.parts[g]
        if not parts:return
        data={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
        if len(data['priority'])>self.cap:
            ii=np.argpartition(data['priority'],self.cap-1)[:self.cap]
            data={k:v[ii] for k,v in data.items()}
        self.parts[g]=[data];self.count[g]=len(data['priority'])

    def add(self,d):
        focus=np.isin(d['phase'],(4,5))
        tag=2*d['val'].astype(np.int8)+focus
        for g in range(4):
            ii=np.flatnonzero(tag==g)
            if not len(ii):continue
            self.seen[g]+=len(ii);self.count[g]+=len(ii)
            self.parts[g].append(dict({k:v[ii] for k,v in d.items()},priority=self.rng.random(len(ii))))
            if self.count[g]>=2*self.cap:self._compact(g)

    def finish(self):
        for g in range(4):self._compact(g)
        parts=[d for v in self.parts for d in v]
        if not parts:raise ValueError('No eligible supervised transitions')
        return {k:np.concatenate([d[k] for d in parts]) for k in parts[0] if k!='priority'}
