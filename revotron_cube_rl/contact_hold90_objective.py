"""Episode rewards weight supervised imitation; independent physical validation is never an optimizer input."""
import numpy as np
import torch
from contact_hold90_cuda_fit import Balanced, PHASES, _json, _sha
from contact_hold90_controls import validate_params

def phase_weight(name,c):
    if name in ('preshape','lower','approach','squeeze'):return c['grasp_weight']
    if name in ('primary_lift','elbow_outward'):return c['lift_weight']
    if name in ('tilt_90','wrist_reconfigure','tilt_hold'):return c['turn_weight']
    return 1.

def weights_for_rows(details,params):
    chunks=[];tally={k:0 for k in ('hold_score','ending_score','turn_score')}
    for detail in details:
        from pathlib import Path
        path=Path(detail['path']);report_path=path.with_name('report.json')
        if _sha(path)!=detail['data_sha256'] or _sha(report_path)!=detail['report_sha256']:raise ValueError('Changed collection while loading rewards')
        r=_json(report_path)
        if r['role']!='collection' or r.get('eval_seeds')!=[]:raise ValueError('No validation trajectories in loss')
        training=set(r['train_seeds']);index={}
        for lane in r['per_lane']:
            seed=lane['seed']
            if seed not in training:continue  # no reward statistics or labels from held-out lanes
            if seed in index:raise ValueError('Duplicate row in training reward report')
            weight=1.
            for key,attribute in (('hold_score','ever_hold'),('ending_score','ending_hold'),('turn_score','reward_supported_turn')):
                result=lane.get(attribute)
                if type(result)!=bool:raise ValueError('Unmeasured training outcome: '+attribute)
                if result:tally[key]+=1;weight+=params[key]
            index[seed]=weight
        if set(index)!=training:raise ValueError('Missing training lane outcome')
        with np.load(path,allow_pickle=False) as z:
            seeds,heldout=z['seed'],z['val']
            if not np.array_equal(heldout,~np.isin(seeds,list(training))):raise ValueError('Reward/seed split mismatch')
            chunks.append(np.array([1. if v else index[int(s)] for s,v in zip(seeds,heldout)],dtype=np.float32))
    return np.concatenate(chunks),tally

class Weighted(Balanced):
    def __init__(self,x,y,mask,phase,v,device,weights,params):
        super().__init__(x,y,mask,phase,v,device)
        ids=np.flatnonzero(~v);ids=ids[np.argsort(phase[ids],kind='stable')]
        self.bonus=torch.as_tensor(weights[ids],device=device)
        labels=self.tables[0][1];self.importance=np.array([phase_weight(PHASES[int(i)],params) for i in labels])
    def counts(self,batch,step):
        n=len(self.importance)
        if batch<n:raise ValueError('Batch smaller than represented phases')
        if np.all(self.importance==self.importance[0]):
            base,extra=divmod(batch,n)
            return [base+int((j-step)%n<extra) for j in range(n)]
        mass=(batch-n)*self.importance/self.importance.sum()
        counts=np.floor(mass).astype(int);remain=batch-n-int(counts.sum())
        order=sorted(range(n),key=lambda j:(-(mass[j]-counts[j]),(j-step)%n))
        counts[order[:remain]]+=1
        return counts+1
    def draw(self,batch,step,gen):
        (x,y,m),labels,sizes,bounds=self.tables[0]
        idx=torch.cat([torch.randint(int(bounds[j]),int(bounds[j+1]),(int(n),),generator=gen,device=x.device)
                    for j,n in enumerate(self.counts(batch,step))])
        return x[idx],y[idx],m[idx],self.bonus[idx]

def imitation_loss(err,mask,weights):
    rows=(err.square().mul(mask).sum(-1)/mask.sum(-1))
    return (rows*weights).sum()/weights.sum()
