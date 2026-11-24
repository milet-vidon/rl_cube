"""One-shot diagnostic: compare original CPU and CUDA contact labels at the first live finger contact.

Run as an independent probe with the current v3 sampler's --plan, a NEW --output
directory, and --steps. Its data is diagnostic and is never consumed by training.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

def verify_raw_frame(e):
    """Re-read exactly this PhysX substep into scratch accumulators; do not advance simulation."""
    import numpy as np
    import torch
    from contact_hold_wide_scene import Scene as Reference
    attrs=('contacts','feature_contacts','hand_impulse')
    raw=NS(n=e.n,contact_views=e.contact_views,sv_warp_contact=e.sv_warp_contact,lookup={},_lookup_ids=np.empty(0,np.uint64),cross=0)
    actual=NS(n=e.n,sv_warp_contact=e.sv_warp_contact,cross=0)
    for k in attrs:
        t=getattr(e,k)
        setattr(raw,k,np.zeros(tuple(t.shape),np.float64));setattr(actual,k,torch.zeros_like(t))
    Reference.read_contacts(raw)
    e.reducer.update(actual)
    result={}
    for k in attrs:
        a=getattr(actual,k).cpu().numpy();b=getattr(raw,k)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12);result[k]=float(np.abs(a-b).max())
    n=int((raw.feature_contacts>0).any(axis=1).sum())
    if not n:raise RuntimeError('Expected nonzero finger impulse in the audited raw frame')
    return dict(num_envs=e.n,nonzero_finger_lanes=n,max_abs_error=result,cross_lane_contacts=actual.cross+raw.cross)

def main(a):
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    from isaac_compat import close_application
    try:
        import torch
        from contact_hold_cuda_scene import Scene
        from contact_hold_cuda_sample import sample
        from contact_hold_loop_data import write
        original=Scene.read_contacts
        result={};steps=0
        def inspect(scene):
            nonlocal steps
            original(scene);steps+=1
            # Nonzero cumulative feature at its first occurrence => current raw substep contributed.
            if not result and bool((scene.feature_contacts>0).any()):
                result.update(verify_raw_frame(scene),physics_substep_including_reset=steps)
                write(a.output/'first_finger_audit.json',result)
                print('FIRST_FINGER_PARITY',result,flush=True)
        with patch.object(Scene,'read_contacts',inspect),torch.no_grad():sample(a)
        if not result:raise RuntimeError('No finger contact in this trial; no parity claim.')
    finally:close_application(app)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=550)
    main(p.parse_args())
