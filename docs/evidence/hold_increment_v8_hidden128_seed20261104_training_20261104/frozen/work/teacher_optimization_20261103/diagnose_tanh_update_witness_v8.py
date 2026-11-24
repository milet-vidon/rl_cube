"""Offline finite-logit witness diagnostics; never changes labels or qualification."""
from copy import deepcopy
from pathlib import Path
import argparse,json,time
import torch
import train_hold_increment_v8 as t


def neighbors(value):
    values=[value];low=high=value
    for _ in range(2):
        low=torch.nextafter(low,torch.full_like(low,float('-inf')))
        high=torch.nextafter(high,torch.full_like(high,float('inf')))
        values.extend((low,high))
    return values


def one(data,model,device):
    x,previous,labels,mask=t.prior.frozen.tensor_episode(data,device)
    with torch.no_grad():
        base=model.baseline_targets(x,previous);anchor=torch.where(model.update_anchor_is_previous,previous,base)
        bounds=model.effective_bounds(x)
        project=lambda v:t.policy.project_left_targets(v,previous,data['parameters'],data['left_indices'],data['projector'])
        classes=t.policy.classify_branch_labels(labels,project(base),project(previous))
        required=(classes['class_targets']==t.policy.UPDATE)&model.residual_active(x).unsqueeze(-1)&mask.unsqueeze(-1)
        best=torch.full_like(labels.double(),float('inf'));exact=torch.zeros_like(required)
        min_abs=torch.zeros_like(labels);selected_logit=torch.zeros_like(labels);candidate_count=0
        # First invert a concrete float32 increment, then examine logit ULPs.
        # At a mathematical endpoint use finite magnitude 10; finite float32
        # tanh may saturate exactly. Such cases are reported, never presumed.
        for delta in neighbors((labels-anchor).float()):
            allowed=delta.abs()<=bounds
            ratio=delta.double()/bounds.double()
            interior=ratio.abs()<1
            safe=ratio.clamp(-1+1e-12,1-1e-12)
            logit=torch.where(interior,torch.atanh(safe),torch.sign(ratio)*10.).float()
            for candidate in neighbors(logit):
                update=torch.tanh(candidate)*bounds
                output=project(torch.clamp(anchor+update,model.baseline.lower.float(),model.baseline.upper.float()))
                distance=(output.double()-labels.double()).abs()
                admissible=allowed&torch.isfinite(candidate)&(update.abs()<=bounds)
                improve=admissible&(distance<best)
                min_abs=torch.where(improve,output,min_abs);selected_logit=torch.where(improve,candidate,selected_logit)
                best=torch.where(improve,distance,best)
                exact|=admissible&t.element_bits(output,labels);candidate_count+=1
        t.require(bool(torch.isfinite(best[required]).all()),'No finite-logit witness candidate for eligible UPDATE label')
    return {'episode_id':data['episode_id'],'required_update_or_ambiguous_elements':int(required.sum()),
        'exact_witness_not_found_elements':int((required&~exact).sum()),
        'max_minimum_projected_absolute_difference_rad':float(best[required].max())if required.any()else 0.,
        'candidate_combinations_per_element':candidate_count,
        'nonexact_by_joint':(required&~exact).sum((0,1)).tolist()},(required.cpu(),exact.cpu(),min_abs.cpu(),selected_logit.cpu())


def diagnose(episodes,model,device):
    other=deepcopy(model).to(device);reports=[]
    for data in episodes:
        cpu,ca=one(data,model,'cpu');gpu,ga=one(data,other,device)
        t.require(torch.equal(ca[0],ga[0]),'Required UPDATE class differs by device')
        required=ca[0]
        reports.append({'episode_id':data['episode_id'],'cpu':cpu,'training_device':gpu,
            'exact_witness_status_differences':int(((ca[1]!=ga[1])&required).sum()),
            'selected_projected_candidate_different_float32_elements':int((~t.element_bits(ca[2],ga[2])&required).sum()),
            'selected_logit_different_float32_elements':int((~t.element_bits(ca[3],ga[3])&required).sum())})
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    t.require(not a.output.exists(),'New diagnostic output required');torch.set_num_threads(1)
    started=time.monotonic();root=Path.cwd();source=t.source_hashes();baseline,bh=t.prior.frozen.load_baseline(root/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt')
    episodes,_=t.load_histories([root/'runs'/n for n in t.RUN_NAMES],baseline,bh);model=t.build_model(baseline,episodes,64)
    reports=diagnose(episodes,model,a.device)
    t.require(source==t.source_hashes(),'Sources changed during diagnostic')
    t.dump(a.output,{'schema':'v8_finite_tanh_logit_witness_diagnostic_v1','source_sha256':source,
        'diagnostic_source_sha256':t.sha(__file__),'device':a.device,'elapsed_s':time.monotonic()-started,
        'dataset_evidence':[d['evidence']for d in episodes],'runs':reports,
        'all_original_labels_preserved':True,'qualification_changed':False,'optimizer_created':False,'physical_success_claim':False,
        'scope':'Finite tanh parameterization witnesses around algebraic inversion; misses remain regression targets, not clipped labels or proven unrepresentability.'})
    print('V8_TANH_WITNESS_DIAGNOSTIC_COMPLETE',t.sha(a.output),flush=True)


if __name__=='__main__':main()
