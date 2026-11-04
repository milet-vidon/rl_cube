"""Negative replay: retain original failure ordering under real geometry checks."""
from pathlib import Path
import json,time
import numpy as np
from opposed_grasp_plan import OppositionScene
from opposed_path_validation import PHASES,validate_paths,input_hash
from benchmark_opposed_path_validation import original

def main():
    root=Path(__file__).resolve().parent;run=root/'runs/teacher_joint_paced_passive_v13_20261103';request=json.loads((run/'opposition_request.json').read_text());plan=json.loads((run/'opposition_plan.json').read_text());scene=OppositionScene(request);names=request['joint_names']
    end=next(path[0] for phase,path in plan['phases'] if phase=='opposition_hold');target=scene.k.fk_link(dict(zip(names,end)),'right_palm_link')
    # A deliberately invalid right-arm endpoint, checked through the complete
    # original interpolation rather than bypassing geometry to fake a failure.
    bad=np.asarray([request['initial'][n] for n in names]);bad[names.index('Joint1_R')]=scene.k.limits['Joint1_R'][1]+.02
    phases=[(phase,[bad.tolist()]) for phase in PHASES]
    old,_,elapsed=original(request,phases,target);first=next(r for r in old if not r['valid']);serial=validate_paths(request,phases,target,1);parallel=validate_paths(request,phases,target,4)
    assert serial['predicate_results']==old==parallel['predicate_results']
    assert serial['first_failure']==first==parallel['first_failure']
    assert serial['failure']==parallel['failure'] and serial['checks']==first['index']+1==parallel['checks']
    report={'test':'deliberately invalid arm endpoint, full original interpolation','request_input_sha256':input_hash(request),'path_input_sha256':input_hash(phases),'exact_predicate_match':True,'exact_first_failure_match':True,'first_failure':first,'failure':parallel['failure'],'checks':parallel['checks'],'executed_checks':parallel['executed_checks'],'original_s':elapsed,'parallel_s':parallel['elapsed_s'],'predicate_results':old}
    out=root.parent/'work/teacher_optimization_20261103/path_validation_negative_v1.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='predicate_results'}),flush=True)

if __name__=='__main__':main()
