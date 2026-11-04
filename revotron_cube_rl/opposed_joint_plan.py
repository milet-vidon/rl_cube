"""Joint arm/finger planner, using the unchanged posture/collision/path gates."""
import json,sys
from pathlib import Path
from opposed_grasp_plan import plan
from opposed_joint_fit import refine
if __name__=='__main__':
    try:result=plan(json.loads(Path(sys.argv[1]).read_text()),refine=refine)
    except Exception as e:result={'passed':False,'error':str(e),'physics_validated':False}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
    print('JOINT_OPPOSITION_PLAN',result['passed'],flush=True)
    raise SystemExit(0 if result['passed'] else 1)
