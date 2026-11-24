"""CPU-only preflight of combined original index/ring feedback ranges."""
import json,sys,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from left_support_envelope_v2 import validate_plan
if __name__=='__main__':
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
        result=validate_plan(data['request'],data['plan'],data['baseline_target'],data['index_state'],workers=12)
    except Exception as error:
        result={'passed':False,'error':str(error),'physics_validated':False}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
    print('LEFT_SUPPORT_ENVELOPE',result['passed'],flush=True)
    raise SystemExit(0 if result['passed'] else 1)
