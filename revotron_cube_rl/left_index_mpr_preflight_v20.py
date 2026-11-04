"""Independent host-only entry, matching host_plan(request_file, output_file)."""
import json
import os
import sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from left_index_mpr_envelope_v20 import validate_plan

if __name__ == '__main__':
    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        result = validate_plan(request['request'], request['plan'], request['baseline_target'],
                               request['index_state'], workers=request['request'].get('refinement_options', {}).get('preflight_workers', 12))
    except Exception as error:
        result = dict(passed=False, error=str(error), physics_validated=False)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('V20_MPR_PREFLIGHT', result['passed'], result.get('checks'), result.get('elapsed_s'), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
