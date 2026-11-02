import json
import math
from pathlib import Path

import torch

from joint_safety import max_target_delta, soft_limits_from_hard

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
names = cfg["joint_names"]
hard = torch.tensor([[[-3.1067, 3.1067] if n.startswith("Joint1") or n.startswith("Joint3") or n.startswith("Joint5") else
                     [-2.0944, 2.0944] if n.startswith("Joint2") else
                     [-2.5307, 1.0472] if n.startswith("Joint4") else
                     [-1.0472, 1.0472] if n.startswith("Joint6") else
                     [-1.5708, 1.5708] if n.startswith("Joint7") else
                     [-1., 1.] for n in names]], dtype=torch.float32)

soft = soft_limits_from_hard(hard, names, cfg)
for side in ("L", "R"):
    expected = {
        "Joint1": (-178., 178.), "Joint2": (-120., 120.), "Joint3": (-178., 178.),
        "Joint4": (-145., 60.), "Joint5": (-178., 178.), "Joint6": (-60., 60.), "Joint7": (-90., 90.),
    }
    for j, (lo, hi) in expected.items():
        i = names.index(f"{j}_{side}")
        actual = tuple(math.degrees(float(x)) for x in soft[0, i])
        # Intersection with USD hard limits may clip a few last decimal places.
        assert actual[0] >= lo - 1e-3 and actual[1] <= hi + 1e-3, (j, actual)
        assert actual[0] < actual[1]

scales = torch.tensor([[.025 if n.startswith("Joint") else .06 for n in names]])
step = max_target_delta(names, scales, cfg)
for i, n in enumerate(names):
    if n.startswith("Joint"):
        assert float(step[0, i]) <= cfg["arm_max_step_rad"] + 1e-7
assert all(float(cfg["home"][n]) >= float(soft[0, i, 0]) and float(cfg["home"][n]) <= float(soft[0, i, 1]) for i, n in enumerate(names) if n.startswith("Joint"))
print("ARM_SAFETY_TESTS_PASSED")
