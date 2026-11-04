# Native index continuation v20

V19 passed pickup and centering but lost index contact after exhausting the -.012rad feedback range. It terminated after3,019 steps when ring contact was also lost during right-hand closure; no turn occurred.

V20 changes only the index MPR feedback lower bound to -.023rad. The original5e-5rad step, impulse thresholds, center handover, ring controller, joint limits, projection, native passive cube and complete task gates remain. Activation requires both --left-index-continuation and --left-index-continuation-v20. The original V19 mode remains available.

The wider -.032 candidate was rejected by the unchanged stationary-layer penetration gate. Actual late-state mesh measurements also show that a one-point normal displacement estimate does not establish contact recovery. The selected smaller range is tested from reset to see whether earlier support prevents cube drift; it is not asserted to recover an already-drifted state.

The selected candidate passed1,502 full original path samples and28 index×7 ring values:51,264 factored checks, equivalent to294,392 full Cartesian checks. Twelve workers took253.04s. Dependency independence, original predicates,12 source hashes and190 asset hashes were independently verified. These remain sampled geometry checks, not continuous collision or physical success guarantees. Thirteen candidate host tests and9 installed feedback tests passed.

The single physical run teacher_native_index_extended_v20_20261104 is running as PID4125606; inspect its actual result and process before starting another Isaac instance. It uses the original native baseline and full-task episode contract, with no control-period cube pose writes. Source, rejected candidates, full sampled checks and reproduction launch are under docs/evidence/native_index_v20_candidate_20261104.

The separate left state-policy work continues: both v6 variants failed the primary-lift contact gate. New v7 supervision includes the successful matchedV4 actual history and failed-policy states. Both initial-state teacher collection attempts9502/9503 stopped at the original wrist IK envelope, after1,574/1,505 physical steps respectively; no original result was fabricated. Their error, abort, actual trajectory and schedule audits are archived under docs/evidence/teacher_initial_state_failures_20261104.
