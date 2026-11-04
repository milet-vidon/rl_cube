# Native index V20 candidate

The only control change extends the index MPR lower feedback offset from -.012 to -.023 rad. Original step size, impulse thresholds, center handover, runtime safety, native physics and task gates remain unchanged. Source snapshots, patch, tests, full sampled geometry, negative candidates and the exact launch are archived here.

The -.032 candidate was rejected by the original stationary-layer geometry gate. The chosen -.023 candidate passed 51,264 factored checks over 1,502 original path samples (equivalent full grid 294,392), with 12 workers in 253.04 seconds. These are sampled geometry checks, not continuous collision guarantees or physical success. The purpose is to maintain index contact earlier from reset; late-state mesh measurements do not show that this bound can recover an already-drifted cube.

13 host candidate tests passed. The installed feedback module's 9 tests also passed; its exact log is included. Runtime sources and all bound geometry dependencies remain pinned. The full physical experiment teacher_native_index_extended_v20_20261104 is running separately and is not included as a completed result in this candidate archive.

Each gzip file restores exactly the listed source bytes. The manifest binds raw and compressed content; negative results are retained. No checkpoint or physical-success claim is attached to this candidate.
