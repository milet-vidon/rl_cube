# Native FullTask v18 and finite90 v1 failed evidence

v18 completed 3017 control steps and terminated normally with
`left_grasp_lost_during_opposition`. Pickup and centered transport passed;
opposed grasp did not pass and face turning never started. The task outcome is
failure, with no transitions after termination. This is not a trained policy.

finite90 v1's paired ordinary-body baseline completed. Its passive-articulation
candidate failed the initial center-marker permutation check: 2.5946666986 µm
versus the unchanged 1 µm bound. The 0.0055251462° orientation error passed.
The failure occurs before the first candidate physics step. Host-review PASS
does not mean initialization, finite dynamics, or contact-driven turning passed.

All completed run files, source snapshots, configuration, original manifests,
native verification, finite calibration/results, audits and launch evidence
are preserved. JSON/JSONL members at least 512 KiB use deterministic gzip.
`archive_manifest.json` records original/compressed/decompressed SHA256 and byte
counts. Original manifests remain byte-identical and reference original names;
decompress the corresponding `.gz` member to recover those exact bytes.
`failure_summary.json` records explicit checks and external asset hashes.
No old result or source was modified; no git staging/commit was performed.
