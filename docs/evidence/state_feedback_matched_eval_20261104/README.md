# Matched v5 / v4 left-policy evaluations, 2026-11-04

Both completed runs use development seed 9501, the same actual physics family and byte-identical initial state, and the original safety/contact/wrist gates. v5 failed the original contact-topology/settle gate after 619 physical steps; matched v4 passed the original wrist gate after 1,654. No left teacher position action executed in either run. v5 recorded zero original teacher gain events; v4 recorded one first-tilt_hold K20 gain event.

The public phase controller, right-hand commands, and gain schedule remain external. These are privileged simulator state policies, not visual policies or whole-robot neural policies. `pure_student_episode_claim`, `whole_robot_pure_neural_episode`, `runtime_bridge_validated`, and training eligibility remain false. Final original environment `raw_done` is false in both: v5 has a controller abort, and v4 stops at the scoped wrist evaluation milestone. Neither is evidence of a successful face turn or cube solution.

Every file from both completed run directories is preserved, including exact checkpoints, source snapshots, protocols, scene, initial state, complete trajectories, per-policy-step transitions, the v5 abort, and results. The two final CPU actor replays reproduce all 619 / 1,654 logged feature vectors and model outputs exactly; this is software/log consistency, not simulator rerunning or independent generalization.

The preserved public-phase mask audit was made from 1,143 available v4 transitions and compares all 512 observed masked frames. It is historical development evidence only. Nonzero corrective labels in the failed v5 prefix remain recorded; they were not removed to make the mask fit.

`archive_manifest.json` lists source, gzip, and restored SHA-256 values. Every gzip uses original bytes, deterministic metadata, and post-copy source hash verification. `evaluation_summary.json` includes original result/protocol values and exact nontransition abort/final events. No training source was edited, no simulator launched, and no git files staged or committed by this archive operation.
