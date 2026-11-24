# Left DAgger evidence, 2026-11-04

The v1 preserve attempt failed before any control action because strict JSON rejected a legitimate unbounded USD joint limit. The original error and zero-row trajectory are retained.

The metadata-only v2 preserve run executed 1647 original teacher actions on the selected baseline ordinary 27-body cube. `bridge_validation_v2.json` independently compares every control target, joint position and cube position with the accepted source, recomputes the original wrist/handover gate, and audits done signals and the original gain change. The frozen result is not edited. This validates the single-seed teacher adapter, not a learned policy or generalization.

The v3 run, once present, is an explicit teacher/student mixture with left-only v4 behavior, actual-state expert queries, projected expert supervision and conservative termination on every original done/drop/timeout. Teacher interventions and failed or incomplete episodes remain visible. `eligible_training_episode` and `pure_student_episode_claim` remain false; transition use requires a separate data audit. These data must not be pooled with D6-damped or native-articulation physics.

Each run contains the original manifest, source snapshot, plans and result/error. Large files use deterministic gzip; `archive_file_index.json` records both compressed and original hashes. Decompress before using the v3 provenance reader. The archive-level manifest verifies archived bytes. `archive_status.json` states whether v3 is still pending. No checkpoint is duplicated: the v4 behavior checkpoint SHA256 is recorded there.

This archive does not claim free face rotation, center-policy completion, visual solving, PPO success, or an independent test seed. No original run or frozen result is overwritten.
