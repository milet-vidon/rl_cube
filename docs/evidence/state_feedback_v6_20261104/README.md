# V6 public-phase-gated residual: completed training and failed evaluations

Both completed training runs and their evaluated checkpoints are retained. The short run (2,000 epochs, best 1,999) achieved development projected-label RMSE 1.28769e-5 rad, maximum error 0.000521600 rad. The long run (10,000 epochs, best 9,995) achieved 2.03165e-6 rad, maximum error 0.000142217 rad. These are offline fitting metrics on two development histories, not closed-loop or independent-generalization success.

Both from-reset seed 9501 evaluations failed the original contact-topology/settle timeout after 752 completed physical control steps. Maximum lifts were approximately 0.187796 m (short) and 0.187920 m (long). Original wrist qualification and training/whole-neural/runtime-bridge claims remain false. Complete abort/unfinished-decision evidence remains in each original evaluation journal, with no physics step after the abort.

The policy controls left position targets from reset using actual privileged simulator state/history. Original public phase scheduling, right actions, and the teacher gain contract remain external; neither failed evaluation reached a gain event. There are zero left teacher position action steps and zero cube pose writes during control. This is not a visual policy, reward learning, face-turn validation, or cube solution.

The qualification pins exactly two development histories: v3 mixture supervision and the safe executed prefix of a failed v5 evaluation. Failure and eligibility flags on original data are not rewritten. Masked float32 label floors are retained. The independent completed-checkpoint audit applies to the short trained checkpoint; each longer result remains labeled with its own original scope and hashes.

The exact 11 installed V6 source/test files, required frozen dependency sources, review/qualification files, all four complete run directories, model/checkpoint bytes, training journals, evaluation trajectories, protocols, initial states, scenes, plans, per-step and abort records are preserved. The installed unittest log records 46 tests PASS. The separately retained older `v6_final_tests_20261104.log` records missing pytest and must not be cited as a passing result.

`run_inventories.json` covers every original file in all four completed runs. Deterministic gzip and `archive_manifest.json` preserve original/compressed/restored SHA-256, with each source checked again after copy. No incomplete evaluation artifact is included. This archive operation starts no simulator and performs no git stage, commit, or push.
