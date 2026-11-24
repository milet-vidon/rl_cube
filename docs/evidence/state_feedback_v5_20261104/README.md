# State-history residual v5 development training

Both GPU fits use the same reviewed 1648 safe BC transitions from the completed v3 mixture (1419 teacher, 229 student); raw run eligibility flags remain false. This is one development episode with no held-out episode or independent generalization result. No closed-loop model success is established by these files.

The frozen v4 phase basis is augmented by a GRU64 with 284 actual-state/public-phase features and bounded left28 residuals. Expert values, labels, and previous action source are excluded from actor inputs; right28 remains outside the learned model. Projection reproduces the original float32 command path and limits. All 28 label reachable-distance maxima were exactly zero in inspect02; no label clipping.

The 200-epoch fit (6.255s GPU training loop) reduced RMSE from 0.000671184 to 0.000119426 rad, max error 0.00439084 rad. The separate 2000-epoch fit (24.483s) reached RMSE 0.00000807078 rad, max 0.000395268 rad; its best epoch is 1998. These timings exclude CPU data/provenance inspection. The long model was selected for the first closed-loop evaluation using only development fit, before seeing any closed-loop result. Both checkpoints are preserved.

Dataset snapshots and hashes are in ../left_dagger_training_20261104. Exact source, qualification, commands, training journals and model bytes are archived here. This state policy uses privileged simulation state, not camera vision, and covers pickup through wrist hold rather than layer turning.
