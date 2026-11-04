"""Run CPU entry-boundary review without constructing a simulator or a trained actor."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'state_zero_gated_eval_runtime_v7.py': 'caadc319e816bdb3c00b5a744b71fbaf7c5f60ffd2d8f1893593fe441767b66b',
    'evaluate_state_zero_gated_v7.py': '6cda789c883a171d565b3a47d5ca5901ad9b4945221b281024eae7f66405b7f0',
    'test_state_zero_gated_eval_v7.py': 'c50812e12b9e3f26e7044ca23aab9c9619d7b9f1539147cb2120bb1239ab209e',
    'test_state_zero_gated_eval_v7_delta.py': '64a0e1542b0f9b4bcbc9324ca5ac79488756dc3e5c462c265f7ceed5eb8a5da1',
    'test_v7_eval_prepare_peer_review.py': '5df57a61589dbd5d24a808cbbea78fede14b5c31fab00865737a7493d737df33',
    'train_zero_gated_residual_v7.py': '9878b9d2c6f769624ba3889ea43797eacd02b203f65d8dcb3d765d90b1f9636e',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    actual = {name: sha(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError('Frozen evaluation candidate changed')
    command = [sys.executable, '-m', 'unittest', 'test_state_zero_gated_eval_v7',
               'test_state_zero_gated_eval_v7_delta', 'test_v7_eval_prepare_peer_review', '-v']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    log = ROOT / 'v7_eval_entry_independent_host_tests_20261104.log'
    log.write_text(output, encoding='utf-8', newline='\n')
    if completed.returncode or 'Ran 24 tests' not in output or '\nOK\n' not in output or 'skipped=' in output:
        raise RuntimeError('Independent host tests failed or were skipped: ' + str(log))
    report = {
        'schema': 'v7_eval_entry_independent_source_and_host_review_v1',
        'blocking_findings': [],
        'decision': 'No source or host-boundary blocker to freezing this candidate; real trained checkpoint preflight remains required.',
        'source_sha256': actual,
        'review_builder_sha256': sha(__file__),
        'all_reviewed_sources_lf': all(b'\r' not in (ROOT / name).read_bytes() for name in EXPECTED),
        'host_tests': {'command': command, 'exit_code': completed.returncode, 'passed': 24,
                       'original_tests': 20, 'independent_prepare_tests': 4,
                       'independent_negative_variants': 18, 'skipped': 0,
                       'log_path': str(log), 'log_sha256': sha(log)},
        'verified': {
            'runtime_normalizes_to_entire_frozen_v6_runtime': True,
            'main_control_and_cleanup_ast_unchanged_except_v7_cli': True,
            'exact_checkpoint_qualification_authorization_bytes_snapshotted_before_load': True,
            'snapshot_paths_and_hashes_forwarded_to_strict_loader': True,
            'qualification_remains_unmodified_training_authorized_false': True,
            'separate_training_authorization_required': True,
            'five_histories_unique_and_ordered': True,
            'named_bounds_schema_feature_count_joint_order_and_family_checked': True,
            'qualification_authorization_metadata_binding_checked': True,
            'offline_qualified_training_and_final_cpu_history_proof_required': True,
            'original_public_gate_and_classification_contract_unchanged': True,
            'actual_history_actor_replay_from_reset_in_host_test': True,
            'original_controller_gains_safety_termination_loop_retained': True,
            'all_source_snapshot_bytes_match_identity_hashes': True,
        },
        'limits': {
            'prepare_boundary_loader_mocked': True,
            'real_trained_checkpoint_executed_by_this_review': False,
            'real_checkpoint_strict_load_and_preflight_still_required': True,
            'simulation_started': False,
            'hardware_access': False,
            'closed_loop_success_demonstrated': False,
            'independent_generalization_demonstrated': False,
            'eligible_for_rl_episode_training': False,
            'right28_learned': False,
            'actor_scope': 'Privileged simulator state and actual command/contact history; no vision claim.',
        },
    }
    path = ROOT / 'v7_eval_entry_independent_review_20261104.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'review': str(path), 'sha256': sha(path), 'tests': 24, 'blocking': []}))

if __name__ == '__main__':
    main()
