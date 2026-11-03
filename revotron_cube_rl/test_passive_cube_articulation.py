"""Host-only contract checks; no Isaac imports or physics execution."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import passive_cube_articulation as facade


def fixture():
    paths = ["/World/Core"] + [f"/World/Cube/test_{i}" for i in range(26)]
    matrix = np.diag([1.0148148248845246e-5] * 3).reshape(9).tolist()
    data = {"status": "rigid_baseline_measured",
            "baseline_kind": "fresh_ordinary_rigid_objects_actual_backend_mass_properties",
            "source_assets_unchanged": True, "checks": {"real_baseline": {"passed": True}},
            "rigid_baseline": {path: {"mass": .0025370370130985975, "inertia": matrix,
                                      "local_com": [0., 0., 0., 0., 0., 0., 1.]} for path in paths}}
    masses = np.array([[data["rigid_baseline"][path]["mass"] for path in paths]])
    inertia = np.array([[data["rigid_baseline"][path]["inertia"] for path in paths]])
    coms = np.array([[data["rigid_baseline"][path]["local_com"] for path in paths]])
    return data, paths, masses, inertia, coms


class ContractTests(unittest.TestCase):
    def test_measured_inertia_accepts_positive_baseline_without_usd_override(self):
        data, paths, mass, inertia, coms = fixture()
        result = facade.validate_measured_mass_properties(data, paths, mass, inertia, coms)
        self.assertEqual(result["/World/Core"]["inertia"], inertia[0, 0].tolist())

    def test_zero_sentinel_and_hundred_fold_inertia_change_are_rejected(self):
        data, paths, mass, inertia, coms = fixture()
        for factor in (0., .01, 100.):
            changed = inertia.copy()
            changed[0, 0] *= factor
            with self.assertRaises(RuntimeError):
                facade.validate_measured_mass_properties(data, paths, mass, changed, coms)

    def test_changed_com_nonfinite_mass_and_wrong_root_are_rejected(self):
        data, paths, mass, inertia, coms = fixture()
        changed = coms.copy(); changed[0, 12, 0] = .001
        with self.assertRaisesRegex(RuntimeError, "COM"):
            facade.validate_measured_mass_properties(data, paths, mass, inertia, changed)
        changed_mass = mass.copy(); changed_mass[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            facade.validate_measured_mass_properties(data, paths, changed_mass, inertia, coms)
        with self.assertRaisesRegex(RuntimeError, "Core"):
            facade.validate_measured_mass_properties(data, paths[1:] + paths[:1], mass, inertia, coms)

    def test_com_rotation_accepts_quaternion_sign_but_rejects_changed_frame(self):
        data, paths, mass, inertia, coms = fixture()
        negative = coms.copy(); negative[0, :, 3:7] *= -1
        facade.validate_measured_mass_properties(data, paths, mass, inertia, negative)
        for quaternion in ([1., 0., 0., 0.], [0., 0., 0., 2.], [0., 0., 0., float("nan")]):
            changed = coms.copy(); changed[0, 0, 3:7] = quaternion
            with self.assertRaises(RuntimeError):
                facade.validate_measured_mass_properties(data, paths, mass, inertia, changed)

    def test_control_phase_state_write_is_rejected_before_backend_access(self):
        handle = object.__new__(facade.PassiveCubeHandle)
        handle.allow_pose_reset = False
        with self.assertRaisesRegex(RuntimeError, "episode reset"):
            handle.write_root_state_to_sim(None)

    def test_evidence_hash_and_completed_measurement_required(self):
        data, *_ = fixture()
        workspace = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(workspace))
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(data))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(facade.load_baseline(path, digest)["evidence_sha256"], digest)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                facade.load_baseline(path, "0" * 64)
            data["source_assets_unchanged"] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "complete"):
                facade.load_baseline(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_factory_wrapper_order_and_exception_restore(self):
        class DynamicCubeHandle:
            def __init__(self, *args):
                raise AssertionError("Original 27-RigidObject factory must never be called")

        module = SimpleNamespace(DynamicCubeHandle=DynamicCubeHandle, spawn_dynamic_cube=object())
        old_core = lambda *args: "unsafe Core mutation"
        old_damping = lambda *args: "unsafe D6 mutation"
        legacy_core = SimpleNamespace(apply_source_core_inertia=old_core)
        legacy_damping = SimpleNamespace(apply_source_joint_damping=old_damping)
        marker = object()
        with patch.dict("sys.modules", {"cube_core_inertia": legacy_core, "cube_joint_damping": legacy_damping}), patch.object(facade, "load_baseline", return_value={}), patch.object(facade, "PassiveCubeHandle", return_value=marker) as constructor:
            with self.assertRaisesRegex(RuntimeError, "caller failure"):
                with facade.passive_cube_articulation("unused", "unused", module=module):
                    original_passive = module.DynamicCubeHandle
                    # Existing material wrappers capture the new factory and remain active.
                    module.DynamicCubeHandle = lambda *args: (original_passive(*args), "material applied")
                    result = module.DynamicCubeHandle("stage", "asset", (0, 0, 1), (.9, .8))
                    self.assertEqual(result, (marker, "material applied"))
                    constructor.assert_called_once()
                    with self.assertRaisesRegex(ValueError, "cannot be combined"):
                        legacy_core.apply_source_core_inertia("stage", "asset")
                    with self.assertRaisesRegex(ValueError, "cannot be combined"):
                        legacy_damping.apply_source_joint_damping("stage", "asset")
                    raise RuntimeError("caller failure")
            self.assertIs(module.DynamicCubeHandle, DynamicCubeHandle)
            self.assertIs(legacy_core.apply_source_core_inertia, old_core)
            self.assertIs(legacy_damping.apply_source_joint_damping, old_damping)
            module.DynamicCubeHandle = lambda *args: marker
            with self.assertRaisesRegex(ValueError, "outside"):
                with facade.passive_cube_articulation("unused", "unused", module=module):
                    pass

    def test_explicit_legacy_options_rejected_before_importing_environment(self):
        for options in ({"source_core_inertia": True}, {"source_joint_damping": True}):
            with self.assertRaisesRegex(ValueError, "legacy"):
                with facade.passive_cube_articulation("unused", "unused", **options):
                    pass

    def test_dependency_change_is_rejected_not_only_asset_change(self):
        workspace = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(workspace))
            dependency = Path(directory) / "source.xml"
            dependency.write_text("original")
            digest = facade.sha(dependency)
            data = {"matching_fingerprint": {"source_sha256": {str(dependency): digest},
                    "physics_implementation_sha256": {str(dependency): digest}}}
            facade.verify_baseline_inputs(data)
            dependency.write_text("changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                facade.verify_baseline_inputs(data)


if __name__ == "__main__":
    unittest.main()
