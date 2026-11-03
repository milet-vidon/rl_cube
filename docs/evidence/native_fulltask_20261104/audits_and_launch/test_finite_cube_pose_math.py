import copy
import itertools
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import finite_cube_pose_math as math_helpers


def calibration_fixture():
    records = []
    for components in itertools.product((-1, 0, 1), repeat=3):
        if components == (0, 0, 0):
            continue
        tokens = [("p" if value > 0 else "n") + axis for value, axis in zip(components, "XYZ") if value]
        name = "_".join(tokens)
        axes = [int(np.flatnonzero(components)[0])] if len(tokens) == 1 else [0, 1, 2]
        for axis in axes:
            direction = np.eye(3)[axis]
            # Nontrivial signs ensure the helper uses measured response, not guessed ordering.
            sign = -1 if len(records) % 2 else 1
            perturbation = Rotation.from_rotvec(sign * direction * .001).as_quat()
            records.append(math_helpers.calibrate_native_axis(len(records), "/World/Joints/" + name,
                "/World/Cube/" + name, [0, 0, 0, 1], perturbation))
    return records


class FinitePoseTests(unittest.TestCase):
    def test_all_faces_at_ninety_and_one_eighty_use_measured_signs(self):
        records = calibration_fixture()
        self.assertEqual(len(records), 66)
        for face in ("pX", "nX", "pY", "nY", "pZ", "nZ"):
            for angle in (90., 180.):
                q, assignments = math_helpers.face_initial_coordinates(records, face, angle)
                self.assertEqual(len(assignments), 9)
                self.assertEqual(np.count_nonzero(q), 9)
                for assignment in assignments:
                    record = records[assignment["dof_index"]]
                    observed = Rotation.from_rotvec(np.asarray(record["axis_unit_core"]) * q[record["dof_index"]] * record["gain_rad_per_native_rad"]).as_quat()
                    expected = math_helpers.expected_body_orientation([0, 0, 0, 1], face, angle, True)
                    self.assertLess(math_helpers.orientation_error_degrees(observed, expected), 1e-5)

    def test_quaternion_sign_is_equivalent_and_invalid_quaternion_rejected(self):
        q = Rotation.from_rotvec([0, 0, np.pi]).as_quat()
        self.assertLess(math_helpers.orientation_error_degrees(q, -q), 1e-8)
        with self.assertRaises(ValueError):
            math_helpers.checked_quaternion(q * 2)
        with self.assertRaises(ValueError):
            math_helpers.checked_quaternion([0, 0, 0, float("nan")])

    def test_incorrect_native_axis_response_and_stale_fk_rejected(self):
        with self.assertRaises(ValueError):
            math_helpers.calibrate_native_axis(0, "j", "b", [0, 0, 0, 1], [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            math_helpers.calibrate_native_axis(0, "j", "b", [0, 0, 0, 1], Rotation.from_rotvec(np.ones(3) / np.sqrt(3) * .001).as_quat())

    def test_duplicate_index_and_missing_face_axis_rejected(self):
        records = calibration_fixture()
        bad = copy.deepcopy(records); bad[1]["dof_index"] = bad[0]["dof_index"]
        with self.assertRaises(ValueError):
            math_helpers.face_initial_coordinates(bad, "nZ", 90.)
        bad = copy.deepcopy(records)
        for record in bad:
            if record["body_path"] == "/World/Cube/nZ":
                record["axis_unit_core"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            math_helpers.face_initial_coordinates(bad, "nZ", 90.)

    def test_actor_origin_is_not_used_as_cubie_center(self):
        pose = [0., 0., 1., *Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat()]
        center = math_helpers.transform_marker(pose, [.019, -.019, -.019])
        np.testing.assert_allclose(center, [.019, .019, .981], atol=1e-12)

    def test_marker_permutation_and_nonselected_preservation(self):
        markers = {}
        for record in calibration_fixture():
            name = record["body_path"].rsplit("/", 1)[-1]
            point = np.zeros(3)
            for token in name.split("_"):
                point["XYZ".index(token[1])] = .019 * (1 if token[0] == "p" else -1)
            markers[record["body_path"]] = point
        for angle in (90., 180.):
            expected = math_helpers.face_marker_expectations(markers, "nZ", angle)
            for name, point in expected.items():
                self.assertLess(min(np.linalg.norm(point - other) for other in markers.values()), 1e-12)
                if "nZ" not in name.rsplit("/", 1)[-1].split("_"):
                    np.testing.assert_array_equal(point, markers[name])


if __name__ == "__main__":
    unittest.main()
