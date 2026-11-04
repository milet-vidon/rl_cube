"""Pure host math for an isolated native-initial-state probe, never a controller."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def checked_quaternion(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1.) > 1e-6:
        raise ValueError("Expected a finite unit quaternion in xyzw order")
    return value / np.linalg.norm(value)


def orientation_error_degrees(actual, expected):
    actual = Rotation.from_quat(checked_quaternion(actual))
    expected = Rotation.from_quat(checked_quaternion(expected))
    return float(np.degrees((actual * expected.inv()).magnitude()))


def face_axis(face):
    if face not in ("pX", "nX", "pY", "nY", "pZ", "nZ"):
        raise ValueError("Unknown face")
    result = np.zeros(3)
    result["XYZ".index(face[1])] = 1. if face[0] == "p" else -1.
    return result


def calibrate_native_axis(dof_index, joint_path, body_path, reference_xyzw, perturbed_xyzw, epsilon=1e-3):
    """Infer one native coordinate's axis/sign from actual FK response at reset."""
    if not 1e-5 <= epsilon <= .01:
        raise ValueError("Calibration perturbation outside small-angle protocol")
    reference = Rotation.from_quat(checked_quaternion(reference_xyzw))
    observed = Rotation.from_quat(checked_quaternion(perturbed_xyzw))
    derivative = (observed * reference.inv()).as_rotvec() / epsilon
    gain = float(np.linalg.norm(derivative))
    if not np.isfinite(gain) or abs(gain - 1.) > 1e-3:
        raise ValueError("Native coordinate did not produce the expected unit angular response")
    axis = derivative / gain
    dominant = int(np.argmax(np.abs(axis)))
    if np.max(np.abs(np.delete(axis, dominant))) > 1e-4:
        raise ValueError("Native axis is not aligned with a source Core-frame cardinal axis")
    return {"dof_index": int(dof_index), "joint_path": str(joint_path), "body_path": str(body_path),
            "axis_unit_core": axis.tolist(), "gain_rad_per_native_rad": gain,
            "dominant_axis": "XYZ"[dominant], "sign": 1 if axis[dominant] > 0 else -1}


def face_initial_coordinates(calibration, face, angle_degrees, num_dofs=66):
    """Only solved-reference, one-face pure-axis initial poses are supported."""
    if angle_degrees not in (90., 180.):
        raise ValueError("Use an isolated 90 or 180 degree initial-condition case")
    if len(calibration) != num_dofs or sorted(r["dof_index"] for r in calibration) != list(range(num_dofs)):
        raise ValueError("Require each of the 66 actual native DOF indices exactly once")
    axis = face_axis(face)
    grouped = {}
    for record in calibration:
        grouped.setdefault(record["body_path"], []).append(record)
    selected = sorted(path for path in grouped if face in path.rsplit("/", 1)[-1].split("_"))
    if len(grouped) != 26 or len(selected) != 9:
        raise ValueError("Expected 26 cubelets and nine initially selected face members")
    coordinates = np.zeros(num_dofs)
    assignment = []
    for body in selected:
        matches = []
        for record in grouped[body]:
            direction = np.asarray(record["axis_unit_core"], dtype=float)
            gain = record["gain_rad_per_native_rad"]
            if not np.isfinite(direction).all() or abs(np.linalg.norm(direction) - 1.) > 1e-6 or not math.isfinite(gain) or abs(gain - 1.) > 1e-3:
                raise ValueError("Invalid calibrated axis/gain")
            projection = float(np.dot(direction, axis))
            if abs(abs(projection) - 1.) <= 1e-4:
                matches.append((record, projection))
        if len(matches) != 1:
            raise ValueError("Expected exactly one calibrated collinear DOF for " + body)
        record, projection = matches[0]
        q = math.radians(angle_degrees) / (projection * record["gain_rad_per_native_rad"])
        if abs(q) > math.pi + 1e-5:
            raise ValueError("Finite coordinate would exceed spherical [-pi,pi] protocol domain")
        q = float(np.clip(q, -math.pi, math.pi))
        coordinates[record["dof_index"]] = q
        assignment.append({"body_path": body, "dof_index": record["dof_index"], "native_q_rad": q})
    return coordinates, assignment


def expected_body_orientation(reference_xyzw, face, angle_degrees, selected):
    reference = Rotation.from_quat(checked_quaternion(reference_xyzw))
    delta = Rotation.from_rotvec(face_axis(face) * math.radians(angle_degrees)) if selected else Rotation.identity()
    return (delta * reference).as_quat()


def transform_marker(actor_pose_xyz_xyzw, local_marker):
    pose = np.asarray(actor_pose_xyz_xyzw, dtype=float)
    marker = np.asarray(local_marker, dtype=float)
    if pose.shape != (7,) or marker.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(marker).all():
        raise ValueError("Invalid actor pose or local geometric marker")
    return pose[:3] + Rotation.from_quat(checked_quaternion(pose[3:])).apply(marker)


def face_marker_expectations(initial_markers_core, face, angle_degrees):
    rotation = Rotation.from_rotvec(face_axis(face) * math.radians(angle_degrees))
    selected = [path for path in initial_markers_core if face in path.rsplit("/", 1)[-1].split("_")]
    if len(selected) != 9:
        raise ValueError("Expected nine initially selected cubie center markers")
    result = {path: rotation.apply(point) if path in selected else np.asarray(point, dtype=float)
              for path, point in initial_markers_core.items()}
    return result
