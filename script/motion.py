"""Motion transforms extracted from VLMasAM/motion_gen.py; see SOURCES.md."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from parse_util import MotionCommand
EPS = 1e-9

TCP_FRAME = 'paper_tcp_v1'
LEGACY_TCP_FRAME = 'hand_tcp_v1'
# Columns are paper TCP axes in panda_hand coordinates:
# X = hand +Y (jaw separation), Y = hand -X, Z = hand +Z (fingertips).
PAPER_AXES_IN_HAND = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def paper_tcp_pose(hand_pose, offset):
    pose = np.array(hand_pose, dtype=float, copy=True)
    pose[:3, 3] += offset * pose[:3, 2]
    pose[:3, :3] = pose[:3, :3] @ PAPER_AXES_IN_HAND
    return pose


def transform_motion_axes(motion, basis):
    return replace(motion,
        translation=replace(motion.translation, direction=basis @ motion.translation.direction),
        rotation=replace(motion.rotation, axis=basis @ motion.rotation.axis))


def paper_to_hand_motion(motion):
    return transform_motion_axes(motion, PAPER_AXES_IN_HAND)


def legacy_to_paper_motion(motion):
    return transform_motion_axes(motion, PAPER_AXES_IN_HAND.T)


def tcp_z_offset_transform(tcp_z_offset_m: float) -> np.ndarray:
    offset = float(tcp_z_offset_m)
    if not np.isfinite(offset):
        raise ValueError(f"TCP z offset must be finite, got {tcp_z_offset_m!r}.")
    transform = np.eye(4)
    transform[2, 3] = offset
    return transform

def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        raise ValueError(f"{name} cannot be zero.")
    return value / norm

def translation_delta_tcp_center_base(
    motion: MotionCommand,
    basis_base: np.ndarray,
) -> np.ndarray:
    distance_cm = float(motion.translation.distance_cm)
    if distance_cm < 0.0:
        raise ValueError(f"distance_cm must be non-negative, got {distance_cm}.")
    if distance_cm == 0.0:
        return np.zeros(3, dtype=float)

    direction = np.asarray(motion.translation.direction, dtype=float).reshape(3)
    if float(np.linalg.norm(direction)) < EPS:
        raise ValueError("translation direction cannot be zero when distance_cm is non-zero.")
    base_direction = np.asarray(basis_base, dtype=float) @ direction
    base_direction = _normalized(base_direction, "TCP-frame translation direction")
    return base_direction * (distance_cm / 100.0)

def rotation_delta_tcp_center_base(motion: MotionCommand, basis_base: np.ndarray) -> np.ndarray:
    angle_deg = float(motion.rotation.angle_deg)
    if angle_deg == 0.0:
        return np.eye(3)

    axis = np.asarray(motion.rotation.axis, dtype=float).reshape(3)
    if float(np.linalg.norm(axis)) < EPS:
        raise ValueError("rotation axis cannot be zero when angle_deg is non-zero.")
    axis_base = _normalized(np.asarray(basis_base, dtype=float) @ axis, "TCP-frame rotation axis")
    return axis_angle_to_rotation(axis_base, np.deg2rad(angle_deg))

def build_target_pose(
    current_ee_pose: np.ndarray,
    motion: MotionCommand,
    tcp_z_offset_m: float = 0.0,
) -> np.ndarray:
    """Convert a TCP-center-frame VLM command into an absolute EE/flange pose."""
    current = np.asarray(current_ee_pose, dtype=float)
    validate_transform(current, "current_ee_pose")

    T_ee_to_tcp = tcp_z_offset_transform(tcp_z_offset_m)
    T_tcp_to_ee = invert_transform(T_ee_to_tcp)
    current_tcp = current @ T_ee_to_tcp
    basis_base = current[:3, :3]

    target_tcp = current_tcp.copy()
    target_tcp[:3, 3] = current_tcp[:3, 3] + translation_delta_tcp_center_base(
        motion,
        basis_base,
    )

    R_delta_base = rotation_delta_tcp_center_base(motion, basis_base)
    target_tcp[:3, :3] = orthonormalize_rotation(R_delta_base @ current_tcp[:3, :3])
    target_tcp[3, :] = [0.0, 0.0, 0.0, 1.0]

    target_ee = target_tcp @ T_tcp_to_ee
    target_ee[:3, :3] = orthonormalize_rotation(target_ee[:3, :3])
    target_ee[3, :] = [0.0, 0.0, 0.0, 1.0]
    return target_ee

def validate_motion_limits(
    motion: MotionCommand,
    max_translation_cm: float,
    max_rotation_deg: float,
) -> None:
    distance_cm = float(motion.translation.distance_cm)
    angle_deg = abs(float(motion.rotation.angle_deg))
    if distance_cm > max_translation_cm:
        raise ValueError(
            f"Requested translation {distance_cm:.3f} cm exceeds "
            f"configured limit {max_translation_cm:.3f}."
        )
    if angle_deg > max_rotation_deg:
        raise ValueError(
            f"Requested rotation {angle_deg:.3f} deg exceeds "
            f"configured limit {max_rotation_deg:.3f}."
        )

def axis_angle_to_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float).reshape(3)
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    I = np.eye(3)
    return I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

def invert_transform(transform: np.ndarray) -> np.ndarray:
    T = np.asarray(transform, dtype=float)
    validate_transform(T, "transform")
    inv = np.eye(4)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv

def validate_transform(transform: np.ndarray, name: str) -> None:
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform, got shape {T.shape}.")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    det = float(np.linalg.det(T[:3, :3]))
    if not np.isclose(det, 1.0, atol=1e-2):
        raise ValueError(f"{name} rotation determinant should be near 1, got {det:.6f}.")

def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(rotation)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R
