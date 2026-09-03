"""Normalize MediaPipe landmark vectors while preserving the 1662-value shape."""

from __future__ import annotations

import numpy as np

from sign_ai.config import FACE_VALUES, HAND_VALUES, KEYPOINT_LENGTH, POSE_VALUES


EPSILON = 1e-6
POSE_SLICE = slice(0, POSE_VALUES)
FACE_SLICE = slice(POSE_VALUES, POSE_VALUES + FACE_VALUES)
LEFT_HAND_SLICE = slice(POSE_VALUES + FACE_VALUES, POSE_VALUES + FACE_VALUES + HAND_VALUES)
RIGHT_HAND_SLICE = slice(POSE_VALUES + FACE_VALUES + HAND_VALUES, KEYPOINT_LENGTH)


def _fit_flat(values: np.ndarray, length: int = KEYPOINT_LENGTH) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).flatten()
    if values.size == length:
        return values
    if values.size > length:
        return values[:length]
    return np.pad(values, (0, length - values.size)).astype(np.float32)


def _normalize_hand(hand_values: np.ndarray) -> np.ndarray:
    hand = _fit_flat(hand_values, HAND_VALUES).reshape(21, 3)
    if not np.any(hand):
        return hand.reshape(-1).astype(np.float32)

    wrist = hand[0].copy()
    centered = hand - wrist
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < EPSILON:
        scale = 1.0
    return (centered / scale).reshape(-1).astype(np.float32)


def _normalize_pose(pose_values: np.ndarray) -> np.ndarray:
    pose = _fit_flat(pose_values, POSE_VALUES).reshape(33, 4)
    if not np.any(pose[:, :3]):
        return pose.reshape(-1).astype(np.float32)

    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    left_hip = pose[23, :3]
    right_hip = pose[24, :3]
    torso_points = np.array([left_shoulder, right_shoulder, left_hip, right_hip], dtype=np.float32)
    visible = np.any(torso_points, axis=1)

    origin = torso_points[visible].mean(axis=0) if np.any(visible) else pose[0, :3]
    centered_xyz = pose[:, :3] - origin
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    torso_height = float(np.linalg.norm(((left_shoulder + right_shoulder) / 2.0) - ((left_hip + right_hip) / 2.0)))
    scale = max(shoulder_width, torso_height, float(np.max(np.linalg.norm(centered_xyz, axis=1))), EPSILON)

    output = pose.copy()
    output[:, :3] = centered_xyz / scale
    return output.reshape(-1).astype(np.float32)


def normalize_keypoints(keypoints: np.ndarray, include_face: bool = True) -> np.ndarray:
    """Normalize pose and hands without changing the vector length."""
    values = _fit_flat(keypoints)
    pose = _normalize_pose(values[POSE_SLICE])
    face = values[FACE_SLICE].astype(np.float32) if include_face else np.zeros(FACE_VALUES, dtype=np.float32)
    left_hand = _normalize_hand(values[LEFT_HAND_SLICE])
    right_hand = _normalize_hand(values[RIGHT_HAND_SLICE])
    return np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)


def normalize_sequence(sequence: np.ndarray, include_face: bool = True) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    return np.array([normalize_keypoints(frame, include_face=include_face) for frame in sequence], dtype=np.float32)


def has_hand_keypoints(keypoints: np.ndarray) -> bool:
    values = _fit_flat(keypoints)
    return bool(np.any(values[LEFT_HAND_SLICE]) or np.any(values[RIGHT_HAND_SLICE]))

