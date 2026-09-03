"""Motion features for sequential sign recognition."""

from __future__ import annotations

import numpy as np

from sign_ai.config import LANDMARK_ONLY, LANDMARK_PLUS_VELOCITY


def velocity(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == 0:
        return sequence
    delta = np.zeros_like(sequence, dtype=np.float32)
    delta[1:] = sequence[1:] - sequence[:-1]
    return delta


def acceleration(sequence: np.ndarray) -> np.ndarray:
    return velocity(velocity(sequence))


def add_motion_features(sequence: np.ndarray, include_acceleration: bool = False) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    parts = [sequence, velocity(sequence)]
    if include_acceleration:
        parts.append(acceleration(sequence))
    return np.concatenate(parts, axis=1).astype(np.float32)


def apply_feature_mode(sequence: np.ndarray, feature_mode: str = LANDMARK_ONLY) -> np.ndarray:
    mode = (feature_mode or LANDMARK_ONLY).strip().lower()
    if mode == LANDMARK_PLUS_VELOCITY:
        return add_motion_features(sequence, include_acceleration=False)
    if mode == LANDMARK_ONLY:
        return np.asarray(sequence, dtype=np.float32)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def movement_magnitude(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape:
        return 0.0
    return float(np.mean(np.abs(current - previous)))
