"""Utilities for variable-length landmark sequences."""

from __future__ import annotations

import numpy as np


def trim_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) <= target_length:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, target_length).round().astype(int)
    return sequence[indices]


def pad_sequence(sequence: np.ndarray, target_length: int, mode: str = "edge") -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) >= target_length:
        return sequence
    if len(sequence) == 0:
        raise ValueError("Cannot pad an empty sequence without a feature dimension.")

    pad_count = target_length - len(sequence)
    if mode == "zero":
        padding = np.zeros((pad_count, sequence.shape[1]), dtype=np.float32)
    else:
        padding = np.repeat(sequence[-1:], pad_count, axis=0).astype(np.float32)
    return np.concatenate([sequence, padding], axis=0)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == target_length:
        return sequence
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if len(sequence) == 1:
        return pad_sequence(sequence, target_length)

    old_x = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    output = np.empty((target_length, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(new_x, old_x, sequence[:, feature_index])
    return output


def fit_sequence_length(
    sequence: np.ndarray,
    target_length: int,
    method: str = "resample",
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == target_length:
        return sequence
    if method == "pad_trim":
        return pad_sequence(trim_sequence(sequence, target_length), target_length)
    return resample_sequence(sequence, target_length)

