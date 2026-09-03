"""Optional landmark-level augmentation for training data only."""

from __future__ import annotations

import numpy as np

from sign_ai.features.sequence_tools import fit_sequence_length


def augment_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.008,
    scale_range: tuple[float, float] = (0.97, 1.03),
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    scale = rng.uniform(scale_range[0], scale_range[1])
    noise = rng.normal(0.0, noise_std, size=sequence.shape).astype(np.float32)
    augmented = sequence * scale + noise

    speed = rng.choice([0.85, 1.0, 1.15])
    temp_length = max(2, int(round(len(sequence) * float(speed))))
    augmented = fit_sequence_length(augmented, temp_length)
    return fit_sequence_length(augmented, len(sequence)).astype(np.float32)


def augment_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    factor: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if factor <= 1:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    augmented_X = [np.asarray(X_train, dtype=np.float32)]
    augmented_y = [np.asarray(y_train, dtype=np.float32)]
    for _ in range(factor - 1):
        augmented_X.append(np.array([augment_sequence(sequence, rng) for sequence in X_train], dtype=np.float32))
        augmented_y.append(y_train.copy())
    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)

