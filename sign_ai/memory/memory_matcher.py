"""DTW-based immediate-memory matcher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sign_ai.config import MAX_MEMORY_SAMPLES_PER_SIGN, MEMORY_SIMILARITY_THRESHOLD
from sign_ai.features.landmark_normalizer import normalize_sequence
from sign_ai.memory.memory_manager import MemoryExample, load_memory_examples


@dataclass(frozen=True)
class MemoryMatch:
    label: str
    similarity: float
    path: str
    source: str


def _downsample(sequence: np.ndarray, max_frames: int = 48) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) <= max_frames:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, max_frames).round().astype(int)
    return sequence[indices]


def dtw_distance(a: np.ndarray, b: np.ndarray, max_frames: int = 48) -> float:
    a = _downsample(a, max_frames=max_frames)
    b = _downsample(b, max_frames=max_frames)
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    prev = np.full(len(b) + 1, np.inf, dtype=np.float32)
    curr = np.full(len(b) + 1, np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, len(a) + 1):
        curr[0] = np.inf
        for j in range(1, len(b) + 1):
            cost = float(np.mean(np.abs(a[i - 1] - b[j - 1])))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[len(b)] / (len(a) + len(b)))


def distance_to_similarity(distance: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + distance * 35.0)))


class DTWMemoryMatcher:
    def __init__(self, examples: list[MemoryExample], threshold: float = MEMORY_SIMILARITY_THRESHOLD):
        self.examples = examples
        self.threshold = threshold

    @classmethod
    def from_memory(cls, max_per_label: int = MAX_MEMORY_SAMPLES_PER_SIGN) -> "DTWMemoryMatcher":
        return cls(load_memory_examples(max_per_label=max_per_label))

    def match(self, sequence: np.ndarray, top_n: int = 3, already_normalized: bool = False) -> list[MemoryMatch]:
        if not self.examples:
            return []
        current = np.asarray(sequence, dtype=np.float32)
        current = current if already_normalized else normalize_sequence(current)
        matches: list[MemoryMatch] = []
        for example in self.examples:
            distance = dtw_distance(current, example.sequence)
            similarity = distance_to_similarity(distance)
            matches.append(
                MemoryMatch(
                    label=example.label,
                    similarity=similarity,
                    path=str(example.path),
                    source=example.source or "memory",
                )
            )
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_n]
