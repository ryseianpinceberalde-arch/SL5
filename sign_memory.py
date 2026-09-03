"""Backward-compatible wrappers for the new sign_ai memory system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from sign_ai.memory.memory_manager import (
    MemoryExample,
    load_memory,
    load_memory_examples,
    save_memory_example_from_sample,
)
from sign_ai.memory.memory_matcher import DTWMemoryMatcher


@dataclass(frozen=True)
class MemoryReference:
    action: str
    path: Path
    sequence: np.ndarray


@dataclass(frozen=True)
class MemoryPrediction:
    action: str
    confidence: float
    path: str


def record_sample(action: str, sample_dir: Path, source: str = "camera") -> dict:
    return save_memory_example_from_sample(action, sample_dir, source=source)


def rebuild_memory_from_dataset(include_legacy_batches: bool = False) -> dict:
    from sign_language_common import DATA_PATH
    from sign_language_dataset import find_sample_dirs

    for action_dir in sorted([item for item in DATA_PATH.iterdir() if item.is_dir()], key=lambda item: item.name):
        for sample_dir in find_sample_dirs(action_dir, include_legacy_batches=include_legacy_batches):
            save_memory_example_from_sample(action_dir.name, sample_dir, source="dataset")
    return load_memory()


def load_memory_references(
    actions: Iterable[str] | None = None,
    max_samples_per_action: int = 12,
) -> list[MemoryReference]:
    allowed = set(actions) if actions is not None else None
    references = []
    for example in load_memory_examples(max_per_label=max_samples_per_action):
        if allowed is not None and example.label not in allowed:
            continue
        references.append(MemoryReference(action=example.label, path=example.path, sequence=example.sequence))
    return references


def top_memory_matches(
    sequence: np.ndarray,
    references: list[MemoryReference],
    top_n: int = 3,
) -> list[MemoryPrediction]:
    examples = [
        MemoryExample(
            label=reference.action,
            path=reference.path,
            sequence=reference.sequence,
            source="memory",
            metadata={},
        )
        for reference in references
    ]
    matcher = DTWMemoryMatcher(examples)
    return [
        MemoryPrediction(action=match.label, confidence=match.similarity, path=match.path)
        for match in matcher.match(sequence, top_n=top_n, already_normalized=False)
    ]
