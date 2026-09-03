"""Combine model, memory, motion state, and thresholds into one decision."""

from __future__ import annotations

from dataclasses import dataclass

from sign_ai.config import CONFIDENCE_THRESHOLD, MEMORY_SIMILARITY_THRESHOLD, UNKNOWN_THRESHOLD
from sign_ai.recognition.motion_detector import IDLE, TRANSITION


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float
    source: str


@dataclass(frozen=True)
class Decision:
    label: str
    confidence: float
    source: str
    reason: str


def choose_decision(
    motion_state: str,
    model_candidate: Candidate | None = None,
    memory_candidate: Candidate | None = None,
    model_threshold: float = CONFIDENCE_THRESHOLD,
    memory_threshold: float = MEMORY_SIMILARITY_THRESHOLD,
    unknown_threshold: float = UNKNOWN_THRESHOLD,
) -> Decision:
    if motion_state == IDLE:
        return Decision("IDLE", 1.0, "motion", "no_sign_motion")
    if motion_state == TRANSITION:
        return Decision("TRANSITION", 1.0, "motion", "movement_not_complete")

    if memory_candidate and memory_candidate.confidence >= memory_threshold:
        return Decision(
            memory_candidate.label,
            memory_candidate.confidence,
            memory_candidate.source,
            "strong_memory_match",
        )

    if model_candidate and model_candidate.confidence >= model_threshold:
        return Decision(
            model_candidate.label,
            model_candidate.confidence,
            model_candidate.source,
            "strong_model_match",
        )

    best = memory_candidate or model_candidate
    if best and best.confidence >= unknown_threshold:
        return Decision(best.label, best.confidence, best.source, "weak_but_known")
    if best:
        return Decision("UNKNOWN", best.confidence, best.source, "below_unknown_threshold")
    return Decision("UNKNOWN", 0.0, "none", "no_candidate")
