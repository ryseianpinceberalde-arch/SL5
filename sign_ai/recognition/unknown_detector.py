"""UNKNOWN and IDLE decision helpers."""

from __future__ import annotations

from sign_ai.config import UNKNOWN_THRESHOLD
from sign_ai.recognition.motion_detector import IDLE


def is_unknown(confidence: float, threshold: float = UNKNOWN_THRESHOLD) -> bool:
    return float(confidence) < threshold


def label_for_motion_state(motion_state: str) -> str | None:
    if motion_state == IDLE:
        return "IDLE"
    return None

