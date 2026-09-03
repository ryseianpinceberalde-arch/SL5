"""Rolling prediction smoothing and sentence emission control."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

from sign_ai.config import PREDICTION_COOLDOWN, PREDICTION_WINDOW, STABLE_FRAMES


@dataclass(frozen=True)
class SmoothedPrediction:
    label: str
    confidence: float
    source: str
    stable_count: int


class PredictionSmoother:
    def __init__(
        self,
        window: int = PREDICTION_WINDOW,
        stable_frames: int = STABLE_FRAMES,
        cooldown_seconds: float = PREDICTION_COOLDOWN,
    ):
        self.history: deque[tuple[str, float, str]] = deque(maxlen=window)
        self.stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.last_emitted = ""
        self.last_emit_time = 0.0

    def clear(self) -> None:
        self.history.clear()
        self.last_emitted = ""
        self.last_emit_time = 0.0

    def add(self, label: str, confidence: float, source: str) -> SmoothedPrediction | None:
        if not label:
            return None
        self.history.append((label, float(confidence), source))
        label_counts = Counter(item[0] for item in self.history)
        best_label, count = label_counts.most_common(1)[0]
        if count < self.stable_frames:
            return None

        matching = [item for item in self.history if item[0] == best_label]
        avg_confidence = sum(item[1] for item in matching) / len(matching)
        source = Counter(item[2] for item in matching).most_common(1)[0][0]
        return SmoothedPrediction(best_label, avg_confidence, source, count)

    def should_emit(self, label: str) -> bool:
        now = time.time()
        if label == self.last_emitted and now - self.last_emit_time < self.cooldown_seconds:
            return False
        self.last_emitted = label
        self.last_emit_time = now
        return True

