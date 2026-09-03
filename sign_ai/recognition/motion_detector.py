"""Motion-based sign start/end detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sign_ai.config import (
    MAX_SIGN_FRAMES,
    MIN_SIGN_FRAMES,
    MOTION_START_THRESHOLD,
    MOTION_STOP_THRESHOLD,
    START_CONFIRM_FRAMES,
    STOP_CONFIRM_FRAMES,
)
from sign_ai.features.motion_features import movement_magnitude


IDLE = "IDLE"
RECORDING = "RECORDING"
SIGN_COMPLETE = "SIGN_COMPLETE"
TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class MotionEvent:
    state: str
    event: str
    movement: float
    frames_recorded: int


class MotionDetector:
    def __init__(
        self,
        start_threshold: float = MOTION_START_THRESHOLD,
        stop_threshold: float = MOTION_STOP_THRESHOLD,
        start_confirm_frames: int = START_CONFIRM_FRAMES,
        stop_confirm_frames: int = STOP_CONFIRM_FRAMES,
        min_sign_frames: int = MIN_SIGN_FRAMES,
        max_sign_frames: int = MAX_SIGN_FRAMES,
    ):
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.start_confirm_frames = start_confirm_frames
        self.stop_confirm_frames = stop_confirm_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.previous: np.ndarray | None = None
        self.state = IDLE
        self.start_hits = 0
        self.stop_hits = 0
        self.frames_recorded = 0

    def reset(self) -> None:
        self.state = IDLE
        self.start_hits = 0
        self.stop_hits = 0
        self.frames_recorded = 0

    def update(self, keypoints: np.ndarray, hand_present: bool = True) -> MotionEvent:
        movement = movement_magnitude(self.previous, keypoints)
        self.previous = np.asarray(keypoints, dtype=np.float32)

        if not hand_present:
            if self.state == RECORDING and self.frames_recorded >= self.min_sign_frames:
                self.state = SIGN_COMPLETE
                return MotionEvent(SIGN_COMPLETE, "complete", movement, self.frames_recorded)
            self.reset()
            return MotionEvent(IDLE, "idle", movement, self.frames_recorded)

        if self.state == IDLE:
            if movement >= self.start_threshold:
                self.start_hits += 1
            else:
                self.start_hits = 0
            if self.start_hits >= self.start_confirm_frames:
                self.state = RECORDING
                self.frames_recorded = 0
                self.stop_hits = 0
                return MotionEvent(RECORDING, "start", movement, self.frames_recorded)
            return MotionEvent(IDLE, "idle", movement, self.frames_recorded)

        if self.state == RECORDING:
            self.frames_recorded += 1
            if movement <= self.stop_threshold:
                self.stop_hits += 1
            else:
                self.stop_hits = 0

            if self.frames_recorded >= self.max_sign_frames:
                self.state = SIGN_COMPLETE
                return MotionEvent(SIGN_COMPLETE, "complete", movement, self.frames_recorded)

            if self.frames_recorded >= self.min_sign_frames and self.stop_hits >= self.stop_confirm_frames:
                self.state = SIGN_COMPLETE
                return MotionEvent(SIGN_COMPLETE, "complete", movement, self.frames_recorded)

            if self.frames_recorded < self.min_sign_frames and self.stop_hits:
                return MotionEvent(TRANSITION, "transition", movement, self.frames_recorded)

            return MotionEvent(RECORDING, "recording", movement, self.frames_recorded)

        return MotionEvent(self.state, "waiting", movement, self.frames_recorded)

