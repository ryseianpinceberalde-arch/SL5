"""Frame sequence buffer with model-length fitting."""

from __future__ import annotations

from collections import deque

import numpy as np

from sign_ai.features.sequence_tools import fit_sequence_length


class SequenceBuffer:
    def __init__(self, max_frames: int):
        self.max_frames = max_frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)

    def append(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32))

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0, 0), dtype=np.float32)
        return np.array(list(self._frames), dtype=np.float32)

    def fitted(self, target_length: int, method: str = "resample") -> np.ndarray:
        return fit_sequence_length(self.as_array(), target_length=target_length, method=method)


class PreRollBuffer(SequenceBuffer):
    def __init__(self, max_frames: int = 8):
        super().__init__(max_frames=max_frames)

