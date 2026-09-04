"""Thread-safe adapter between the existing recognizer and the web dashboard.

The bridge does not define a second recognition system. It loads the optimized
runtime as a module and reuses its camera, MediaPipe, model, memory, decision,
sequence, smoothing, and prediction-worker components.
"""

from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from typing import Iterator

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = PROJECT_ROOT / "11_test_in_real_time_OPTIMIZED.py"
RUNTIME_MODULE_NAME = "_sign_ai_optimized_runtime"
WEB_STREAM_FPS = 15.0
JPEG_QUALITY = 78
HISTORY_LIMIT = 50
CAMERA_FRAME_ERROR = "Camera frame not received."
CAMERA_FRAME_MISS_LIMIT = 3

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_CAMERA_OWNER_LOCK = threading.Lock()


class UnsupportedModeError(ValueError):
    """Raised when the browser requests recognition logic that is not present."""


def _load_runtime_module() -> ModuleType:
    existing = sys.modules.get(RUNTIME_MODULE_NAME)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(RUNTIME_MODULE_NAME, RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load recognition runtime from {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNTIME_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(RUNTIME_MODULE_NAME, None)
        raise
    return module


def _read_active_model_metadata() -> dict:
    registry_path = PROJECT_ROOT / "models" / "model_registry.json"
    if not registry_path.exists():
        return {}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    active_id = registry.get("active_model")
    info = registry.get("models", {}).get(active_id, {}) if active_id else {}
    return {"id": active_id, **info} if isinstance(info, dict) else {}


def _readable_label(label: str) -> str:
    return str(label or "").replace("_", " ").strip().title()


class RecognitionBridge:
    """Own one recognition loop and provide immutable state snapshots."""

    interface_modes = (
        {
            "name": "HYBRID",
            "available": False,
            "reason": "Unavailable: no SignDETR or letter recognizer exists in this project.",
        },
        {
            "name": "LETTERS",
            "available": False,
            "reason": "Unavailable: no SignDETR or letter recognizer exists in this project.",
        },
        {
            "name": "WORDS",
            "available": True,
            "reason": "Uses the existing LSTM model with DTW memory fallback.",
        },
    )
    sequence_modes = ("FIXED", "MOTION")

    def __init__(self, load_models: bool = True):
        self.runtime = _load_runtime_module()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._clear_event = threading.Event()
        self._sequence_mode_changed = threading.Event()
        self._thread: threading.Thread | None = None
        self._owns_camera = False
        self._frame_version = 0
        self._latest_jpeg = b""
        self._history: deque[dict] = deque(maxlen=HISTORY_LIMIT)
        self._interface_mode = "WORDS"
        self._sequence_mode = "FIXED"
        self.model = None
        self.model_actions: list[str] = []
        self.matcher = self.runtime.DTWMemoryMatcher([])
        self.model_sequence_length = int(self.runtime.SEQUENCE_LENGTH)
        self._model_error = ""
        self._memory_error = ""
        self._model_metadata = _read_active_model_metadata()

        if load_models:
            self._initialize_ai_once()

        self._state = self._initial_state()
        self._set_placeholder_frame("Recognition stopped")

    def _initialize_ai_once(self) -> None:
        try:
            self.model, self.model_actions = self.runtime.load_model_if_compatible()
            self.model_sequence_length = self.runtime.model_sequence_length_for(self.model)
            if self.model is None:
                self._model_error = "LSTM model was not found or is incompatible with its labels."
        except Exception as exc:
            self.model = None
            self.model_actions = []
            self._model_error = str(exc)

        try:
            self.matcher = self.runtime.DTWMemoryMatcher.from_memory(
                self.runtime.MAX_MEMORY_SAMPLES_PER_SIGN
            )
        except Exception as exc:
            self.matcher = self.runtime.DTWMemoryMatcher([])
            self._memory_error = str(exc)

    def _initial_state(self) -> dict:
        model_path = self.runtime.current_model_path() if self.model is not None else ""
        model_name = self._model_metadata.get("id") or (Path(model_path).name if model_path else "Unavailable")
        model_type = str(self._model_metadata.get("model_type", "lstm")).upper()
        return {
            "running": False,
            "system_online": False,
            "camera_connected": False,
            "mediapipe_active": False,
            "recognition_state": "IDLE",
            "mode": self._interface_mode,
            "available_modes": [dict(item) for item in self.interface_modes],
            "sequence_mode": self._sequence_mode,
            "available_sequence_modes": list(self.sequence_modes),
            "fps": 0.0,
            "ai_fps": 0.0,
            "prediction_time_ms": 0.0,
            "detected_sign": "IDLE",
            "confidence": 0.0,
            "prediction_source": "none",
            "prediction_reason": "recognition_stopped",
            "top_predictions": [],
            "memory_matches": [],
            "sentence": [],
            "translated_text": "",
            "history": [],
            "sequence_length": self.model_sequence_length,
            "model": {
                "loaded": self.model is not None,
                "status": "Loaded" if self.model is not None else "Not loaded",
                "name": model_name,
                "type": model_type,
                "path": model_path,
                "labels": list(self.model_actions),
                "error": self._model_error,
            },
            "memory": {
                "loaded": bool(self.matcher.examples),
                "status": "Loaded" if self.matcher.examples else "Not loaded",
                "examples": len(self.matcher.examples),
                "error": self._memory_error,
            },
            "signdetr": {
                "loaded": False,
                "status": "Not available",
                "error": "No SignDETR implementation or checkpoint was found during project inspection.",
            },
            "camera": {
                "index": int(self.runtime.CAMERA_INDEX),
                "width": int(self.runtime.CAMERA_WIDTH),
                "height": int(self.runtime.CAMERA_HEIGHT),
                "target_fps": int(self.runtime.CAMERA_FPS),
            },
            "error": "",
            "updated_at": time.time(),
        }

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self._state)
            state["available_modes"] = [dict(item) for item in self._state["available_modes"]]
            state["available_sequence_modes"] = list(self._state["available_sequence_modes"])
            state["top_predictions"] = [dict(item) for item in self._state["top_predictions"]]
            state["memory_matches"] = [dict(item) for item in self._state["memory_matches"]]
            state["sentence"] = list(self._state["sentence"])
            state["history"] = [dict(item) for item in self._history]
            state["model"] = dict(self._state["model"])
            state["model"]["labels"] = list(self._state["model"]["labels"])
            state["memory"] = dict(self._state["memory"])
            state["signdetr"] = dict(self._state["signdetr"])
            state["camera"] = dict(self._state["camera"])
            return state

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._clear_event.clear()
            self._state.update(
                running=True,
                system_online=True,
                recognition_state="STARTING",
                error="",
                updated_at=time.time(),
            )
            self._thread = threading.Thread(
                target=self._recognition_loop,
                name="sign-ai-web-recognition",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 6.0) -> bool:
        with self._lock:
            thread = self._thread
            was_running = bool(thread and thread.is_alive())
            self._stop_event.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            still_running = bool(thread and thread.is_alive())
            if not still_running:
                self._state.update(
                    running=False,
                    system_online=False,
                    camera_connected=False,
                    mediapipe_active=False,
                    recognition_state="IDLE",
                    fps=0.0,
                    ai_fps=0.0,
                    updated_at=time.time(),
                )
                self._set_placeholder_frame("Recognition stopped")
            return was_running and not still_running

    def clear_sentence(self) -> None:
        self._clear_event.set()
        with self._lock:
            self._state["sentence"] = []
            self._state["translated_text"] = ""
            self._state["updated_at"] = time.time()

    def set_interface_mode(self, mode: str) -> str:
        requested = str(mode or "").strip().upper()
        available = {item["name"] for item in self.interface_modes if item["available"]}
        known = {item["name"] for item in self.interface_modes}
        if requested not in known:
            raise ValueError(f"Unknown recognition mode: {requested or '(empty)'}")
        if requested not in available:
            reason = next(item["reason"] for item in self.interface_modes if item["name"] == requested)
            raise UnsupportedModeError(reason)
        with self._lock:
            self._interface_mode = requested
            self._state["mode"] = requested
            self._state["updated_at"] = time.time()
        return requested

    def set_sequence_mode(self, mode: str) -> str:
        requested = str(mode or "").strip().upper()
        if requested not in self.sequence_modes:
            raise ValueError(f"Sequence mode must be one of: {', '.join(self.sequence_modes)}")
        with self._lock:
            self._sequence_mode = requested
            self._state["sequence_mode"] = requested
            self._state["updated_at"] = time.time()
        self._sequence_mode_changed.set()
        return requested

    def mjpeg_stream(self) -> Iterator[bytes]:
        last_version = -1
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: self._frame_version != last_version,
                    timeout=1.0,
                )
                jpeg = self._latest_jpeg
                last_version = self._frame_version
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def _claim_camera(self) -> bool:
        claimed = _CAMERA_OWNER_LOCK.acquire(blocking=False)
        self._owns_camera = claimed
        return claimed

    def _release_camera(self) -> None:
        if self._owns_camera:
            self._owns_camera = False
            _CAMERA_OWNER_LOCK.release()

    def _set_placeholder_frame(self, message: str) -> None:
        height = max(360, int(self.runtime.CAMERA_HEIGHT))
        width = max(640, int(self.runtime.CAMERA_WIDTH))
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (15, 20, 29)
        cv2.putText(
            image,
            "SIGN AI",
            (max(24, width // 2 - 95), max(60, height // 2 - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (224, 231, 239),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(message)[:80],
            (max(24, width // 2 - 180), max(100, height // 2 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (145, 158, 174),
            1,
            cv2.LINE_AA,
        )
        self._store_frame(image)

    def _store_frame(self, image: np.ndarray) -> None:
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not encoded:
            return
        with self._frame_condition:
            self._latest_jpeg = buffer.tobytes()
            self._frame_version += 1
            self._frame_condition.notify_all()

    def _update_camera_frame_health(self, frame_received: bool) -> None:
        """Report sustained frame loss and clear that error after recovery."""
        with self._lock:
            if frame_received:
                self._state["camera_connected"] = True
                if self._state["error"] == CAMERA_FRAME_ERROR:
                    self._state["error"] = ""
            else:
                self._state["camera_connected"] = False
                self._state["error"] = CAMERA_FRAME_ERROR
            self._state["updated_at"] = time.time()

    def _top_predictions(self, scores: np.ndarray) -> list[dict]:
        if scores is None or len(scores) == 0 or not self.model_actions:
            return []
        indices = np.argsort(scores)[::-1][: min(self.runtime.TOP_N, len(self.model_actions))]
        return [
            {
                "label": self.model_actions[int(index)],
                "display_label": _readable_label(self.model_actions[int(index)]),
                "confidence": float(scores[int(index)]),
            }
            for index in indices
        ]

    @staticmethod
    def _memory_matches(matches) -> list[dict]:
        return [
            {
                "label": match.label,
                "display_label": _readable_label(match.label),
                "confidence": float(match.similarity),
                "source": match.source,
            }
            for match in matches
        ]

    def _record_result(self, result, smoother, sentence: list[str]) -> None:
        emitted = None
        smoothed = smoother.add(result.decision.label, result.decision.confidence, result.decision.source)
        if (
            smoothed
            and smoothed.label not in {"UNKNOWN", "IDLE", "TRANSITION"}
            and smoother.should_emit(smoothed.label)
        ):
            sentence.append(smoothed.label)
            sentence[:] = sentence[-self.runtime.MAX_SENTENCE_WORDS :]
            emitted = {
                "time": time.strftime("%H:%M:%S"),
                "sign": smoothed.label,
                "display_sign": _readable_label(smoothed.label),
                "confidence": float(smoothed.confidence),
                "source": smoothed.source,
            }

        with self._lock:
            if emitted:
                self._history.appendleft(emitted)
            self._state.update(
                detected_sign=result.decision.label,
                confidence=float(result.decision.confidence),
                prediction_source=result.decision.source,
                prediction_reason=result.decision.reason,
                prediction_time_ms=float(result.prediction_time_ms),
                top_predictions=self._top_predictions(result.model_res),
                memory_matches=self._memory_matches(result.memory_matches) if result.used_memory else [],
                sentence=list(sentence),
                translated_text=" ".join(_readable_label(item) for item in sentence),
                updated_at=time.time(),
            )

    def _recognition_loop(self) -> None:
        raw_capture = None
        capture = None
        worker = None
        camera_claimed = False
        sentence: list[str] = []

        try:
            camera_claimed = self._claim_camera()
            if not camera_claimed:
                raise RuntimeError("Camera is already owned by another web recognition loop.")

            raw_capture = self.runtime.open_camera(self.runtime.CAMERA_INDEX)
            actual_fps, _, actual_width, actual_height = self.runtime.configure_camera_capture(
                raw_capture,
                self.runtime.CAMERA_FPS,
                self.runtime.CAMERA_WIDTH,
                self.runtime.CAMERA_HEIGHT,
            )
            capture = self.runtime.LatestFrameCapture(raw_capture)
            worker = self.runtime.LatestPredictionWorker(
                self.model,
                self.model_actions,
                self.matcher,
                self.model_sequence_length,
            )

            motion_detector = self.runtime.MotionDetector()
            pre_roll = self.runtime.PreRollBuffer(max_frames=8)
            sign_buffer = self.runtime.SequenceBuffer(max_frames=motion_detector.max_sign_frames)
            fixed_buffer = self.runtime.SequenceBuffer(max_frames=self.model_sequence_length)
            smoother = self.runtime.PredictionSmoother()

            last_prediction_submit = 0.0
            last_hand_seen = 0.0
            latest_ai_result_time = 0.0
            frame_count = 0
            ai_result_count = 0
            measured_fps = 0.0
            measured_ai_fps = 0.0
            fps_window_started = time.perf_counter()
            last_stream_frame = 0.0
            consecutive_frame_misses = 0

            with self._lock:
                self._state["camera_connected"] = True
                self._state["camera"].update(
                    reported_fps=float(actual_fps),
                    reported_width=int(actual_width),
                    reported_height=int(actual_height),
                )
                self._state["recognition_state"] = "RUNNING"
                self._state["error"] = ""
                self._state["updated_at"] = time.time()

            with self.runtime.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                with self._lock:
                    self._state["mediapipe_active"] = True

                while not self._stop_event.is_set() and capture.isOpened():
                    received, frame = capture.read(timeout=1.0)
                    if not received or frame is None:
                        if self._stop_event.is_set():
                            break
                        consecutive_frame_misses += 1
                        if consecutive_frame_misses >= CAMERA_FRAME_MISS_LIMIT:
                            self._update_camera_frame_health(frame_received=False)
                        continue

                    consecutive_frame_misses = 0
                    self._update_camera_frame_health(frame_received=True)

                    if self._clear_event.is_set():
                        self._clear_event.clear()
                        sentence.clear()
                        smoother.clear()

                    if self._sequence_mode_changed.is_set():
                        self._sequence_mode_changed.clear()
                        fixed_buffer.clear()
                        sign_buffer.clear()
                        pre_roll.clear()
                        smoother.clear()
                        motion_detector.reset()

                    process_frame = self.runtime.maybe_resize_for_processing(
                        frame,
                        self.runtime.PROCESS_WIDTH,
                    )
                    _, results = self.runtime.mediapipe_detection(process_frame, holistic)
                    image = frame.copy()
                    self.runtime.draw_styled_landmarks(image, results)

                    raw_keypoints = self.runtime.extract_keypoints(results)
                    normalized_keypoints = self.runtime.normalize_keypoints(raw_keypoints)
                    hand_present = self.runtime.has_hand(results) or self.runtime.has_hand_keypoints(raw_keypoints)
                    now = time.perf_counter()

                    if hand_present:
                        last_hand_seen = now
                    else:
                        fixed_buffer.clear()
                        pre_roll.clear()
                        if now - last_hand_seen >= self.runtime.NO_HAND_RESET_SECONDS:
                            smoother.clear()
                            with self._lock:
                                self._state.update(
                                    detected_sign="IDLE",
                                    confidence=0.0,
                                    prediction_source="motion",
                                    prediction_reason="no_hand",
                                    top_predictions=[],
                                    memory_matches=[],
                                    updated_at=time.time(),
                                )

                    sequence_to_predict = None
                    prediction_motion_state = self.runtime.RECORDING
                    with self._lock:
                        sequence_mode = self._sequence_mode

                    if sequence_mode == "FIXED":
                        motion_state = "FIXED_SEQUENCE"
                        if hand_present:
                            fixed_buffer.append(normalized_keypoints)
                        if hand_present and len(fixed_buffer) == self.model_sequence_length:
                            sequence_to_predict = fixed_buffer.as_array()
                    else:
                        if hand_present:
                            pre_roll.append(normalized_keypoints)
                        motion = motion_detector.update(normalized_keypoints, hand_present=hand_present)
                        motion_state = motion.state
                        if motion.event == "start":
                            sign_buffer.clear()
                            for frame_item in pre_roll.as_array():
                                sign_buffer.append(frame_item)
                        if motion.event in {"recording", "transition", "complete"} and hand_present:
                            sign_buffer.append(normalized_keypoints)
                        if motion.event == "complete" and len(sign_buffer) > 0:
                            sequence_to_predict = sign_buffer.as_array()
                            prediction_motion_state = self.runtime.SIGN_COMPLETE
                            sign_buffer.clear()
                            pre_roll.clear()
                            motion_detector.reset()

                    if (
                        sequence_to_predict is not None
                        and len(sequence_to_predict) > 0
                        and now - last_prediction_submit >= self.runtime.PREDICTION_INTERVAL_SECONDS
                    ):
                        submitted = worker.submit_latest(
                            self.runtime.PredictionRequest(
                                sequence=np.asarray(sequence_to_predict, dtype=np.float32).copy(),
                                motion_state=prediction_motion_state,
                                requested_at=now,
                            )
                        )
                        if submitted:
                            last_prediction_submit = now

                    result = worker.get_latest_result()
                    if result is not None:
                        ai_result_count += 1
                        latest_ai_result_time = result.completed_at
                        self._record_result(result, smoother, sentence)

                    if sequence_mode == "MOTION":
                        if motion_state == self.runtime.IDLE and now - latest_ai_result_time > self.runtime.PREDICTION_INTERVAL_SECONDS:
                            with self._lock:
                                self._state.update(
                                    detected_sign="IDLE",
                                    confidence=0.0,
                                    prediction_source="motion",
                                    prediction_reason="idle",
                                )
                        elif motion_state == self.runtime.TRANSITION:
                            with self._lock:
                                self._state.update(
                                    detected_sign="TRANSITION",
                                    confidence=0.0,
                                    prediction_source="motion",
                                    prediction_reason="movement_not_complete",
                                )

                    frame_count += 1
                    elapsed = time.perf_counter() - fps_window_started
                    if elapsed >= 1.0:
                        measured_fps = self.runtime.smooth_fps(measured_fps, frame_count / elapsed)
                        measured_ai_fps = self.runtime.smooth_fps(measured_ai_fps, ai_result_count / elapsed)
                        frame_count = 0
                        ai_result_count = 0
                        fps_window_started = time.perf_counter()
                        with self._lock:
                            self._state.update(
                                fps=float(measured_fps),
                                ai_fps=float(measured_ai_fps),
                                recognition_state=motion_state,
                                updated_at=time.time(),
                            )

                    if now - last_stream_frame >= 1.0 / WEB_STREAM_FPS:
                        self._store_frame(image)
                        last_stream_frame = now

        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            with self._lock:
                self._state.update(
                    error=message,
                    recognition_state="ERROR",
                    updated_at=time.time(),
                )
            self._set_placeholder_frame(message)
        finally:
            if worker is not None:
                worker.stop()
            if capture is not None:
                capture.release()
            elif raw_capture is not None:
                raw_capture.release()
            if camera_claimed:
                self._release_camera()
            with self._lock:
                self._state.update(
                    running=False,
                    system_online=False,
                    camera_connected=False,
                    mediapipe_active=False,
                    fps=0.0,
                    ai_fps=0.0,
                    updated_at=time.time(),
                )
