#!/usr/bin/env python
# coding: utf-8
"""Optimized real-time sign recognition with non-blocking AI prediction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import os
import queue
import sys
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import numpy as np

from sign_ai.config import (
    CAMERA_FPS,
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    KEYPOINT_LENGTH,
    LANDMARK_ONLY,
    LANDMARK_PLUS_VELOCITY,
    MAX_MEMORY_SAMPLES_PER_SIGN,
    MAX_SENTENCE_WORDS,
    MEMORY_SIMILARITY_THRESHOLD,
    SEQUENCE_LENGTH,
    TOP_N,
)
from sign_ai.features.landmark_normalizer import has_hand_keypoints, normalize_keypoints
from sign_ai.features.motion_features import apply_feature_mode
from sign_ai.features.sequence_tools import fit_sequence_length
from sign_ai.memory.memory_manager import save_correction, teach_sequence
from sign_ai.memory.memory_matcher import DTWMemoryMatcher, MemoryMatch
from sign_ai.recognition.decision_engine import Candidate, Decision, choose_decision
from sign_ai.recognition.motion_detector import IDLE, MotionDetector, RECORDING, SIGN_COMPLETE, TRANSITION
from sign_ai.recognition.prediction_smoother import PredictionSmoother
from sign_ai.recognition.sequence_buffer import PreRollBuffer, SequenceBuffer
from sign_language_common import (
    LEGACY_MODEL_PATH,
    MODEL_PATH,
    draw_styled_landmarks,
    extract_keypoints,
    load_actions,
    load_model_actions,
    load_model_safely,
    mediapipe_detection,
    mp_holistic,
    open_camera,
    sanitize_action_name,
)


WINDOW_NAME = "Real Time Sign Recognition - Optimized"
PREDICTION_INTERVAL_SECONDS = float(os.getenv("SL_PREDICTION_INTERVAL_SECONDS", "0.15"))
NO_HAND_RESET_SECONDS = float(os.getenv("SL_NO_HAND_RESET_SECONDS", "0.75"))
PROCESS_WIDTH = int(os.getenv("SL_PROCESS_WIDTH", "0"))


@dataclass(frozen=True)
class PredictionRequest:
    sequence: np.ndarray
    motion_state: str
    requested_at: float


@dataclass(frozen=True)
class PredictionResult:
    model_candidate: Candidate | None
    model_res: np.ndarray
    memory_candidate: Candidate | None
    memory_matches: list[MemoryMatch]
    decision: Decision
    sequence: np.ndarray
    prediction_time_ms: float
    used_memory: bool
    requested_at: float
    completed_at: float


class LatestPredictionWorker:
    """Runs expensive model and DTW work away from the camera/display loop."""

    def __init__(
        self,
        model,
        model_actions: list[str],
        matcher: DTWMemoryMatcher,
        model_sequence_length: int,
    ):
        self.model = model
        self.model_actions = model_actions
        self.matcher = matcher
        self.model_sequence_length = model_sequence_length
        self.requests: queue.Queue[PredictionRequest | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[PredictionResult] = queue.Queue()
        self._matcher_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sign-ai-prediction-worker", daemon=True)
        self._thread.start()

    def submit_latest(self, request: PredictionRequest) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self.requests.put_nowait(request)
            return True
        except queue.Full:
            try:
                self.requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self.requests.put_nowait(request)
                return True
            except queue.Full:
                return False

    def update_matcher(self, matcher: DTWMemoryMatcher) -> None:
        with self._matcher_lock:
            self.matcher = matcher

    def get_latest_result(self) -> PredictionResult | None:
        latest = None
        while True:
            try:
                latest = self.results.get_nowait()
            except queue.Empty:
                return latest

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            try:
                self.requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break

            started = time.perf_counter()
            model_candidate = None
            memory_candidate = None
            memory_matches: list[MemoryMatch] = []
            used_memory = False
            model_res = np.zeros(len(self.model_actions), dtype=np.float32)

            try:
                model_candidate, model_res = model_candidate_for_sequence(
                    self.model,
                    self.model_actions,
                    request.sequence,
                    self.model_sequence_length,
                )
            except Exception as exc:
                print(f"Model prediction failed: {exc}")

            if model_candidate is None or model_candidate.confidence < CONFIDENCE_THRESHOLD:
                try:
                    with self._matcher_lock:
                        matcher = self.matcher
                    memory_candidate, memory_matches = memory_candidate_for_sequence(matcher, request.sequence)
                    used_memory = True
                except Exception as exc:
                    print(f"Memory matching failed: {exc}")

            decision = choose_decision(
                motion_state=request.motion_state,
                model_candidate=model_candidate,
                memory_candidate=memory_candidate,
                model_threshold=CONFIDENCE_THRESHOLD,
                memory_threshold=MEMORY_SIMILARITY_THRESHOLD,
            )
            completed = time.perf_counter()

            self.results.put(
                PredictionResult(
                    model_candidate=model_candidate,
                    model_res=model_res,
                    memory_candidate=memory_candidate,
                    memory_matches=memory_matches,
                    decision=decision,
                    sequence=request.sequence,
                    prediction_time_ms=(completed - started) * 1000.0,
                    used_memory=used_memory,
                    requested_at=request.requested_at,
                    completed_at=completed,
                )
            )


@contextmanager
def suppress_library_stderr():
    """Hide noisy TensorFlow/MediaPipe native warnings during live detection."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def parse_args():
    parser = argparse.ArgumentParser(description="Run optimized real-time sign recognition.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument(
        "--fixed-sequence",
        action="store_true",
        default=True,
        help="Classify throttled rolling windows. This is the default.",
    )
    parser.add_argument(
        "--motion-sequence",
        action="store_false",
        dest="fixed_sequence",
        help="Only classify after motion detection decides a sign is complete.",
    )
    return parser.parse_args()


def has_hand(results) -> bool:
    return bool(results.left_hand_landmarks or results.right_hand_landmarks)


def readable_label(label: str) -> str:
    return str(label).replace("_", " ").title()


def put_panel_text(
    image: np.ndarray,
    lines: list[str],
    start_y: int = 285,
    x: int = 12,
    line_height: int = 28,
) -> None:
    if not lines:
        return

    width = min(image.shape[1] - 20, 590)
    height = 18 + line_height * len(lines)
    y1 = max(0, start_y - 24)
    y2 = min(image.shape[0], y1 + height)

    cv2.rectangle(image, (x - 6, y1), (x + width, y2), (35, 35, 35), -1)
    cv2.rectangle(image, (x - 6, y1), (x + width, y2), (255, 255, 255), 1)

    for index, line in enumerate(lines):
        y = start_y + index * line_height
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)


def draw_top3_prediction_box(
    image: np.ndarray,
    res: np.ndarray,
    actions: list[str],
    top_n: int = TOP_N,
) -> np.ndarray:
    output = image.copy()
    if res is None or len(res) == 0 or len(actions) == 0:
        return output

    top_indices = np.argsort(res)[::-1][: min(top_n, len(actions))]
    _, width = output.shape[:2]
    box_w = min(360, max(300, width - 40))
    box_h = 170
    x1 = max(10, width - box_w - 10)
    y1 = 65
    x2 = x1 + box_w
    y2 = y1 + box_h

    cv2.rectangle(output, (x1, y1), (x2, y2), (35, 35, 35), -1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(output, "TOP MODEL SIGNS", (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    row_start_y = y1 + 48
    row_h = 36
    bar_x1 = x1 + 12
    bar_x2 = x2 - 12
    bar_w = bar_x2 - bar_x1
    for row, idx in enumerate(top_indices):
        probability = float(res[idx])
        label = readable_label(actions[idx])
        y = row_start_y + row * row_h
        cv2.rectangle(output, (bar_x1, y), (bar_x2, y + 28), (60, 60, 60), -1)
        fill_w = int(max(0, min(1, probability)) * bar_w)
        cv2.rectangle(output, (bar_x1, y), (bar_x1 + fill_w, y + 28), (70, 150, 70), -1)
        cv2.putText(output, f"{row + 1}. {label}", (bar_x1 + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(output, f"{probability * 100:.2f}%", (bar_x2 - 95, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def current_model_path() -> str:
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    if LEGACY_MODEL_PATH.exists():
        return str(LEGACY_MODEL_PATH)
    return "not found"


def load_model_if_compatible():
    try:
        model = load_model_safely()
    except FileNotFoundError:
        return None, []
    except Exception as exc:
        print(f"Model could not be loaded: {exc}")
        return None, []

    try:
        model_actions = load_model_actions(require_existing=False)
    except Exception as exc:
        print(f"Model labels could not be loaded: {exc}")
        return None, []

    if model.output_shape[-1] != len(model_actions):
        print("Model output count does not match model_actions.json. Memory matching remains available.")
        return None, []
    return model, model_actions


def model_sequence_length_for(model) -> int:
    if model is None:
        return SEQUENCE_LENGTH
    try:
        length = model.input_shape[1]
        if length is not None:
            return max(1, int(length))
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    return SEQUENCE_LENGTH


def feature_mode_for_model(model) -> str:
    if model is None:
        return LANDMARK_ONLY
    feature_length = int(model.input_shape[-1])
    if feature_length == KEYPOINT_LENGTH * 2:
        return LANDMARK_PLUS_VELOCITY
    return LANDMARK_ONLY


def model_candidate_for_sequence(
    model,
    actions: list[str],
    sequence: np.ndarray,
    model_sequence_length: int,
) -> tuple[Candidate | None, np.ndarray]:
    if model is None or not actions:
        return None, np.zeros(0, dtype=np.float32)
    fitted = fit_sequence_length(sequence, model_sequence_length)
    features = apply_feature_mode(fitted, feature_mode_for_model(model))
    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    index = int(np.argmax(prediction))
    return Candidate(actions[index], float(prediction[index]), "model"), prediction


def memory_candidate_for_sequence(matcher: DTWMemoryMatcher, sequence: np.ndarray) -> tuple[Candidate | None, list[MemoryMatch]]:
    matches = matcher.match(sequence, top_n=TOP_N, already_normalized=True)
    if not matches:
        return None, []
    best = matches[0]
    return Candidate(best.label, float(best.similarity), "memory"), matches


def ask_positive_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def teach_new_sign(cap, holistic, examples: int = 3, frames: int = SEQUENCE_LENGTH) -> DTWMemoryMatcher:
    label = sanitize_action_name(input("Teach sign name: "))
    if not label:
        print("No sign name entered.")
        return DTWMemoryMatcher.from_memory(MAX_MEMORY_SAMPLES_PER_SIGN)
    examples = ask_positive_int("How many memory examples", examples)
    frames = ask_positive_int("Frames per example", frames)

    for example_index in range(examples):
        print(f"Teaching {label}: example {example_index + 1}/{examples}")
        for number in (3, 2, 1):
            start = time.time()
            while time.time() - start < 0.8:
                ret, frame = cap.read()
                if not ret:
                    continue
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                cv2.putText(image, f"Prepare: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(image, str(number), (image.shape[1] // 2 - 30, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5)
                cv2.imshow(WINDOW_NAME, image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return DTWMemoryMatcher.from_memory(MAX_MEMORY_SAMPLES_PER_SIGN)

        sequence = []
        for frame_num in range(frames):
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            if has_hand(results) or has_hand_keypoints(keypoints):
                sequence.append(keypoints)
            cv2.putText(image, f"Teaching: {label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(image, f"Frame {frame_num + 1}/{frames}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if sequence:
            item, sample_dir = teach_sequence(label, np.array(sequence, dtype=np.float32), source="realtime_teach")
            print(f"Saved training sample: {sample_dir}")
            print(f"Taught AI memory: memory/{item['path']}")

    return DTWMemoryMatcher.from_memory(MAX_MEMORY_SAMPLES_PER_SIGN)


def correct_prediction(decision: Decision, sequence: np.ndarray | None) -> bool:
    correct_label = sanitize_action_name(input("Correct label, or ENTER to clear sentence: "))
    if not correct_label:
        return False
    if sequence is None or len(sequence) == 0:
        print("No completed sequence is available for correction.")
        return True
    save_correction(decision.label, correct_label, decision.confidence, sequence, normalize=False)
    print(f"Saved correction: predicted={decision.label} actual={correct_label}")
    return True


def maybe_resize_for_processing(frame: np.ndarray) -> np.ndarray:
    if PROCESS_WIDTH <= 0 or frame.shape[1] <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / float(frame.shape[1])
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (PROCESS_WIDTH, height), interpolation=cv2.INTER_AREA)


def draw_performance_panel(
    image: np.ndarray,
    camera_fps: float,
    ai_fps: float,
    prediction_time_ms: float,
    mode: str,
    used_memory: bool,
) -> None:
    put_panel_text(
        image,
        [
            f"Camera FPS: {camera_fps:.1f}",
            f"AI Prediction FPS: {ai_fps:.1f}",
            f"Prediction Time: {prediction_time_ms:.0f} ms",
            f"Mode: {mode}",
            f"DTW: {'used' if used_memory else 'skipped'}",
        ],
        start_y=650,
        x=12,
        line_height=24,
    )


def main(camera_index: int = CAMERA_INDEX) -> None:
    args = parse_args()
    camera_index = args.camera if camera_index == CAMERA_INDEX else camera_index
    model, model_actions = load_model_if_compatible()
    model_sequence_length = model_sequence_length_for(model)
    try:
        matcher = DTWMemoryMatcher.from_memory(MAX_MEMORY_SAMPLES_PER_SIGN)
    except Exception as exc:
        print(f"Memory could not be loaded: {exc}")
        matcher = DTWMemoryMatcher([])

    actions = load_actions(require_existing=False)
    cap = open_camera(camera_index)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    worker = LatestPredictionWorker(model, model_actions, matcher, model_sequence_length)
    motion_detector = MotionDetector()
    pre_roll = PreRollBuffer(max_frames=8)
    sign_buffer = SequenceBuffer(max_frames=motion_detector.max_sign_frames)
    fixed_buffer = SequenceBuffer(max_frames=model_sequence_length)
    smoother = PredictionSmoother()
    sentence: list[str] = []

    latest_model_res = np.zeros(len(model_actions), dtype=np.float32)
    latest_model = Candidate("Waiting", 0.0, "model")
    latest_memory = Candidate("Waiting", 0.0, "memory")
    latest_memory_matches: list[MemoryMatch] = []
    latest_decision = Decision("IDLE", 1.0, "motion", "startup")
    latest_completed_sequence: np.ndarray | None = None
    latest_prediction_time_ms = 0.0
    latest_used_memory = False
    latest_ai_result_time = 0.0
    motion_state = IDLE
    fixed_sequence_mode = args.fixed_sequence
    last_prediction_submit = 0.0
    last_hand_seen = 0.0

    camera_frame_count = 0
    camera_fps = 0.0
    ai_result_count = 0
    ai_fps = 0.0
    fps_window_started = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    print("Using model:", current_model_path() if model is not None else "not found or not compatible")
    print("Model actions:", model_actions)
    print("Collected actions:", actions)
    print("Memory examples loaded:", len(matcher.examples))
    print("Model sequence length:", model_sequence_length)
    print(f"Prediction interval: {PREDICTION_INTERVAL_SECONDS:.2f}s (~{1.0 / PREDICTION_INTERVAL_SECONDS:.1f} predictions/sec max)")
    print("Recognition mode:", "fixed rolling throttled windows" if fixed_sequence_mode else "motion-triggered signs")
    print("Controls: T teach | C correct/clear | R reset sentence | F fixed/motion mode | Q/ESC quit")

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                frame_started = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame not received.")
                    continue

                process_frame = maybe_resize_for_processing(frame)
                image, results = mediapipe_detection(process_frame, holistic)
                draw_styled_landmarks(image, results)

                raw_keypoints = extract_keypoints(results)
                normalized_keypoints = normalize_keypoints(raw_keypoints)
                hand_present = has_hand(results) or has_hand_keypoints(raw_keypoints)
                now = time.perf_counter()

                if hand_present:
                    last_hand_seen = now
                else:
                    fixed_buffer.clear()
                    pre_roll.clear()
                    if now - last_hand_seen >= NO_HAND_RESET_SECONDS:
                        latest_model_res = np.zeros(len(model_actions), dtype=np.float32)
                        latest_model = Candidate("Waiting", 0.0, "model")
                        latest_memory = Candidate("Waiting", 0.0, "memory")
                        latest_memory_matches = []
                        latest_decision = Decision("IDLE", 1.0, "motion", "no_hand")
                        latest_used_memory = False
                        smoother.clear()

                sequence_to_predict: np.ndarray | None = None
                prediction_motion_state = RECORDING
                if fixed_sequence_mode:
                    motion_state = "FIXED_SEQUENCE"
                    if hand_present:
                        fixed_buffer.append(normalized_keypoints)
                    if hand_present and len(fixed_buffer) == model_sequence_length:
                        sequence_to_predict = fixed_buffer.as_array()
                        prediction_motion_state = RECORDING
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
                        prediction_motion_state = SIGN_COMPLETE
                        sign_buffer.clear()
                        pre_roll.clear()
                        motion_detector.reset()

                if (
                    sequence_to_predict is not None
                    and len(sequence_to_predict) > 0
                    and now - last_prediction_submit >= PREDICTION_INTERVAL_SECONDS
                ):
                    latest_completed_sequence = sequence_to_predict
                    submitted = worker.submit_latest(
                        PredictionRequest(
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
                    latest_prediction_time_ms = result.prediction_time_ms
                    latest_used_memory = result.used_memory
                    latest_model_res = result.model_res
                    latest_completed_sequence = result.sequence
                    if result.model_candidate:
                        latest_model = result.model_candidate
                    if result.used_memory:
                        latest_memory_matches = result.memory_matches
                        latest_memory = result.memory_candidate or Candidate("No Match", 0.0, "memory")
                    latest_decision = result.decision
                    smoothed = smoother.add(latest_decision.label, latest_decision.confidence, latest_decision.source)
                    if smoothed and smoothed.label not in {"UNKNOWN", "IDLE", "TRANSITION"} and smoother.should_emit(smoothed.label):
                        sentence.append(smoothed.label)
                        sentence = sentence[-MAX_SENTENCE_WORDS:]

                if not fixed_sequence_mode and motion_state == IDLE and now - latest_ai_result_time > PREDICTION_INTERVAL_SECONDS:
                    latest_decision = Decision("IDLE", 1.0, "motion", "idle")
                elif not fixed_sequence_mode and motion_state == TRANSITION:
                    latest_decision = Decision("TRANSITION", 1.0, "motion", "movement_not_complete")

                camera_frame_count += 1
                elapsed = frame_started - fps_window_started
                if elapsed >= 1.0:
                    camera_fps = camera_frame_count / elapsed
                    ai_fps = ai_result_count / elapsed
                    camera_frame_count = 0
                    ai_result_count = 0
                    fps_window_started = frame_started

                cv2.rectangle(image, (0, 0), (image.shape[1], 52), (40, 40, 40), -1)
                cv2.putText(
                    image,
                    " ".join(sentence) if sentence else "Recognized words will appear here",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                image = draw_top3_prediction_box(image, latest_model_res, model_actions, top_n=TOP_N)
                memory_lines = ["Memory matches:"]
                if latest_memory_matches:
                    memory_lines.extend(
                        f"{index + 1}. {readable_label(match.label)} ({match.similarity * 100:.2f}%)"
                        for index, match in enumerate(latest_memory_matches)
                    )
                else:
                    memory_lines.append("No memory match yet.")

                mode_label = "FIXED" if fixed_sequence_mode else "MOTION"
                put_panel_text(
                    image,
                    [
                        f"Prediction: {readable_label(latest_decision.label)} ({latest_decision.confidence * 100:.2f}%)",
                        f"Source: {latest_decision.source} | Reason: {latest_decision.reason}",
                        f"Model: {readable_label(latest_model.label)} ({latest_model.confidence * 100:.2f}%)",
                        f"Memory: {readable_label(latest_memory.label)} ({latest_memory.confidence * 100:.2f}%)",
                        f"Motion: {motion_state}",
                        "T teach | C correct/clear | R reset | F mode | Q quit",
                    ],
                    start_y=285,
                )
                put_panel_text(image, memory_lines, start_y=485)
                draw_performance_panel(image, camera_fps, ai_fps, latest_prediction_time_ms, mode_label, latest_used_memory)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    sentence.clear()
                    smoother.clear()
                if key == ord("f"):
                    fixed_sequence_mode = not fixed_sequence_mode
                    fixed_buffer.clear()
                    sign_buffer.clear()
                    pre_roll.clear()
                    smoother.clear()
                    motion_detector.reset()
                if key == ord("t"):
                    matcher = teach_new_sign(cap, holistic, frames=model_sequence_length)
                    worker.update_matcher(matcher)
                    print("Reloaded memory examples:", len(matcher.examples))
                if key == ord("c"):
                    saved = correct_prediction(latest_decision, latest_completed_sequence)
                    if not saved:
                        sentence.clear()
                        smoother.clear()
                    else:
                        matcher = DTWMemoryMatcher.from_memory(MAX_MEMORY_SAMPLES_PER_SIGN)
                        worker.update_matcher(matcher)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with suppress_library_stderr():
        main()
