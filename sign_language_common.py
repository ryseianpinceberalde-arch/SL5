"""Shared helpers for the upgraded sign language recognition scripts.

The original tutorial was written as notebook cells. These helpers make the
upgraded scripts safe to run one file at a time from the terminal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from sign_ai.config import (
    ACTIONS_FILE,
    BASE_DIR,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DATA_PATH,
    KEYPOINT_LENGTH,
    LEGACY_MODEL_PATH,
    MEMORY_FILE,
    MODEL_ACTIONS_FILE,
    MODEL_PATH,
    POSE_VALUES,
    FACE_VALUES,
    HAND_VALUES,
    PROCESSED_DATA_FILE,
    SEQUENCE_LENGTH,
    TEST_DATA_FILE,
)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def sanitize_action_name(raw_name: str) -> str:
    """Convert user text into a safe folder name."""
    name = raw_name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return re.sub(r"_+", "_", name).strip("_")


def ensure_data_path() -> Path:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return DATA_PATH


def read_actions_from_folders() -> list[str]:
    """Read action names from MP_Data/sign_name folders."""
    ensure_data_path()
    return sorted(
        item.name
        for item in DATA_PATH.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    )


def save_actions(actions: Iterable[str]) -> list[str]:
    """Save action labels in the exact order used by training/testing."""
    ensure_data_path()
    clean_actions = sorted(dict.fromkeys(action for action in actions if action))
    ACTIONS_FILE.write_text(json.dumps(clean_actions, indent=2), encoding="utf-8")
    return clean_actions


def save_model_actions(actions: Iterable[str]) -> list[str]:
    """Save the action labels that belong to the trained model output."""
    clean_actions = list(dict.fromkeys(action for action in actions if action))
    MODEL_ACTIONS_FILE.write_text(json.dumps(clean_actions, indent=2), encoding="utf-8")
    return clean_actions


def load_actions(require_existing: bool = True) -> list[str]:
    """Load saved actions, or rebuild them from MP_Data folders."""
    if ACTIONS_FILE.exists():
        actions = json.loads(ACTIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(actions, list) and all(isinstance(item, str) for item in actions):
            return actions
        raise ValueError(f"{ACTIONS_FILE.name} is not a valid list of action labels.")

    actions = read_actions_from_folders()
    if actions:
        return save_actions(actions)

    if require_existing:
        raise FileNotFoundError(
            "No actions found. Create sign folders or collect samples first."
        )
    return []


def load_model_actions(require_existing: bool = True) -> list[str]:
    """Load the labels used by the current trained model."""
    if MODEL_ACTIONS_FILE.exists():
        actions = json.loads(MODEL_ACTIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(actions, list) and all(isinstance(item, str) for item in actions):
            return actions
        raise ValueError(f"{MODEL_ACTIONS_FILE.name} is not a valid list of model labels.")

    return load_actions(require_existing=require_existing)


def update_actions_from_folders() -> list[str]:
    """Refresh actions.json from current MP_Data folders."""
    actions = read_actions_from_folders()
    return save_actions(actions)


def next_sample_number(action: str) -> int:
    """Return the next available numeric sample folder for an action."""
    action_dir = DATA_PATH / action
    action_dir.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(item.name)
        for item in action_dir.iterdir()
        if item.is_dir() and item.name.isdigit()
    ]
    return max(numbers) + 1 if numbers else 0


def open_camera(camera_index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open a webcam and raise a clear error if it is unavailable."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Camera {camera_index} could not be opened. "
            "Close other camera apps, check Windows camera permissions, "
            "or change CAMERA_INDEX in sign_language_common.py."
        )

    # Request a larger frame so previews are not limited to the webcam default.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


def face_connections():
    """MediaPipe renamed face connection constants across versions."""
    return getattr(
        mp_holistic,
        "FACE_CONNECTIONS",
        mp.solutions.face_mesh.FACEMESH_CONTOURS,
    )


def mediapipe_detection(image, model):
    """Run MediaPipe on one OpenCV BGR frame."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results) -> None:
    """Draw available face, pose, and hand landmarks."""
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        face_connections(),
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def _fit_length(values: np.ndarray, expected_length: int) -> np.ndarray:
    """Pad or trim landmark arrays so model input length never changes."""
    values = values.astype(np.float32).flatten()
    if values.size == expected_length:
        return values
    if values.size > expected_length:
        return values[:expected_length]
    return np.pad(values, (0, expected_length - values.size))


def extract_keypoints(results) -> np.ndarray:
    """Return exactly 1662 values: pose, face, left hand, right hand."""
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(POSE_VALUES, dtype=np.float32)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.face_landmarks
        else np.zeros(FACE_VALUES, dtype=np.float32)
    )
    left_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(HAND_VALUES, dtype=np.float32)
    )
    right_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(HAND_VALUES, dtype=np.float32)
    )

    keypoints = np.concatenate(
        [
            _fit_length(pose, POSE_VALUES),
            _fit_length(face, FACE_VALUES),
            _fit_length(left_hand, HAND_VALUES),
            _fit_length(right_hand, HAND_VALUES),
        ]
    ).astype(np.float32)
    return _fit_length(keypoints, KEYPOINT_LENGTH)


def valid_sample_folder(sample_dir: Path) -> bool:
    """True when a sample has readable numeric frames of 1662 finite values."""
    if not sample_dir.is_dir():
        return False
    frame_paths = sorted(
        [path for path in sample_dir.glob("*.npy") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    if not frame_paths:
        return False
    for frame_path in frame_paths:
        try:
            data = np.load(frame_path)
        except Exception:
            return False
        if data.shape != (KEYPOINT_LENGTH,):
            return False
        if not np.all(np.isfinite(data)):
            return False
    return True


def load_model_safely():
    """Load the preferred .keras model, with old action.h5 as fallback."""
    from tensorflow.keras.models import load_model

    from sign_ai.config import BEST_MODEL_PATH

    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    if BEST_MODEL_PATH.exists():
        return load_model(BEST_MODEL_PATH)
    if LEGACY_MODEL_PATH.exists():
        return load_model(LEGACY_MODEL_PATH, compile=False)
    raise FileNotFoundError(
        "No trained model found. Run 07_build_and_train_lstm_neural_network_UPGRADED.py first."
    )
