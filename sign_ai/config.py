"""Central configuration for the sign-language recognition system."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "MP_Data"
ACTIONS_FILE = DATA_PATH / "actions.json"
ACTIONS_METADATA_FILE = DATA_PATH / "actions_metadata.json"

PROCESSED_DATA_FILE = BASE_DIR / "processed_data.npz"
TEST_DATA_FILE = BASE_DIR / "test_data.npz"
MODEL_PATH = BASE_DIR / "action.keras"
LEGACY_MODEL_PATH = BASE_DIR / "action.h5"
MODEL_ACTIONS_FILE = BASE_DIR / "model_actions.json"

MEMORY_DIR = BASE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.json"
MEMORY_EXAMPLES_DIR = MEMORY_DIR / "examples"
MEMORY_CORRECTIONS_DIR = MEMORY_DIR / "corrections"
MISTAKE_LOG_FILE = MEMORY_DIR / "mistakes.jsonl"

MODELS_DIR = BASE_DIR / "models"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"

TRAINING_RESULTS_DIR = BASE_DIR / "training_results"
PROCESSED_BACKUP_DIR = BASE_DIR / "processed_data_backups"

SEQUENCE_LENGTH = int(os.getenv("SL_SEQUENCE_LENGTH", "30"))
KEYPOINT_LENGTH = 1662
CAMERA_INDEX = int(os.getenv("SL_CAMERA_INDEX", "1"))
CAMERA_WIDTH = int(os.getenv("SL_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("SL_CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("SL_CAMERA_FPS", "30"))

POSE_VALUES = 33 * 4
FACE_VALUES = 468 * 3
HAND_VALUES = 21 * 3

LANDMARK_ONLY = "landmark_only"
LANDMARK_PLUS_VELOCITY = "landmark_plus_velocity"
FEATURE_MODE = os.getenv("SL_FEATURE_MODE", LANDMARK_ONLY).strip().lower()

CONFIDENCE_THRESHOLD = float(os.getenv("SL_CONFIDENCE_THRESHOLD", "0.70"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("SL_MEMORY_SIMILARITY_THRESHOLD", "0.86"))
UNKNOWN_THRESHOLD = float(os.getenv("SL_UNKNOWN_THRESHOLD", "0.55"))
PREDICTION_WINDOW = int(os.getenv("SL_PREDICTION_WINDOW", "10"))
STABLE_FRAMES = int(os.getenv("SL_STABLE_FRAMES", "6"))
PREDICTION_COOLDOWN = float(os.getenv("SL_PREDICTION_COOLDOWN", "1.2"))
MAX_SENTENCE_WORDS = int(os.getenv("SL_MAX_SENTENCE_WORDS", "8"))
TOP_N = int(os.getenv("SL_TOP_N", "3"))

MOTION_START_THRESHOLD = float(os.getenv("SL_MOTION_START_THRESHOLD", "0.020"))
MOTION_STOP_THRESHOLD = float(os.getenv("SL_MOTION_STOP_THRESHOLD", "0.010"))
START_CONFIRM_FRAMES = int(os.getenv("SL_START_CONFIRM_FRAMES", "2"))
STOP_CONFIRM_FRAMES = int(os.getenv("SL_STOP_CONFIRM_FRAMES", "5"))
MIN_SIGN_FRAMES = int(os.getenv("SL_MIN_SIGN_FRAMES", "12"))
MAX_SIGN_FRAMES = int(os.getenv("SL_MAX_SIGN_FRAMES", "90"))
STATIC_CONFIRM_FRAMES = int(os.getenv("SL_STATIC_CONFIRM_FRAMES", "18"))

USE_AUGMENTATION = os.getenv("SL_USE_AUGMENTATION", "0") in {"1", "true", "True"}
AUGMENTATION_FACTOR = int(os.getenv("SL_AUGMENTATION_FACTOR", "1"))

MAX_MEMORY_SAMPLES_PER_SIGN = int(os.getenv("SL_MAX_MEMORY_SAMPLES_PER_SIGN", "12"))


def ensure_runtime_dirs() -> None:
    """Create non-dataset runtime directories if they are needed."""
    for path in [DATA_PATH, MEMORY_DIR, MEMORY_EXAMPLES_DIR, MEMORY_CORRECTIONS_DIR, MODELS_DIR, TRAINING_RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
