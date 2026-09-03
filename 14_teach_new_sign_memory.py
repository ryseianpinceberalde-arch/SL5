"""Teach a sign into MP_Data and immediate memory without retraining."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from sign_ai.config import CAMERA_INDEX, SEQUENCE_LENGTH
from sign_ai.memory.memory_manager import teach_sequence
from sign_language_common import (
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
    open_camera,
    sanitize_action_name,
)


WINDOW_NAME = "Teach New Sign Memory"


def ask_label(current: str | None = None) -> str:
    while True:
        suffix = f" [{current}]" if current else ""
        label = sanitize_action_name(input(f"Sign name{suffix}: "))
        if label:
            return label
        if current:
            return current
        print("Please type a sign name, for example: thank_you")


def show_countdown(cap, holistic, label: str) -> bool:
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
                return False
    return True


def record_sequence(cap, holistic, label: str, frames: int) -> np.ndarray | None:
    sequence = []
    for frame_num in range(frames):
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received.")
            return None
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        sequence.append(extract_keypoints(results))
        cv2.putText(image, f"Teaching: {label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(image, f"Frame {frame_num + 1}/{frames}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return None
    return np.array(sequence, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Teach sign examples into MP_Data and immediate memory.")
    parser.add_argument("--label")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--frames", type=int, default=SEQUENCE_LENGTH)
    return parser.parse_args()


def main():
    args = parse_args()
    label = sanitize_action_name(args.label) if args.label else ask_label()
    if not label:
        raise ValueError("Sign label cannot be empty.")

    cap = open_camera(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for example_index in range(args.examples):
                print(f"Recording memory example {example_index + 1}/{args.examples} for {label}")
                if not show_countdown(cap, holistic, label):
                    break
                sequence = record_sequence(cap, holistic, label, args.frames)
                if sequence is None:
                    break
                item, sample_dir = teach_sequence(label, sequence, source="teach")
                print(f"Saved training sample: {sample_dir}")
                print(f"Taught AI memory: memory/{item['path']}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
