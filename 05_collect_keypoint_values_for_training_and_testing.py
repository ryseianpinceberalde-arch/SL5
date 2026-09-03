#!/usr/bin/env python
# coding: utf-8
"""
UPGRADED ADD-ON for: 05_collect_keypoint_values_for_training_and_testing.py

This file does NOT delete or replace your old collection file.
It keeps your original idea:
    MP_Data / sign_name / sequence_number / frame_number.npy
    30 frames per sequence
    MediaPipe Holistic keypoints: pose + face + left hand + right hand = 1662 values

New features added:
    - Type the sign/action name before collecting
    - Click START RECORD button or press SPACE to collect one 30-frame movement sample
    - Countdown before recording
    - Progress display: frame 1/30 ... 30/30
    - Automatic next sequence number
    - Creates folders automatically
    - Saves/updates MP_Data/actions.txt with every sign name you collect
    - Warns if hand is not detected enough during the movement

Controls:
    SPACE or click START RECORD = collect one sample
    N = change sign/action name
    H = turn hand-quality check ON/OFF
    Q or ESC = quit
"""

import os
import re
import shutil
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Same important settings as your old files
# -----------------------------
DATA_PATH = Path("MP_Data")
SEQUENCE_LENGTH = 30              # same 30-frame movement sample
MIN_HAND_FRAMES = 15              # minimum frames where at least one hand should be visible
REQUIRE_HAND_QUALITY = True       # press H to toggle this while running
CAMERA_INDEX = 1                  # change to 1 if you use another webcam/phone camera


# -----------------------------
# MediaPipe setup from your 02 and 03 files
# -----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def face_connections():
    """Compatible face connections for old/new MediaPipe versions."""
    return getattr(mp_holistic, "FACE_CONNECTIONS", mp.solutions.face_mesh.FACEMESH_CONTOURS)


def mediapipe_detection(image, model):
    """Convert BGR -> RGB, process MediaPipe, then convert RGB -> BGR."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draw face, pose, left hand, and right hand landmarks."""
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


def extract_keypoints(results):
    """Return one 1662-value vector from MediaPipe results."""
    pose = (
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def hand_detected(results):
    """True if at least one hand is visible in the current frame."""
    return bool(results.left_hand_landmarks or results.right_hand_landmarks)


# -----------------------------
# Folder and label helpers
# -----------------------------
def sanitize_action_name(raw_name):
    """Make a safe folder name from a sign/action name."""
    name = raw_name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def ask_action_name(current_name=None):
    """Ask the user for the sign/action name in the terminal."""
    while True:
        prompt = "Enter sign/action name"
        if current_name:
            prompt += f" [{current_name}]"
        prompt += ": "

        raw = input(prompt)
        if not raw.strip() and current_name:
            return current_name

        name = sanitize_action_name(raw)
        if name:
            return name

        print("Please type a valid name, example: hello, thank_you, please")


def ensure_action_folder(action):
    action_dir = DATA_PATH / action
    action_dir.mkdir(parents=True, exist_ok=True)
    update_actions_txt(action)
    return action_dir


def update_actions_txt(action):
    """Save collected sign names for future automatic training upgrades."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    actions_file = DATA_PATH / "actions.txt"

    actions = []
    if actions_file.exists():
        actions = [line.strip() for line in actions_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    if action not in actions:
        actions.append(action)
        actions_file.write_text("\n".join(sorted(actions)) + "\n", encoding="utf-8")


def next_sequence_number(action):
    """Find the next available numbered sequence folder for this action."""
    action_dir = ensure_action_folder(action)
    numbers = []
    for item in action_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            numbers.append(int(item.name))
    return max(numbers) + 1 if numbers else 0


def safe_make_sequence_dir(action, sequence_id):
    seq_dir = DATA_PATH / action / str(sequence_id)
    seq_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir


# -----------------------------
# UI helpers
# -----------------------------
WINDOW_NAME = "Manual Sign Movement Collection"
BUTTON_RECT = (20, 390, 250, 450)  # x1, y1, x2, y2
record_requested = False


def mouse_callback(event, x, y, flags, param):
    global record_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BUTTON_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            record_requested = True


def put_text(image, text, position, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(image, action, sequence_id, status, quality_on):
    """Draw instructions and START button."""
    height, width = image.shape[:2]

    # top bar
    cv2.rectangle(image, (0, 0), (width, 105), (25, 25, 25), -1)
    put_text(image, f"SIGN: {action}", (15, 32), 0.8, (0, 255, 255), 2)
    put_text(image, f"NEXT SAMPLE: {sequence_id}", (15, 65), 0.65, (255, 255, 255), 2)
    put_text(image, f"STATUS: {status}", (15, 95), 0.6, (180, 255, 180), 2)

    # controls
    put_text(image, "SPACE/click = record | N = new sign | H = hand check | Q/ESC = quit", (15, height - 15), 0.45, (255, 255, 255), 1)
    hand_text = "Hand quality: ON" if quality_on else "Hand quality: OFF"
    put_text(image, hand_text, (width - 230, 32), 0.55, (255, 255, 255), 1)

    # button
    x1, y1, x2, y2 = BUTTON_RECT
    if y2 < height:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 120, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        put_text(image, "START RECORD", (x1 + 18, y1 + 38), 0.75, (255, 255, 255), 2)


def show_countdown(cap, holistic, action):
    """Show 3-2-1 countdown before movement recording."""
    for number in [3, 2, 1]:
        start = time.time()
        while time.time() - start < 0.75:
            ret, frame = cap.read()
            if not ret:
                continue
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.rectangle(image, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
            put_text(image, f"Prepare: {action}", (20, 35), 0.8, (255, 255, 255), 2)
            put_text(image, str(number), (image.shape[1] // 2 - 30, image.shape[0] // 2), 3.0, (0, 255, 255), 5)
            cv2.imshow(WINDOW_NAME, image)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                return False
    return True


def collect_one_sequence(cap, holistic, action, sequence_id, require_hand_quality=True):
    """Record exactly 30 frames and save them in MP_Data/action/sequence_id."""
    seq_dir = safe_make_sequence_dir(action, sequence_id)
    hand_frames = 0

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not read. Stopping this sample.")
            return False, 0

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        if hand_detected(results):
            hand_frames += 1

        keypoints = extract_keypoints(results)
        np.save(seq_dir / f"{frame_num}.npy", keypoints)

        # Recording overlay
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, 0), (width, 115), (0, 0, 0), -1)
        put_text(image, f"RECORDING: {action}", (15, 35), 0.8, (0, 255, 255), 2)
        put_text(image, f"Sample {sequence_id} | Frame {frame_num + 1}/{SEQUENCE_LENGTH}", (15, 70), 0.65, (255, 255, 255), 2)
        put_text(image, f"Hand frames: {hand_frames}/{frame_num + 1}", (15, 100), 0.55, (180, 255, 180), 2)

        # progress bar
        progress = int((frame_num + 1) / SEQUENCE_LENGTH * (width - 30))
        cv2.rectangle(image, (15, height - 45), (width - 15, height - 25), (80, 80, 80), -1)
        cv2.rectangle(image, (15, height - 45), (15 + progress, height - 25), (0, 255, 0), -1)

        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            return False, hand_frames

    if require_hand_quality and hand_frames < MIN_HAND_FRAMES:
        shutil.rmtree(seq_dir, ignore_errors=True)
        print(f"Rejected sample {sequence_id}: hand detected only {hand_frames}/{SEQUENCE_LENGTH} frames.")
        return False, hand_frames

    print(f"Saved: {seq_dir} | hand frames: {hand_frames}/{SEQUENCE_LENGTH}")
    return True, hand_frames


def main():
    global record_requested, REQUIRE_HAND_QUALITY

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    action = ask_action_name()
    ensure_action_folder(action)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera cannot open. Try changing CAMERA_INDEX from 0 to 1.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    status = "Ready. Press SPACE or click START RECORD."

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            sequence_id = next_sequence_number(action)

            ret, frame = cap.read()
            if not ret:
                status = "Camera frame not available."
                continue

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            draw_panel(image, action, sequence_id, status, REQUIRE_HAND_QUALITY)
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(10) & 0xFF

            if key in [ord("q"), 27]:
                break

            if key == ord("h"):
                REQUIRE_HAND_QUALITY = not REQUIRE_HAND_QUALITY
                status = "Hand quality check ON." if REQUIRE_HAND_QUALITY else "Hand quality check OFF."

            if key == ord("n"):
                print("\nChange sign/action name")
                action = ask_action_name(current_name=action)
                ensure_action_folder(action)
                status = f"Changed sign to {action}. Ready to collect."

            if key == 32:  # SPACE
                record_requested = True

            if record_requested:
                record_requested = False
                status = "Countdown... prepare your movement."

                if not show_countdown(cap, holistic, action):
                    break

                sequence_id = next_sequence_number(action)
                saved, hand_frames = collect_one_sequence(
                    cap,
                    holistic,
                    action,
                    sequence_id,
                    require_hand_quality=REQUIRE_HAND_QUALITY,
                )

                if saved:
                    status = f"Saved sample {sequence_id}. Press SPACE/click for another."
                else:
                    status = f"Sample not saved. Hand frames: {hand_frames}/{SEQUENCE_LENGTH}. Try again."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
