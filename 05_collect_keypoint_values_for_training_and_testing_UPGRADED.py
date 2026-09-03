"""Collect 30-frame sign samples into MP_Data/sign_name/sample/frame.npy."""

import time

import cv2
import numpy as np

from sign_language_common import (
    CAMERA_INDEX,
    DATA_PATH,
    SEQUENCE_LENGTH,
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
    open_camera,
    sanitize_action_name,
    update_actions_from_folders,
)
from sign_language_dataset import next_sample_dir
from sign_memory import record_sample


WINDOW_NAME = "Sign Sample Collection"
BUTTON_RECT = (20, 370, 250, 430)
record_requested = False


def ask_action_name() -> str:
    while True:
        action = sanitize_action_name(input("Enter sign name to collect: "))
        if action:
            return action
        print("Please type a sign name, for example: hello or thank_you")


def ask_recording_count() -> int:
    while True:
        raw_value = input("How many samples to record each time? [1]: ").strip()
        if not raw_value:
            return 1
        try:
            sample_count = int(raw_value)
        except ValueError:
            print("Please type a whole number, for example: 5")
            continue
        if sample_count > 0:
            return sample_count
        print("Sample count must be at least 1.")


def mouse_callback(event, x, y, flags, param):
    global record_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BUTTON_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            record_requested = True


def put_text(image, text, pos, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_idle_screen(image, action, next_sample, sample_count):
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 95), (20, 20, 20), -1)
    put_text(image, f"Sign: {action}", (15, 32), 0.8, (0, 255, 255))
    put_text(image, f"Next sample: {next_sample}", (15, 65), 0.65)
    put_text(image, f"Record count: {sample_count}", (300, 65), 0.65)
    put_text(image, "SPACE/click = collect + teach AI | N = new sign | B = sample count | Q/ESC = quit", (15, height - 15), 0.5)

    x1, y1, x2, y2 = BUTTON_RECT
    if y2 < height:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 120, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        put_text(image, "START RECORD", (x1 + 18, y1 + 38), 0.75)


def show_countdown(cap, holistic, action) -> bool:
    for number in (3, 2, 1):
        start = time.time()
        while time.time() - start < 0.8:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera opened, but no frame could be read.")
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            put_text(image, f"Prepare: {action}", (20, 40), 0.9, (0, 255, 255))
            put_text(image, str(number), (image.shape[1] // 2 - 30, image.shape[0] // 2), 3.0, (0, 255, 255), 5)
            cv2.imshow(WINDOW_NAME, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False
    return True


def collect_sample(cap, holistic, action, sample_num) -> bool:
    sample_dir = DATA_PATH / action / str(sample_num)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera opened, but no frame could be read.")

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        np.save(sample_dir / f"{frame_num}.npy", keypoints)

        height, width = image.shape[:2]
        cv2.rectangle(image, (0, 0), (width, 90), (0, 0, 0), -1)
        put_text(image, f"Recording: {action}", (15, 32), 0.8, (0, 255, 255))
        put_text(image, f"Sample {sample_num} | Frame {frame_num + 1}/{SEQUENCE_LENGTH}", (15, 66), 0.65)
        progress = int((frame_num + 1) / SEQUENCE_LENGTH * (width - 30))
        cv2.rectangle(image, (15, height - 45), (width - 15, height - 25), (80, 80, 80), -1)
        cv2.rectangle(image, (15, height - 45), (15 + progress, height - 25), (0, 255, 0), -1)
        cv2.imshow(WINDOW_NAME, image)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False

    print(f"Saved sample: {sample_dir}")
    try:
        item = record_sample(action, sample_dir, source="camera")
        print(f"Taught AI memory: memory/{item['path']}")
    except Exception as exc:
        print(f"Sample saved, but memory update failed: {exc}")
    return True


def main():
    global record_requested

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    action = ask_action_name()
    sample_count = ask_recording_count()
    (DATA_PATH / action).mkdir(parents=True, exist_ok=True)
    update_actions_from_folders()

    cap = open_camera(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            next_sample = int(next_sample_dir(action).name)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera opened, but no frame could be read.")

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            draw_idle_screen(image, action, next_sample, sample_count)
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                action = ask_action_name()
                (DATA_PATH / action).mkdir(parents=True, exist_ok=True)
                update_actions_from_folders()
            if key == ord("b"):
                sample_count = ask_recording_count()
            if key == 32:
                record_requested = True

            if record_requested:
                record_requested = False
                stop_collection = False
                for recording_index in range(sample_count):
                    sample_num = int(next_sample_dir(action).name)
                    print(f"Recording {action}/{sample_num} sample {recording_index + 1}/{sample_count}")
                    if not show_countdown(cap, holistic, action):
                        stop_collection = True
                        break
                    if not collect_sample(cap, holistic, action, sample_num):
                        stop_collection = True
                        break
                    update_actions_from_folders()
                if stop_collection:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
