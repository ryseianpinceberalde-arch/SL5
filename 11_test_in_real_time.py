#!/usr/bin/env python
# coding: utf-8
"""Simple tutorial-style real-time sign recognition test."""

from contextlib import contextmanager
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import numpy as np
from scipy import stats

from sign_language_common import (
    draw_styled_landmarks,
    extract_keypoints,
    load_model_actions,
    load_model_safely,
    mediapipe_detection,
    mp_holistic,
)


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


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


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num % len(colors)]
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output_frame


def main():
    model = load_model_safely()
    actions = load_model_actions()

    if model.output_shape[-1] != len(actions):
        raise RuntimeError("Model output count does not match model_actions.json.")

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Camera 1 could not be opened. Try changing cv2.VideoCapture(1) to cv2.VideoCapture(0).")

    with suppress_library_stderr():
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                image, results = mediapipe_detection(frame, holistic)

                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predicted_index = int(np.argmax(res))
                    predictions.append(predicted_index)

                    if len(predictions) >= 10:
                        mode_result = stats.mode(predictions[-10:], keepdims=False)
                        stable_prediction = int(mode_result.mode) == predicted_index
                    else:
                        stable_prediction = True

                    if stable_prediction and res[predicted_index] > threshold:
                        if len(sentence) > 0:
                            if actions[predicted_index] != sentence[-1]:
                                sentence.append(actions[predicted_index])
                        else:
                            sentence.append(actions[predicted_index])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    image = prob_viz(res, actions, image, colors)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    " ".join(sentence),
                    (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("OpenCV Feed", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
