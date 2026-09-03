"""Preview MediaPipe Holistic landmarks from the webcam."""

import cv2

from sign_language_common import (
    CAMERA_INDEX,
    draw_styled_landmarks,
    mediapipe_detection,
    mp_holistic,
    open_camera,
)


def main():
    cap = open_camera(CAMERA_INDEX)
    print("Camera opened. Press Q or ESC to quit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera opened, but no frame could be read.")

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow("MediaPipe Holistic Preview", image)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
