"""Verify that keypoint extraction always returns 1662 values."""

import cv2

from sign_language_common import (
    CAMERA_INDEX,
    KEYPOINT_LENGTH,
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
    open_camera,
)


def main():
    cap = open_camera(CAMERA_INDEX)
    print("Press SPACE to print one keypoint vector shape. Press Q or ESC to quit.")

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
            cv2.putText(
                image,
                "SPACE = test keypoints | Q/ESC = quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Extract Keypoints Test", image)

            key = cv2.waitKey(10) & 0xFF
            if key == 32:
                keypoints = extract_keypoints(results)
                print(f"extract_keypoints shape: {keypoints.shape}")
                if keypoints.shape != (KEYPOINT_LENGTH,):
                    raise ValueError("Keypoint extraction did not return 1662 values.")
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
