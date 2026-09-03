"""Check imports for the upgraded project."""

import cv2
import mediapipe as mp
import numpy as np
import sklearn
import tensorflow as tf


def main():
    print("All main imports worked.")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"scikit-learn module: {sklearn.__version__}")


if __name__ == "__main__":
    main()
