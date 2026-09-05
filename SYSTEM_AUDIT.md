# Current Architecture

Audited on 2026-09-04 in `C:\Users\Ryse\Downloads\SL5`.

The project currently has three real-time test scripts:

- `11_test_in_real_time.py`: legacy/simple tutorial-style rolling 30-frame LSTM recognizer.
- `11_test_in_real_time_UPGRADED.py`: documented normal upgraded runtime with motion, memory, teach, correction, smoothing, and UI panels.
- `11_test_in_real_time_OPTIMIZED.py`: newer optimized runtime with background prediction worker, camera FPS/AI FPS/prediction time display, no-hand reset, and throttled prediction. This file is modified in the current working tree.

Current upgraded architecture:

```text
Camera
  |
  v
MediaPipe Holistic
  |
  v
extract_keypoints -> 1662 raw values
  |
  v
normalize_keypoints -> 1662 normalized values
  |
  +-----------------------------+
  |                             |
  v                             v
Fixed rolling buffer       Motion detector + pre-roll/sign buffer
  |                             |
  +-------------+---------------+
                |
                v
fit_sequence_length -> model input length
                |
      +---------+----------+
      |                    |
      v                    v
LSTM model prediction   DTW memory matching
      |                    |
      +---------+----------+
                |
                v
Decision engine
                |
                v
Prediction smoother / sentence emit
                |
                v
OpenCV UI
```

## Components

| Component | File | Class/function | Purpose | Input | Output | Status | Reuse |
|---|---|---|---|---|---|---|---|
| Camera opening | `sign_language_common.py` | `open_camera` | Opens configured webcam and sets width/height. | Camera index | `cv2.VideoCapture` | Working | Reuse |
| Camera FPS tuning | `11_test_in_real_time_OPTIMIZED.py` | `configure_camera_capture` | Requests FPS, size, buffer, optional MJPG. | Capture object/config | Actual camera settings | Working in optimized script | Reuse if optimized script is chosen |
| MediaPipe init | Real-time and collection scripts | `mp_holistic.Holistic(...)` | Creates Holistic detector. | Detection/tracking thresholds | MediaPipe context | Working | Reuse |
| MediaPipe processing | `sign_language_common.py` | `mediapipe_detection` | Converts BGR/RGB and runs MediaPipe. | Frame, model | Processed image, results | Working | Reuse |
| Landmark drawing | `sign_language_common.py` | `draw_styled_landmarks` | Draws face, pose, hands. | Image, results | Mutated image | Working | Reuse |
| Keypoint extraction | `sign_language_common.py` | `extract_keypoints` | Produces fixed pose+face+hands vector. | MediaPipe results | `(1662,) float32` | Working | Reuse |
| Normalization | `sign_ai/features/landmark_normalizer.py` | `normalize_keypoints`, `normalize_sequence` | Normalizes pose/hands while preserving length. | Raw keypoints/sequence | Same shape normalized data | Working | Reuse |
| Sequence fitting | `sign_ai/features/sequence_tools.py` | `fit_sequence_length` | Resamples/pads/trims variable sequences. | Sequence, target length | Fixed-length sequence | Working | Reuse |
| Sequence buffers | `sign_ai/recognition/sequence_buffer.py` | `SequenceBuffer`, `PreRollBuffer` | Maintains rolling/preroll frame buffers. | Keypoint frames | Sequence arrays | Working | Reuse |
| Motion recognition | `sign_ai/recognition/motion_detector.py` | `MotionDetector` | Detects idle, recording, transition, complete. | Keypoints + hand presence | `MotionEvent` | Working but threshold-sensitive | Reuse/improve |
| Model loading | `sign_language_common.py` | `load_model_safely` | Loads `action.keras`, `models/best_model.keras`, or `action.h5`. | Files on disk | Keras model | Working | Reuse |
| Label loading | `sign_language_common.py` | `load_model_actions`, `load_actions` | Loads model labels and collected actions. | JSON files/folders | Label lists | Working | Reuse |
| Model prediction | Real-time scripts | `model_candidate_for_sequence` | Fits sequence, applies feature mode, runs `model.predict`. | Sequence | Candidate + full probabilities | Working | Reuse |
| Top 3 model UI | Real-time upgraded/optimized | `draw_top3_prediction_box` | Displays top model probabilities. | Prediction vector, labels | Image | Working | Reuse |
| Memory persistence | `sign_ai/memory/memory_manager.py` | `load_memory`, `save_memory`, `save_memory_example` | Stores registry + `.npy` sequence samples. | Label, sequence | JSON metadata + `.npy` | Working | Reuse |
| Memory matching | `sign_ai/memory/memory_matcher.py` | `DTWMemoryMatcher` | DTW/mean-absolute-distance nearest examples. | Sequence | Top memory matches | Working but O(samples) | Reuse/improve |
| Decision engine | `sign_ai/recognition/decision_engine.py` | `choose_decision` | Chooses motion, memory, model, weak/unknown result. | Candidates + thresholds | `Decision` | Partial: memory can override model first | Improve |
| Prediction smoothing | `sign_ai/recognition/prediction_smoother.py` | `PredictionSmoother` | Stable label window + cooldown. | Decisions | Optional smoothed prediction | Working | Reuse |
| Teach mode | `11_test_in_real_time_UPGRADED.py`, `11_test_in_real_time_OPTIMIZED.py`, `14_teach_new_sign_memory.py` | `teach_new_sign`, `teach_sequence` | Records examples and stores them. | Label + webcam sequence | Memory sample and MP_Data sample | Partial: no duplicate/quality gate | Improve |
| Correction mode | Real-time upgraded/optimized, `memory_manager.py` | `correct_prediction`, `save_correction` | Saves current sequence under corrected label. | Previous decision + sequence + user label | Memory example, correction copy, mistakes log | Partial: no duplicate/quality gate and no UI learning status | Improve |
| Dataset promotion | `15_merge_memory_to_dataset.py` | `merge_memory_to_dataset` | Copies memory examples into `MP_Data`. | Memory registry | New sample folders | Partial: can duplicate teach samples already saved to dataset | Improve |
| Dataset checking | `sign_ai/training/dataset_checker.py` | `check_dataset` | Validates frame counts, shapes, finite values, exact duplicate hashes. | `MP_Data` | Health report | Working for dataset, not memory | Reuse/extend |
| Training | `07_build_and_train_lstm_neural_network_UPGRADED.py` | `build_model`, `main` | Builds processed data, trains versioned LSTM/GRU/BiLSTM. | `MP_Data` | Versioned model, metrics, compatibility files | Working | Reuse |

## Verified Current Data/Model Compatibility

- `processed_data.npz`: `X=(240, 30, 1662)`, `y=(240, 8)`, `feature_mode=landmark_only`.
- `test_data.npz`: `X_test=(36, 30, 1662)`, `y_test=(36, 8)`, `feature_mode=landmark_only`.
- `action.keras`, `models/best_model.keras`, `models/model_v003.keras`: input `(None, 30, 1662)`, output `(None, 8)`.
- `model_actions.json` and `MP_Data/actions.json`: `good_morning`, `hello`, `how`, `iloveyou`, `no`, `sorry`, `thankyou`, `yes`.
- Current `MP_Data`: 8 labels, 30 samples each, 240 total valid samples by `python 13_check_dataset.py`.
- Current `memory/examples`: 8 labels, 30 `.npy` examples each, 240 examples.
