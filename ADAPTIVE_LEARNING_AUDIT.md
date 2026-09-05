# 1. Current System Summary

Audited on 2026-09-04 in `C:\Users\Ryse\Downloads\SL5`. No trained model, dataset, memory sample, or runtime behavior was modified during this audit.

The project is not only the original tutorial scripts anymore. It now has a reusable `sign_ai` package with shared configuration, feature processing, recognition helpers, memory storage/matching, training support, and versioned model management.

The documented normal runtime is `11_test_in_real_time_UPGRADED.py` according to `README_RUN_ORDER.txt`. The newer `11_test_in_real_time_OPTIMIZED.py` appears to be the most feature-complete live test program because it adds camera FPS, AI prediction FPS, prediction time, no-hand reset, camera FPS configuration, and a background prediction worker. It is currently modified in the working tree, so later implementation should inspect the exact diff before editing it.

Current pipeline:

```text
Camera
  v
MediaPipe Holistic
  v
extract_keypoints -> 1662 raw landmarks
  v
normalize_keypoints -> 1662 normalized landmarks
  v
Sequence buffer: fixed rolling window or motion-triggered sign buffer
  v
LSTM model prediction + DTW immediate-memory matching
  v
choose_decision
  v
PredictionSmoother
  v
OpenCV UI with sentence, Top 3 model predictions, memory matches, source, motion, FPS
```

# 2. Important Files

| File | Purpose | Status |
|---|---|---|
| `11_test_in_real_time.py` | Legacy/simple rolling 30-frame model-only test. | [EXISTS] Working baseline, lacks adaptive memory features. |
| `11_test_in_real_time_UPGRADED.py` | Real-time model + memory + motion + teach + correction. | [EXISTS] Main documented upgraded runtime. |
| `11_test_in_real_time_OPTIMIZED.py` | Optimized real-time runtime with prediction thread, FPS panels, no-hand reset. | [PARTIAL] Most complete, but currently modified in git working tree. |
| `sign_language_common.py` | Shared MediaPipe, extraction, camera, model/label helpers. | [EXISTS] Reuse. |
| `sign_ai/config.py` | Central paths, sequence length, thresholds, camera config, memory caps. | [PARTIAL] Missing auto-learning and duplicate thresholds. |
| `sign_ai/features/landmark_normalizer.py` | Pose/hand normalization while preserving 1662 features. | [EXISTS] Reuse for memory/model consistency. |
| `sign_ai/features/sequence_tools.py` | Resample/pad/trim sequence lengths. | [EXISTS] Reuse. |
| `sign_ai/features/motion_features.py` | Velocity/feature mode and movement magnitude. | [EXISTS] Reuse. |
| `sign_ai/recognition/sequence_buffer.py` | Rolling and pre-roll frame buffers. | [EXISTS] Reuse. |
| `sign_ai/recognition/motion_detector.py` | IDLE/RECORDING/TRANSITION/SIGN_COMPLETE state machine. | [EXISTS] Reuse/tune. |
| `sign_ai/recognition/prediction_smoother.py` | Stable prediction window and sentence cooldown. | [EXISTS] Reuse. |
| `sign_ai/recognition/decision_engine.py` | Combines motion/model/memory candidates. | [PARTIAL] Needs safer conflict policy. |
| `sign_ai/memory/memory_manager.py` | Persistent memory registry, examples, corrections, mistake log, merge support. | [PARTIAL] Strong base, missing duplicate and quality gates. |
| `sign_ai/memory/memory_matcher.py` | DTW memory matching. | [PARTIAL] Works, but scales linearly with loaded samples. |
| `sign_memory.py` | Backward-compatible wrapper around new memory package. | [EXISTS] Reuse for legacy compatibility. |
| `05_collect_keypoint_values_for_training_and_testing_UPGRADED.py` | Collects samples into `MP_Data` and also records memory. | [EXISTS] Reuse; lacks duplicate/quality gate. |
| `06_preprocess_data_and_create_labels_and_features_UPGRADED.py` | Builds `processed_data.npz`. | [EXISTS] Reuse. |
| `07_build_and_train_lstm_neural_network_UPGRADED.py` | Trains versioned LSTM/GRU/BiLSTM models. | [EXISTS] Reuse. |
| `13_check_dataset.py` | Dataset health report. | [EXISTS] Reuse/extend. |
| `14_teach_new_sign_memory.py` | Standalone teach script. | [PARTIAL] Saves memory and dataset, but no duplicate/quality gate. |
| `15_merge_memory_to_dataset.py` | Promotes memory examples into `MP_Data`. | [PARTIAL] Needs safer batch/promotion semantics. |
| `16_switch_model_version.py` | Switches active model version. | [EXISTS] Reuse. |
| `17_set_action_metadata.py` | Marks action type metadata. | [EXISTS] Reuse. |
| `memory/memory.json` | Persistent memory registry. | [EXISTS] 240 registered examples verified. |
| `memory/examples/<label>/*.npy` | Stored memory sequences. | [EXISTS] 8 labels x 30 examples verified. |
| `memory/corrections/` | Correction sequence copies. | [EXISTS] Directory configured; used by corrections. |
| `memory/mistakes.jsonl` | Mistake/correction log. | [EXISTS] Used by corrections. |
| `MP_Data/<label>/<sample>/<frame>.npy` | Training dataset. | [EXISTS] 240 valid samples verified. |
| `processed_data.npz` | Processed training arrays. | [EXISTS] `(240, 30, 1662)` verified. |
| `test_data.npz` | Saved evaluation split. | [EXISTS] `(36, 30, 1662)` verified. |
| `action.keras`, `models/best_model.keras`, `models/model_v003.keras` | Current model files. | [EXISTS] Input `(None, 30, 1662)`, output `(None, 8)` verified. |
| `model_actions.json`, `MP_Data/actions.json` | Label order files. | [EXISTS] Both contain the same 8 labels. |

# 3. Current Recognition Pipeline

[EXISTS] Camera opening is centralized in `sign_language_common.open_camera`. The optimized script additionally calls `configure_camera_capture` to request FPS, width, height, buffer size, and MJPG for high requested FPS.

[EXISTS] MediaPipe uses `mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)` in the runtime and collection scripts.

[EXISTS] Keypoint extraction uses `sign_language_common.extract_keypoints`, returning exactly 1662 values: pose `33*4`, face `468*3`, left hand `21*3`, right hand `21*3`.

[EXISTS] The upgraded/optimized runtime normalizes extracted keypoints before recognition with `normalize_keypoints`. Training also normalizes samples through `sign_language_dataset.load_sample`, so live memory/model comparison is mostly consistent.

[EXISTS] Sequence length is configured as `SEQUENCE_LENGTH=30` in `sign_ai/config.py`, but the optimized runtime detects `model.input_shape[1]` and uses that as `model_sequence_length`.

[EXISTS] LSTM prediction is implemented in `model_candidate_for_sequence`, using `fit_sequence_length`, `apply_feature_mode`, and `model.predict`.

[EXISTS] Top 3 model predictions are drawn by `draw_top3_prediction_box`.

[EXISTS] Smoothing is handled by `PredictionSmoother` using `PREDICTION_WINDOW=10`, `STABLE_FRAMES=6`, and `PREDICTION_COOLDOWN=1.2`.

[EXISTS] Idle/motion recognition is handled by `MotionDetector`, and the optimized runtime separately handles no-hand reset after `NO_HAND_RESET_SECONDS=0.75`.

# 4. Current Memory System

1. Memory currently stores JSON metadata in `memory/memory.json` and full sequence `.npy` files under `memory/examples/<label>/example_###.npy`.
2. Full frame sequences are stored. Verified sample shape: `(30, 1662)`.
3. Raw MediaPipe keypoints are collected first, then memory examples are normally stored normalized through `save_memory_example(normalize=True)`.
4. Memory does not store only labels/confidence; it stores labels, paths, source metadata, sequence length, feature length, timestamps, and full sequences.
5. Memory is persistent after closing the program because it is stored on disk.
6. Memory can recognize labels that are not part of LSTM training, as long as the sequence is saved in memory and the decision engine accepts the memory candidate. This is independent of model output labels.
7. Similarity uses DTW-like sequence comparison over mean absolute frame distance.
8. Memory uses nearest-neighbor over stored examples with DTW distance, converted to similarity by `1 / (1 + distance * 35)`.
9. The T teach feature is implemented, but only partially safe.
10. In live runtime, pressing T prompts for a label and number of examples, shows a countdown, records frames, then saves through `teach_sequence`.
11. Pressing C prompts for a correct label. Empty input clears the sentence. A non-empty label saves the current completed sequence under the corrected label through `save_correction`.
12. A wrong model prediction can be corrected into memory.
13. The correction is stored as a memory example and also copied under `memory/corrections/<label>/correction_###.npy`; a mistake record is appended to `memory/mistakes.jsonl`.
14. The stored correction can affect future predictions after the matcher is reloaded, because it becomes part of the memory examples matched by `DTWMemoryMatcher`.

Important limitation: there is no duplicate-protection gate before saving memory examples or corrections.

# 5. Current Teach Feature

[EXISTS] Standalone teach script: `14_teach_new_sign_memory.py`.

[EXISTS] Live T key: `teach_new_sign` in both upgraded and optimized real-time scripts.

[PARTIAL] Teach supports multiple examples and immediate memory recognition without LSTM retraining.

[NEEDS IMPROVEMENT] Teach currently saves to both `MP_Data` and `memory/examples` by default. This is useful but blurs "live memory" and "training dataset promotion"; the requested architecture wants live memory and neural retraining data to remain separate unless promoted intentionally.

[NEEDS IMPROVEMENT] Optimized live teach only appends frames when a hand is detected, but then `teach_sequence` can accept a shorter sequence and fit it later for dataset saving. Memory itself stores the variable length unless the recording produced exactly model length. That is flexible, but the user requested same sequence shape as the model input for adaptive memory.

[MISSING] No duplicate detection.

[MISSING] No reusable memory sample quality validator.

# 6. Current Correction Feature

[EXISTS] `correct_prediction` is implemented in live runtime.

[EXISTS] `save_correction` stores the corrected label, previous predicted label, model confidence, a correction `.npy`, and a mistakes log line.

[PARTIAL] Corrections immediately affect memory matching after the runtime reloads/updates the matcher.

[NEEDS IMPROVEMENT] Corrections do not validate that the current sequence has a hand, correct length, enough non-empty frames, no duplicated sample, or a non-system label.

[NEEDS IMPROVEMENT] The UI prints correction status to the console, not a persistent on-screen learning status like `Correction saved`.

# 7. Existing Features

[EXISTS] Python/OpenCV webcam.

[EXISTS] MediaPipe Holistic landmarks/keypoints.

[EXISTS] TensorFlow/Keras LSTM model.

[EXISTS] Real-time sign prediction.

[EXISTS] Fixed sequence recognition.

[EXISTS] Model confidence and top 3 model predictions.

[EXISTS] Idle state and no-hand reset behavior in optimized runtime.

[EXISTS] Camera FPS, AI prediction FPS, and prediction time in optimized runtime.

[EXISTS] Teach function using T.

[EXISTS] Correction/clear function using C.

[EXISTS] Reset using R.

[EXISTS] Recognition mode toggle using F.

[EXISTS] Memory prediction and memory matches.

[EXISTS] Trained action/sign labels.

[EXISTS] Saved training sequences.

[EXISTS] Persistent memory JSON and `.npy` examples.

[EXISTS] Separate model versioning and normal retraining script.

# 8. Missing Features

[MISSING] Safe automatic learning.

[MISSING] Auto-learning config values: `AUTO_LEARN_ENABLED`, `AUTO_LEARN_MIN_CONFIDENCE`, `AUTO_LEARN_STABLE_COUNT`, `AUTO_LEARN_MIN_DIFFERENCE`, `AUTO_LEARN_MIN_HAND_FRAMES`, `AUTO_LEARN_MIN_MOTION`, and explicit denied system labels.

[MISSING] Duplicate protection for memory samples.

[MISSING] Reusable memory sequence validator.

[MISSING] Explicit prevention of saving `IDLE`, `UNKNOWN`, `no_hand`, `Waiting`, `TRANSITION`, or other system states as examples.

[MISSING] Learning status UI state.

[MISSING] Memory decision conflict handling that returns `Uncertain` when model and memory strongly disagree.

[MISSING] Background-safe memory reload/update protocol beyond simple matcher replacement.

[MISSING] Promotion script that creates explicit `adaptive_batch_###` groups and avoids overwriting or duplicating current dataset samples.

# 9. Problems Found

1. `choose_decision` gives a strong memory match priority over a strong model match without checking whether labels agree. This can let memory override the model too aggressively.
2. `teach_sequence` saves to both memory and `MP_Data` by default. That conflicts with the requested separation between live memory and later retraining promotion.
3. `15_merge_memory_to_dataset.py` can promote examples that were already saved into `MP_Data` by teach/collection unless `merged_dataset_paths` is already present. Many examples have `source_dataset_path`, not necessarily `merged_dataset_paths`.
4. No duplicate check exists in `save_memory_example`, `teach_sequence`, or `save_correction`.
5. Memory validation only checks array dimensionality, feature length, non-empty sequence, and finite values. It does not check hand presence, mostly-empty sequences, exact target length, minimum motion, or system labels.
6. The optimized runtime runs memory matching only when the model candidate is missing or below `CONFIDENCE_THRESHOLD`. This protects FPS, but means memory agreement is unavailable for high-confidence model predictions and cannot support the requested `model+memory` confirmation or safe auto-learning conditions.
7. There is duplicated recognition logic between `11_test_in_real_time_UPGRADED.py` and `11_test_in_real_time_OPTIMIZED.py`.
8. There is also legacy extraction/training logic in old numbered scripts alongside upgraded shared helpers. Later changes should target shared helpers and the chosen runtime, not all scripts.
9. `README_RUN_ORDER.txt` recommends `11_test_in_real_time_UPGRADED.py`, but the optimized runtime contains the performance UI requested by the user. The project should choose one primary runtime before adaptive-learning changes.

# 10. Adaptive Learning Requirements

Adaptive memory should reuse the existing `memory/` system instead of creating a separate `adaptive_memory/` tree unless the user explicitly wants a rename. The current layout already matches the requested concept closely:

```text
memory/
  memory.json
  examples/
    yes/
      example_001.npy
  corrections/
    yes/
      correction_001.npy
  mistakes.jsonl
```

The live memory sample shape should be forced to the model-compatible shape detected from the loaded model: currently `(30, 1662)`.

Required behavior:

- Store approved examples separately from model weights.
- Do not call `model.fit()` inside the camera loop.
- Validate samples before saving.
- Reject system states as labels.
- Detect duplicates before saving.
- Let memory classify signs immediately after teach/correction.
- Use memory/model agreement for final decisions and safe learning.
- Keep promotion into the neural dataset as an explicit script-driven step.

# 11. Files That Need Modification

| File | Change needed | Risk |
|---|---|---|
| `11_test_in_real_time_OPTIMIZED.py` | Integrate safe learning status, stricter teach/correction calls, optional memory agreement pass, auto-learning hook, and safer decision display. | Medium |
| `11_test_in_real_time_UPGRADED.py` | Either mirror the same changes or mark optimized runtime as primary to avoid duplicate behavior. | Medium |
| `sign_ai/config.py` | Add adaptive-learning, duplicate, denied-label, and quality thresholds. | Low |
| `sign_ai/memory/memory_manager.py` | Add validation/duplicate checks and memory-only teach default or explicit mode. | Medium |
| `sign_ai/memory/memory_matcher.py` | Add duplicate/similarity helper and possibly vectorized/prototype cache later. | Low-Medium |
| `sign_ai/recognition/decision_engine.py` | Add model+memory agreement and disagreement/uncertain policy. | Medium |
| `14_teach_new_sign_memory.py` | Use the same validator/duplicate policy as live teach. | Low |
| `15_merge_memory_to_dataset.py` | Promote into explicit adaptive batches or otherwise avoid duplicate dataset copies. | Medium |
| `README_RUN_ORDER.txt` | Document the chosen primary runtime and adaptive-learning workflow after implementation. | Low |

# 12. New Files Recommended

Only create these if keeping the code clean is easier than extending current modules:

| File | Reason |
|---|---|
| `sign_ai/memory/memory_validator.py` | Central reusable validation for hand presence, shape, finite values, mostly-empty frames, motion amount, system label rejection. |
| `sign_ai/memory/duplicate_detector.py` | Central duplicate/similarity gate for teach, correction, auto-learning, and promotion. This could also live in `memory_matcher.py` if kept small. |
| `sign_ai/learning/auto_learner.py` | Optional. Use only if auto-learning logic becomes too large for the runtime script. |
| `promote_memory_to_dataset.py` | Safer replacement or complement for `15_merge_memory_to_dataset.py`, if explicit adaptive batches are required. |

Do not create `adaptive_memory.py` yet; the project already has `sign_ai/memory/memory_manager.py`.

# 13. Database/File Storage Recommendation

Reuse the current disk-backed memory structure:

```text
memory/
  memory.json
  examples/<label>/example_###.npy
  corrections/<label>/correction_###.npy
  mistakes.jsonl
```

Add metadata fields instead of changing the layout:

- `approved: true`
- `quality`
- `duplicate_similarity`
- `source`: `realtime_teach`, `correction`, `auto_learn`
- `model_version`
- `model_label`
- `model_confidence`
- `memory_agreement`
- `sequence_shape`

This preserves existing memory and avoids a duplicate adaptive memory system.

# 14. Auto Learning Safety Rules

Auto-learning should remain disabled by default until teach/correction validation is solid.

Recommended rules:

- `AUTO_LEARN_ENABLED=False` by default.
- Never save system labels: `IDLE`, `UNKNOWN`, `TRANSITION`, `Waiting`, `No Match`, `no_hand`, `FIXED_SEQUENCE`.
- Require a hand in enough frames.
- Require exact model-compatible shape after fitting.
- Require finite values and non-empty hand landmarks.
- Require `model_confidence >= AUTO_LEARN_MIN_CONFIDENCE`, for example `0.97`.
- Require stable smoothed prediction for configured count.
- Require memory agreement with the model if memory has examples for that label.
- Require sufficient difference from existing examples.
- Cap examples per label.
- Save with source `auto_learn` and metadata for audit.

# 15. Performance Risks

Current memory matching compares the incoming sequence against representative samples for each label with DTW. It uses `MAX_MEMORY_SAMPLES_PER_SIGN=12`, so with 8 labels it loads at most 96 examples for matching despite 240 stored examples.

The optimized runtime protects FPS by using a background prediction worker and `PREDICTION_INTERVAL_SECONDS=0.15`. It also skips DTW when the model is already confident. That helps performance but prevents memory agreement checks for high-confidence predictions.

If auto-learning needs memory agreement on high-confidence predictions, use one of these before comparing every frame against everything:

- Run memory only at the throttled prediction interval.
- Keep `MAX_MEMORY_SAMPLES_PER_SIGN`.
- Use representative examples or per-label prototypes.
- Cache loaded memory arrays.
- Run matching in the existing prediction worker.
- Add a cheap duplicate check before DTW if needed.

# 16. Recommended Implementation Order

STEP 1: Choose primary runtime. Prefer `11_test_in_real_time_OPTIMIZED.py` because it contains the performance UI and worker, but reconcile this with `README_RUN_ORDER.txt`.

STEP 2: Add reusable memory validation using existing config/model shape/keypoint helpers.

STEP 3: Add duplicate detection before saving memory examples.

STEP 4: Update teach mode to save model-shaped memory samples and report on-screen learning status.

STEP 5: Update correction mode to reject invalid/system labels, save only under the corrected label, and report status.

STEP 6: Improve `choose_decision` so model+memory agreement is explicit and strong disagreement becomes `UNCERTAIN` instead of blind override.

STEP 7: Add optional memory agreement checks in the optimized worker without blocking the camera loop.

STEP 8: Add safe auto-learning disabled by default, with strict config gates.

STEP 9: Improve memory-to-dataset promotion so it preserves existing data and avoids duplicate copies.

STEP 10: Run dataset check, shape checks, and compile/import checks after each stage.

# Existing Adaptive-Learning Components

- Persistent memory registry: `memory/memory.json`.
- Persistent full-sequence memory examples: `memory/examples/<label>/example_###.npy`.
- Correction storage: `memory/corrections/<label>/correction_###.npy`.
- Mistake log: `memory/mistakes.jsonl`.
- Live T teach mode.
- Live C correction mode.
- Immediate memory matching using DTW nearest examples.
- Decision engine combining model, memory, motion, and unknown handling.
- Separate retraining script.
- Existing memory-to-dataset merge script.

# Missing Components

- Duplicate protection before memory save.
- Strong quality validation before memory save.
- System-label rejection before memory save.
- Safe auto-learning.
- Auto-learning config.
- Model+memory agreement decision path.
- Strong disagreement/uncertain decision path.
- On-screen learning status.
- Safe adaptive batch promotion workflow.

# Files Requiring Changes

- `11_test_in_real_time_OPTIMIZED.py`
- `11_test_in_real_time_UPGRADED.py` or retire it as the primary runtime path
- `sign_ai/config.py`
- `sign_ai/memory/memory_manager.py`
- `sign_ai/memory/memory_matcher.py`
- `sign_ai/recognition/decision_engine.py`
- `14_teach_new_sign_memory.py`
- `15_merge_memory_to_dataset.py`
- `README_RUN_ORDER.txt`

# New Files Recommended

- `sign_ai/memory/memory_validator.py`
- `sign_ai/memory/duplicate_detector.py` or small duplicate helpers inside `memory_matcher.py`
- `promote_memory_to_dataset.py` only if `15_merge_memory_to_dataset.py` cannot be safely evolved
- `sign_ai/learning/auto_learner.py` only if auto-learning logic becomes too large for the runtime script

# Highest Priority

Fix the safety layer before adding auto-learning: validation, duplicate detection, denied system labels, and safer model/memory decision conflicts. After that, improve teach/correction using the existing memory system. Only then add disabled-by-default safe auto-learning.

AUDIT COMPLETE
