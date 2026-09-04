Sign Language Recognition - Upgraded Run Order

Active training dataset layout:

MP_Data/sign_name/sample_number/frame_number.npy
30 frames per sample by default
1662 keypoint values per frame

Immediate memory layout:

memory/memory.json
memory/examples/sign_name/example_001.npy
memory/corrections/sign_name/correction_001.npy
memory/mistakes.jsonl

Versioned model layout:

models/model_v001.keras
models/model_v001_actions.json
models/model_registry.json
models/best_model.keras
action.keras
action.h5
model_actions.json

Fresh run order:

pip install -r requirements.txt
python 01_import_and_install_dependencies_UPGRADED.py
python 02_keypoints_using_mp_holistic_UPGRADED.py
python 04_setup_folders_for_collection_UPGRADED.py
python 05_collect_keypoint_values_for_training_and_testing_UPGRADED.py
python 13_check_dataset.py
python 06_preprocess_data_and_create_labels_and_features_UPGRADED.py
python 07_build_and_train_lstm_neural_network_UPGRADED.py
python 10_evaluation_using_confusion_matrix_and_accuracy_UPGRADED.py
python 11_test_in_real_time_UPGRADED.py

Useful commands:

python 14_teach_new_sign_memory.py --label SIGN_NAME --examples 3
python 15_merge_memory_to_dataset.py
python 16_switch_model_version.py --list
python 16_switch_model_version.py model_v001
python 17_set_action_metadata.py hello dynamic
python 12_migrate_batch_folders_to_samples.py --dry-run

Notes:

- Use the UPGRADED scripts for normal work. The original numbered scripts are
  kept for reference and backward compatibility.
- Configuration defaults live in sign_ai/config.py and can be overridden with
  environment variables such as SL_CAMERA_INDEX or SL_SEQUENCE_LENGTH.
- Training accepts variable-length samples and resamples/pads them to the target
  model sequence length.
- Default model input remains landmark-only 1662 features. Optional velocity
  features are available with:
  python 07_build_and_train_lstm_neural_network_UPGRADED.py --feature-mode landmark_plus_velocity
- Available training architectures are LSTM (default), GRU, bidirectional
  LSTM, and MLP. To train and activate an MLP model, run:
  python 07_build_and_train_lstm_neural_network_UPGRADED.py --model-type mlp
- Dataset checking is non-destructive and only reports issues.
- Training backs up an existing processed_data.npz before writing a new one.
- Training saves versioned models in models/ and also updates action.keras for
  compatibility with the old workflow.
- Real-time recognition defaults to fixed rolling 30-frame recognition so it
  starts predicting shortly after the camera opens. Motion-triggered recognition
  is still available with:
  python 11_test_in_real_time_UPGRADED.py --motion-sequence
- When fixed mode is toggled off, the script still falls back to rolling
  predictions while a hand is visible, so recognition does not go silent if
  motion completion is not detected.
- Real-time recognition combines the trained model, immediate memory matching,
  unknown handling, and smoothing.
- Normal data collection also teaches immediate memory. Every saved sample from
  05_collect_keypoint_values_for_training_and_testing_UPGRADED.py is saved to
  MP_Data for retraining and to memory/examples for immediate recognition.
- Real-time controls:
  T = teach new sign into MP_Data and memory
  C = correct current prediction, or press ENTER at the prompt to clear
  R = reset sentence
  F = toggle fixed 30-frame mode / motion-triggered mode
  Q/ESC = quit
- The teach command and realtime T key already save to both MP_Data and memory.
  To copy older memory-only examples into neural training, run:
  python 15_merge_memory_to_dataset.py
  python 07_build_and_train_lstm_neural_network_UPGRADED.py
- If camera does not open, change CAMERA_INDEX in sign_ai/config.py or run with:
  set SL_CAMERA_INDEX=0
