"""Train a versioned LSTM/GRU model from the current MP_Data samples."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, GRU, Input, LSTM
from tensorflow.keras.models import Sequential, load_model

from sign_ai.config import (
    AUGMENTATION_FACTOR,
    FEATURE_MODE,
    LANDMARK_ONLY,
    LANDMARK_PLUS_VELOCITY,
    MODEL_ACTIONS_FILE,
    MODEL_PATH,
    PROCESSED_DATA_FILE,
    SEQUENCE_LENGTH,
    TEST_DATA_FILE,
    TRAINING_RESULTS_DIR,
    USE_AUGMENTATION,
)
from sign_ai.training.augmentation import augment_dataset
from sign_ai.training.dataset_checker import check_dataset, print_report
from sign_ai.training.evaluation import save_evaluation_artifacts
from sign_ai.training.model_manager import (
    next_model_version,
    register_model,
    update_compatibility_files,
    write_actions_file,
)
from sign_language_common import save_actions, save_model_actions
from sign_language_dataset import build_and_save_processed_dataset


MIN_SAMPLES_PER_ACTION = 2
EPOCHS = 200
BATCH_SIZE = 32


def build_model(
    action_count: int,
    sequence_length: int,
    feature_length: int,
    model_type: str = "lstm",
    dropout: float = 0.2,
    use_batch_norm: bool = False,
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(sequence_length, feature_length)))

    if model_type == "gru":
        model.add(GRU(64, return_sequences=True, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
        model.add(GRU(128, return_sequences=True, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
        model.add(GRU(64, return_sequences=False, activation="relu"))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(64, return_sequences=True, activation="relu")))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(64, return_sequences=True, activation="relu")))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(32, return_sequences=False, activation="relu")))
    else:
        model.add(LSTM(64, return_sequences=True, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
        model.add(LSTM(64, return_sequences=False, activation="relu"))

    model.add(Dense(64, activation="relu"))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(action_count, activation="softmax"))
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train sign language sequence model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--feature-mode", choices=[LANDMARK_ONLY, LANDMARK_PLUS_VELOCITY], default=FEATURE_MODE)
    parser.add_argument("--model-type", choices=["lstm", "gru", "bilstm"], default="lstm")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-normalization", action="store_true")
    parser.add_argument("--use-cache", action="store_true", help="Use existing processed_data.npz.")
    parser.add_argument("--augment-factor", type=int, default=AUGMENTATION_FACTOR if USE_AUGMENTATION else 1)
    parser.add_argument("--no-legacy-h5", action="store_true", help="Skip writing action.h5 compatibility copy.")
    return parser.parse_args()


def load_training_data(args):
    if not args.use_cache:
        result = build_and_save_processed_dataset(
            target_length=args.sequence_length,
            normalize=True,
            feature_mode=args.feature_mode,
        )
        return result.X, result.y, result.actions, result.feature_mode

    if not PROCESSED_DATA_FILE.exists():
        result = build_and_save_processed_dataset(
            target_length=args.sequence_length,
            normalize=True,
            feature_mode=args.feature_mode,
        )
        return result.X, result.y, result.actions, result.feature_mode

    data = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    actions = [str(action) for action in data["actions"]]
    feature_mode = str(data["feature_mode"]) if "feature_mode" in data else LANDMARK_ONLY
    return X, y, actions, feature_mode


def split_dataset(X: np.ndarray, y: np.ndarray):
    labels = np.argmax(y, axis=1)
    stratify = labels if np.min(np.bincount(labels)) >= 3 else None
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=stratify,
        )
        temp_labels = np.argmax(y_temp, axis=1)
        temp_stratify = temp_labels if len(set(temp_labels)) > 1 and np.min(np.bincount(temp_labels)) >= 2 else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=temp_stratify,
        )
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def class_weights_for(y_train: np.ndarray) -> dict[int, float] | None:
    labels = np.argmax(y_train, axis=1)
    classes = np.unique(labels)
    if len(classes) <= 1:
        return None
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def main():
    args = parse_args()

    health = check_dataset()
    print_report(health)
    if not health.signs:
        raise RuntimeError("No dataset found. Collect sign samples before training.")

    X, y, actions, feature_mode = load_training_data(args)
    save_actions(actions)
    save_model_actions(actions)

    class_counts = y.sum(axis=0).astype(int)
    print(f"\nTraining with {len(X)} total complete samples.")
    print("Samples per sign:")
    for index, action in enumerate(actions):
        print(f"  - {action}: {class_counts[index]}")

    too_small = [actions[index] for index, count in enumerate(class_counts) if count < MIN_SAMPLES_PER_ACTION]
    if too_small:
        raise RuntimeError(
            "Not enough complete samples for training. "
            f"Need at least {MIN_SAMPLES_PER_ACTION} per sign. Missing: {too_small}"
        )

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    if args.augment_factor > 1:
        X_train, y_train = augment_dataset(X_train, y_train, factor=args.augment_factor)
        print(f"Augmented training data to {len(X_train)} samples.")

    version = next_model_version()
    version.model_path.parent.mkdir(parents=True, exist_ok=True)
    version.results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        action_count=len(actions),
        sequence_length=X.shape[1],
        feature_length=X.shape[2],
        model_type=args.model_type,
        dropout=args.dropout,
        use_batch_norm=args.batch_normalization,
    )
    callbacks = [
        TensorBoard(log_dir="Logs"),
        ModelCheckpoint(
            filepath=str(version.model_path),
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_categorical_accuracy",
            mode="max",
            patience=25,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights_for(y_train),
    )

    if not version.model_path.exists():
        model.save(version.model_path)

    best_model = load_model(version.model_path)
    metrics_dir = TRAINING_RESULTS_DIR / version.version_id
    metrics = save_evaluation_artifacts(
        best_model,
        actions,
        metrics_dir,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        history=history,
    )

    write_actions_file(version.actions_path, actions)
    metadata = {
        "model_type": args.model_type,
        "sequence_length": int(X.shape[1]),
        "feature_length": int(X.shape[2]),
        "feature_mode": feature_mode,
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "validation_accuracy": metrics["validation"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
    }
    register_model(version, actions, metadata, make_active=True, make_best=True)
    update_compatibility_files(version.model_path, actions, save_legacy_h5=not args.no_legacy_h5)

    np.savez_compressed(
        TEST_DATA_FILE,
        X_test=X_test,
        y_test=y_test,
        actions=np.array(actions),
        feature_mode=np.array(feature_mode),
        sequence_length=np.array(X.shape[1]),
        feature_length=np.array(X.shape[2]),
    )

    print(f"\nSaved versioned model: {version.model_path}")
    print(f"Saved compatibility model: {MODEL_PATH.name}")
    print(f"Saved model labels: {MODEL_ACTIONS_FILE.name}")
    print(f"Saved evaluation split: {TEST_DATA_FILE.name}")
    print(f"Saved training results: {metrics_dir}")


if __name__ == "__main__":
    main()
