"""Dataset loading and legacy MP_Data migration helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from sign_ai.config import FEATURE_MODE, LANDMARK_ONLY, PROCESSED_BACKUP_DIR
from sign_ai.features.landmark_normalizer import normalize_sequence
from sign_ai.features.motion_features import apply_feature_mode
from sign_ai.features.sequence_tools import fit_sequence_length
from sign_language_common import (
    DATA_PATH,
    KEYPOINT_LENGTH,
    PROCESSED_DATA_FILE,
    SEQUENCE_LENGTH,
    update_actions_from_folders,
    valid_sample_folder,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    X: np.ndarray
    y: np.ndarray
    actions: list[str]
    counts: dict[str, int]
    skipped: list[str]
    sample_paths: list[str]
    feature_mode: str = LANDMARK_ONLY


@dataclass(frozen=True)
class MigrationMove:
    action: str
    source: Path
    destination: Path


def relative_to_data(path: Path, data_path: Path = DATA_PATH) -> str:
    return path.resolve().relative_to(data_path.resolve()).as_posix()


def _require_inside(path: Path, base: Path) -> Path:
    resolved_base = base.resolve()
    resolved_path = path.resolve()
    resolved_path.relative_to(resolved_base)
    return resolved_path


def _numeric_dirs(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return sorted(
        [item for item in parent.iterdir() if item.is_dir() and item.name.isdigit()],
        key=lambda item: int(item.name),
    )


def _legacy_batch_dirs(action_dir: Path) -> list[Path]:
    if not action_dir.exists():
        return []

    def batch_sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("batch")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return sorted(
        [
            item
            for item in action_dir.iterdir()
            if item.is_dir() and item.name.startswith("batch")
        ],
        key=batch_sort_key,
    )


def find_sample_dirs(action_dir: Path, include_legacy_batches: bool = False) -> list[Path]:
    """Return sample folders in MP_Data/sign/sample format.

    Legacy MP_Data/sign/batch/sample folders are only included when requested,
    so training no longer depends on batches after migration.
    """
    sample_dirs = _numeric_dirs(action_dir)
    if include_legacy_batches:
        for batch_dir in _legacy_batch_dirs(action_dir):
            sample_dirs.extend(_numeric_dirs(batch_dir))
    return sample_dirs


def next_sample_dir(action: str, data_path: Path = DATA_PATH) -> Path:
    action_dir = data_path / action
    action_dir.mkdir(parents=True, exist_ok=True)
    used = {int(item.name) for item in _numeric_dirs(action_dir)}
    sample_num = 0
    while sample_num in used:
        sample_num += 1
    return action_dir / str(sample_num)


def sample_frame_paths(sample_dir: Path) -> list[Path]:
    return sorted(
        [path for path in sample_dir.glob("*.npy") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )


def load_raw_sample(sample_dir: Path) -> np.ndarray:
    if not valid_sample_folder(sample_dir):
        raise ValueError(f"Incomplete or broken sample folder: {sample_dir}")
    frame_paths = sample_frame_paths(sample_dir)
    return np.array(
        [np.load(frame_path) for frame_path in frame_paths],
        dtype=np.float32,
    )


def load_sample(
    sample_dir: Path,
    target_length: int = SEQUENCE_LENGTH,
    normalize: bool = True,
    feature_mode: str = LANDMARK_ONLY,
) -> np.ndarray:
    sample = load_raw_sample(sample_dir)
    if normalize:
        sample = normalize_sequence(sample)
    sample = fit_sequence_length(sample, target_length=target_length)
    return apply_feature_mode(sample, feature_mode=feature_mode)


def _one_hot(labels: list[int], action_count: int) -> np.ndarray:
    y = np.zeros((len(labels), action_count), dtype=np.float32)
    if labels:
        y[np.arange(len(labels)), labels] = 1.0
    return y


def build_dataset(
    include_legacy_batches: bool = False,
    target_length: int = SEQUENCE_LENGTH,
    normalize: bool = True,
    feature_mode: str = FEATURE_MODE,
) -> DatasetBuildResult:
    actions = update_actions_from_folders()
    if not actions:
        raise RuntimeError("No signs found in MP_Data. Collect samples first.")

    label_map = {label: number for number, label in enumerate(actions)}
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    sample_paths: list[str] = []
    counts = {action: 0 for action in actions}
    skipped: list[str] = []

    for action in actions:
        action_dir = DATA_PATH / action
        for sample_dir in find_sample_dirs(action_dir, include_legacy_batches=include_legacy_batches):
            if not valid_sample_folder(sample_dir):
                skipped.append(relative_to_data(sample_dir))
                continue
            sequences.append(
                load_sample(
                    sample_dir,
                    target_length=target_length,
                    normalize=normalize,
                    feature_mode=feature_mode,
                )
            )
            labels.append(label_map[action])
            sample_paths.append(relative_to_data(sample_dir))
            counts[action] += 1

    if not sequences:
        raise RuntimeError("No complete 30-frame samples found. Run data collection first.")

    X = np.array(sequences, dtype=np.float32)
    y = _one_hot(labels, len(actions))

    if X.shape[1] != target_length:
        raise ValueError(f"Expected X sequence length {target_length}, got {X.shape}")
    if X.shape[2] < KEYPOINT_LENGTH:
        raise ValueError(f"Expected at least {KEYPOINT_LENGTH} features, got {X.shape}")

    return DatasetBuildResult(
        X=X,
        y=y,
        actions=actions,
        counts=counts,
        skipped=skipped,
        sample_paths=sample_paths,
        feature_mode=feature_mode,
    )


def backup_existing_processed_file(output_path: Path) -> Path | None:
    if not output_path.exists():
        return None
    PROCESSED_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = PROCESSED_BACKUP_DIR / f"{output_path.stem}_{timestamp}{output_path.suffix}"
    shutil.copy2(output_path, backup_path)
    return backup_path


def save_processed_dataset(result: DatasetBuildResult, output_path: Path = PROCESSED_DATA_FILE) -> None:
    backup_path = backup_existing_processed_file(output_path)
    if backup_path:
        print(f"Backed up existing processed data to: {backup_path}")
    np.savez_compressed(
        output_path,
        X=result.X,
        y=result.y,
        actions=np.array(result.actions),
        sample_paths=np.array(result.sample_paths),
        feature_mode=np.array(result.feature_mode),
        sequence_length=np.array(result.X.shape[1]),
        feature_length=np.array(result.X.shape[2]),
    )


def build_and_save_processed_dataset(
    output_path: Path = PROCESSED_DATA_FILE,
    include_legacy_batches: bool = False,
    target_length: int = SEQUENCE_LENGTH,
    normalize: bool = True,
    feature_mode: str = FEATURE_MODE,
) -> DatasetBuildResult:
    result = build_dataset(
        include_legacy_batches=include_legacy_batches,
        target_length=target_length,
        normalize=normalize,
        feature_mode=feature_mode,
    )
    save_processed_dataset(result, output_path)
    return result


def plan_flatten_legacy_batches(data_path: Path = DATA_PATH) -> list[MigrationMove]:
    """Plan moves from MP_Data/sign/batch/sample to MP_Data/sign/sample."""
    data_path.mkdir(parents=True, exist_ok=True)
    _require_inside(data_path, data_path)
    moves: list[MigrationMove] = []

    for action_dir in sorted([item for item in data_path.iterdir() if item.is_dir()], key=lambda item: item.name):
        action = action_dir.name
        used_numbers = {int(item.name) for item in _numeric_dirs(action_dir)}
        next_number = 0

        for batch_dir in _legacy_batch_dirs(action_dir):
            for sample_dir in _numeric_dirs(batch_dir):
                while next_number in used_numbers:
                    next_number += 1
                destination = action_dir / str(next_number)
                used_numbers.add(next_number)
                next_number += 1
                moves.append(MigrationMove(action=action, source=sample_dir, destination=destination))

    return moves


def flatten_legacy_batches(data_path: Path = DATA_PATH, dry_run: bool = False) -> list[MigrationMove]:
    moves = plan_flatten_legacy_batches(data_path)
    if dry_run:
        return moves

    resolved_data_path = data_path.resolve()
    for move in moves:
        source = _require_inside(move.source, resolved_data_path)
        destination = _require_inside(move.destination, resolved_data_path)
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    for action_dir in sorted([item for item in data_path.iterdir() if item.is_dir()], key=lambda item: item.name):
        for batch_dir in _legacy_batch_dirs(action_dir):
            _require_inside(batch_dir, resolved_data_path)
            if batch_dir.exists() and not any(batch_dir.iterdir()):
                batch_dir.rmdir()

    update_actions_from_folders()
    return moves
