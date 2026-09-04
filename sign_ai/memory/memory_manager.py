"""Persistent immediate-memory examples and correction logs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sign_ai.config import (
    DATA_PATH,
    KEYPOINT_LENGTH,
    MEMORY_CORRECTIONS_DIR,
    MEMORY_EXAMPLES_DIR,
    MEMORY_FILE,
    MISTAKE_LOG_FILE,
    SEQUENCE_LENGTH,
    ensure_runtime_dirs,
)
from sign_ai.features.landmark_normalizer import normalize_sequence
from sign_ai.features.sequence_tools import fit_sequence_length
from sign_language_common import sanitize_action_name
from sign_language_common import update_actions_from_folders
from sign_language_dataset import next_sample_dir


MEMORY_VERSION = 2


@dataclass(frozen=True)
class MemoryExample:
    label: str
    path: Path
    sequence: np.ndarray
    source: str
    metadata: dict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_memory() -> dict:
    return {
        "version": MEMORY_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "examples": [],
        "labels": {},
    }


def load_memory() -> dict:
    ensure_runtime_dirs()
    if not MEMORY_FILE.exists():
        return _empty_memory()
    memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    if not isinstance(memory, dict):
        raise ValueError(f"{MEMORY_FILE} is not a valid memory registry.")
    memory.setdefault("version", MEMORY_VERSION)
    memory.setdefault("created_at", utc_now())
    memory.setdefault("updated_at", utc_now())
    memory.setdefault("examples", [])
    memory.setdefault("labels", {})
    return memory


def save_memory(memory: dict) -> dict:
    ensure_runtime_dirs()
    memory["version"] = MEMORY_VERSION
    memory["updated_at"] = utc_now()
    _refresh_label_stats(memory)
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    return memory


def _refresh_label_stats(memory: dict) -> None:
    labels: dict[str, dict] = {}
    for item in memory.get("examples", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        if not label:
            continue
        labels.setdefault(
            label,
            {
                "number_of_examples": 0,
                "sequence_lengths": [],
                "type": item.get("type", "dynamic"),
                "last_updated": item.get("created_at", utc_now()),
            },
        )
        labels[label]["number_of_examples"] += 1
        labels[label]["sequence_lengths"].append(int(item.get("sequence_length", 0)))
        labels[label]["last_updated"] = item.get("created_at", utc_now())
    memory["labels"] = dict(sorted(labels.items()))


def _next_memory_path(label: str, root: Path = MEMORY_EXAMPLES_DIR, prefix: str = "example") -> Path:
    safe_label = sanitize_action_name(label)
    if not safe_label:
        raise ValueError("Memory label cannot be empty.")
    label_dir = root / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for item in label_dir.glob(f"{prefix}_*.npy"):
        suffix = item.stem.removeprefix(f"{prefix}_")
        if suffix.isdigit():
            existing.append(int(suffix))
    next_number = max(existing) + 1 if existing else 1
    return label_dir / f"{prefix}_{next_number:03d}.npy"


def save_memory_example(
    label: str,
    sequence: np.ndarray,
    source: str = "teach",
    sign_type: str = "dynamic",
    notes: str = "",
    extra_metadata: dict | None = None,
    normalize: bool = True,
) -> dict:
    """Save one immediate-memory example without touching MP_Data."""
    ensure_runtime_dirs()
    safe_label = sanitize_action_name(label)
    if not safe_label:
        raise ValueError("Memory label cannot be empty.")

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != KEYPOINT_LENGTH or len(sequence) == 0:
        raise ValueError(f"Expected sequence shape (?, {KEYPOINT_LENGTH}), got {sequence.shape}")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("Cannot save memory example with NaN or infinite values.")

    stored_sequence = normalize_sequence(sequence) if normalize else sequence
    path = _next_memory_path(safe_label)
    np.save(path, stored_sequence.astype(np.float32))

    rel_path = path.relative_to(MEMORY_FILE.parent).as_posix()
    item = {
        "id": f"{safe_label}:{path.stem}",
        "label": safe_label,
        "path": rel_path,
        "source": source,
        "type": sign_type or "dynamic",
        "notes": notes,
        "sequence_length": int(len(stored_sequence)),
        "feature_length": int(stored_sequence.shape[1]),
        "created_at": utc_now(),
    }
    if extra_metadata:
        item.update(extra_metadata)

    memory = load_memory()
    memory["examples"].append(item)
    save_memory(memory)
    return item


def save_memory_example_from_sample(label: str, sample_dir: Path, source: str = "dataset") -> dict:
    source_dataset_path = str(sample_dir.resolve())
    memory = load_memory()
    for item in memory.get("examples", []):
        if isinstance(item, dict) and item.get("source_dataset_path") == source_dataset_path:
            return item

    frame_paths = sorted(
        [path for path in sample_dir.glob("*.npy") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    if not frame_paths:
        raise ValueError(f"No numeric .npy frames found in {sample_dir}")
    sequence = np.array([np.load(path) for path in frame_paths], dtype=np.float32)
    return save_memory_example(
        label,
        sequence,
        source=source,
        extra_metadata={"source_dataset_path": source_dataset_path},
    )


def save_sequence_to_dataset(
    label: str,
    sequence: np.ndarray,
    target_length: int = SEQUENCE_LENGTH,
) -> Path:
    """Save one recorded sequence into MP_Data/sign/sample/frame.npy."""
    safe_label = sanitize_action_name(label)
    if not safe_label:
        raise ValueError("Dataset label cannot be empty.")

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != KEYPOINT_LENGTH or len(sequence) == 0:
        raise ValueError(f"Expected sequence shape (?, {KEYPOINT_LENGTH}), got {sequence.shape}")

    sample_dir = next_sample_dir(safe_label, data_path=DATA_PATH)
    sample_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit_sequence_length(sequence, target_length=target_length)
    for frame_num, frame in enumerate(fitted):
        np.save(sample_dir / f"{frame_num}.npy", frame.astype(np.float32))

    update_actions_from_folders()
    return sample_dir


def teach_sequence(
    label: str,
    sequence: np.ndarray,
    source: str = "teach",
    sign_type: str = "dynamic",
    notes: str = "",
    save_to_dataset: bool = True,
    normalize_memory: bool = True,
) -> tuple[dict, Path | None]:
    """Teach a sequence by saving it to both MP_Data and immediate memory."""
    sample_dir = save_sequence_to_dataset(label, sequence) if save_to_dataset else None
    extra_metadata = {"source_dataset_path": str(sample_dir.resolve())} if sample_dir else None
    item = save_memory_example(
        label,
        sequence,
        source=source,
        sign_type=sign_type,
        notes=notes,
        extra_metadata=extra_metadata,
        normalize=normalize_memory,
    )
    return item, sample_dir


def load_memory_examples(max_per_label: int | None = None) -> list[MemoryExample]:
    memory = load_memory()
    examples: list[MemoryExample] = []
    grouped_items: dict[str, list[dict]] = defaultdict(list)
    for item in memory.get("examples", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        rel_path = str(item.get("path", ""))
        if not label or not rel_path:
            continue
        grouped_items[label].append(item)

    for label in sorted(grouped_items):
        items = _representative_memory_items(grouped_items[label], max_per_label)
        for item in items:
            rel_path = str(item.get("path", ""))
            path = (MEMORY_FILE.parent / rel_path).resolve()
            if not path.exists():
                continue
            sequence = np.load(path).astype(np.float32)
            if sequence.ndim != 2 or sequence.shape[1] != KEYPOINT_LENGTH:
                continue
            examples.append(
                MemoryExample(
                    label=label,
                    path=path,
                    sequence=sequence,
                    source=str(item.get("source", "")),
                    metadata=item,
                )
            )
    return examples


def _representative_memory_items(items: list[dict], max_per_label: int | None) -> list[dict]:
    """Pick examples across the whole label history instead of only newest items."""
    if max_per_label is None or max_per_label <= 0 or len(items) <= max_per_label:
        return items

    indices = np.linspace(0, len(items) - 1, max_per_label).round().astype(int)
    selected: list[dict] = []
    seen: set[int] = set()
    for index in indices:
        index = int(index)
        if index in seen:
            continue
        selected.append(items[index])
        seen.add(index)

    # Rounding can collide for small caps; fill from newest remaining examples.
    for index in range(len(items) - 1, -1, -1):
        if len(selected) >= max_per_label:
            break
        if index not in seen:
            selected.append(items[index])
            seen.add(index)
    return selected


def save_correction(
    predicted_label: str,
    correct_label: str,
    confidence: float,
    sequence: np.ndarray,
    normalize: bool = True,
) -> dict:
    safe_label = sanitize_action_name(correct_label)
    if not safe_label:
        raise ValueError("Correct label cannot be empty.")
    item = save_memory_example(
        safe_label,
        sequence,
        source="correction",
        extra_metadata={
            "predicted_label": predicted_label,
            "model_confidence": float(confidence),
        },
        normalize=normalize,
    )
    correction_path = _next_memory_path(safe_label, root=MEMORY_CORRECTIONS_DIR, prefix="correction")
    correction_sequence = normalize_sequence(np.asarray(sequence, dtype=np.float32)) if normalize else np.asarray(sequence, dtype=np.float32)
    np.save(correction_path, correction_sequence)
    log_mistake(predicted_label, safe_label, confidence, item["path"])
    return item


def log_mistake(predicted_label: str, actual_label: str, confidence: float, sample_path: str) -> None:
    ensure_runtime_dirs()
    record = {
        "predicted": predicted_label,
        "actual": actual_label,
        "confidence": float(confidence),
        "timestamp": utc_now(),
        "sample": sample_path,
    }
    with MISTAKE_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def merge_memory_to_dataset(
    labels: list[str] | None = None,
    include_corrections: bool = True,
    force: bool = False,
) -> list[Path]:
    """Copy memory examples into MP_Data so future model training uses them."""
    selected = {sanitize_action_name(label) for label in labels} if labels else None
    written: list[Path] = []
    memory = load_memory()

    for item in memory.get("examples", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        if selected is not None and label not in selected:
            continue
        if not include_corrections and item.get("source") == "correction":
            continue
        if item.get("merged_dataset_paths") and not force:
            continue

        rel_path = str(item.get("path", ""))
        memory_path = (MEMORY_FILE.parent / rel_path).resolve()
        if not memory_path.exists():
            continue
        sequence = np.load(memory_path).astype(np.float32)

        sample_dir = next_sample_dir(label, data_path=DATA_PATH)
        sample_dir.mkdir(parents=True, exist_ok=True)
        fitted = fit_sequence_length(sequence, target_length=SEQUENCE_LENGTH)
        for frame_num, frame in enumerate(fitted):
            np.save(sample_dir / f"{frame_num}.npy", frame.astype(np.float32))
        written.append(sample_dir)
        item.setdefault("merged_dataset_paths", []).append(str(sample_dir.relative_to(DATA_PATH)))

    if written:
        save_memory(memory)
    return written
