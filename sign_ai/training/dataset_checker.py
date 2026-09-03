"""Dataset health checks for MP_Data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sign_ai.config import DATA_PATH, KEYPOINT_LENGTH, SEQUENCE_LENGTH
from sign_language_dataset import find_sample_dirs


@dataclass
class SampleIssue:
    path: str
    issue: str


@dataclass
class SignHealth:
    label: str
    samples: int = 0
    valid_samples: int = 0
    frame_counts: list[int] = field(default_factory=list)
    issues: list[SampleIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.valid_samples < 10:
            return "INSUFFICIENT"
        if self.valid_samples < 30 or self.issues:
            return "WARNING"
        return "OK"


@dataclass
class DatasetHealthReport:
    signs: list[SignHealth]
    duplicate_groups: list[list[str]]
    recommendations: list[str]

    @property
    def total_valid_samples(self) -> int:
        return sum(sign.valid_samples for sign in self.signs)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(DATA_PATH.resolve()).as_posix()


def _frame_paths(sample_dir: Path) -> list[Path]:
    return sorted(
        [path for path in sample_dir.glob("*.npy") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )


def _sample_hash(frames: list[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for frame in frames:
        hasher.update(np.ascontiguousarray(frame).tobytes())
    return hasher.hexdigest()


def check_sample(sample_dir: Path) -> tuple[bool, int, list[str], str | None]:
    frame_paths = _frame_paths(sample_dir)
    issues: list[str] = []
    frames: list[np.ndarray] = []

    if not frame_paths:
        return False, 0, ["no numeric .npy frames"], None

    stems = [int(path.stem) for path in frame_paths]
    expected = list(range(min(stems), max(stems) + 1))
    if stems != expected:
        issues.append("missing frame numbers")

    for frame_path in frame_paths:
        try:
            frame = np.load(frame_path)
        except Exception as exc:
            issues.append(f"{frame_path.name} cannot be loaded: {exc}")
            continue
        if frame.shape != (KEYPOINT_LENGTH,):
            issues.append(f"{frame_path.name} has shape {frame.shape}, expected ({KEYPOINT_LENGTH},)")
            continue
        if not np.all(np.isfinite(frame)):
            issues.append(f"{frame_path.name} contains NaN or infinite values")
        if not np.any(frame):
            issues.append(f"{frame_path.name} is all zeros")
        frames.append(frame.astype(np.float32))

    if len(frame_paths) != SEQUENCE_LENGTH:
        issues.append(f"frame count {len(frame_paths)} will be resampled/padded to {SEQUENCE_LENGTH}")

    valid = bool(frames) and not any("cannot be loaded" in issue or "shape" in issue or "NaN" in issue for issue in issues)
    digest = _sample_hash(frames) if valid else None
    return valid, len(frame_paths), issues, digest


def check_dataset(data_path: Path = DATA_PATH, include_legacy_batches: bool = True) -> DatasetHealthReport:
    data_path.mkdir(parents=True, exist_ok=True)
    signs: list[SignHealth] = []
    hashes: dict[str, list[str]] = {}

    for action_dir in sorted([item for item in data_path.iterdir() if item.is_dir()], key=lambda item: item.name):
        health = SignHealth(label=action_dir.name)
        for sample_dir in find_sample_dirs(action_dir, include_legacy_batches=include_legacy_batches):
            health.samples += 1
            valid, frame_count, issues, digest = check_sample(sample_dir)
            health.frame_counts.append(frame_count)
            if valid:
                health.valid_samples += 1
            for issue in issues:
                health.issues.append(SampleIssue(_rel(sample_dir), issue))
            if digest:
                hashes.setdefault(digest, []).append(_rel(sample_dir))
        signs.append(health)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    recommendations = build_recommendations(signs, duplicate_groups)
    return DatasetHealthReport(signs=signs, duplicate_groups=duplicate_groups, recommendations=recommendations)


def build_recommendations(signs: list[SignHealth], duplicate_groups: list[list[str]]) -> list[str]:
    recommendations: list[str] = []
    if not signs:
        return ["No sign folders found. Run the folder setup and collection scripts first."]

    valid_counts = [sign.valid_samples for sign in signs]
    max_count = max(valid_counts) if valid_counts else 0
    for sign in signs:
        if sign.valid_samples < 10:
            recommendations.append(f"Collect more samples for {sign.label}; less than 10 is not enough.")
        elif sign.valid_samples < 30:
            recommendations.append(f"Collect more samples for {sign.label}; 30+ is a better minimum.")
        if max_count and sign.valid_samples < max_count * 0.5:
            recommendations.append(f"{sign.label} has much less data than the largest class.")
    if duplicate_groups:
        recommendations.append("Duplicate samples were detected; review them before training.")
    return recommendations


def print_report(report: DatasetHealthReport) -> None:
    print("Dataset Health Report")
    print("=====================")
    if not report.signs:
        print("No signs found.")
    for sign in report.signs:
        print(f"{sign.label:<20} {sign.valid_samples:>4}/{sign.samples:<4} samples  {sign.status}")
        if sign.frame_counts:
            print(f"  frames: min={min(sign.frame_counts)} max={max(sign.frame_counts)} expected={SEQUENCE_LENGTH}")
        for issue in sign.issues[:8]:
            print(f"  - {issue.path}: {issue.issue}")
        if len(sign.issues) > 8:
            print(f"  - ... {len(sign.issues) - 8} more issues")

    if report.duplicate_groups:
        print("\nDuplicate groups:")
        for group in report.duplicate_groups[:5]:
            print("  - " + ", ".join(group))

    print("\nRecommendations:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
    print(f"\nTotal valid samples: {report.total_valid_samples}")

