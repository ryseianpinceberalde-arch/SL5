"""Versioned model registry and compatibility model pointers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sign_ai.config import (
    BEST_MODEL_PATH,
    LEGACY_MODEL_PATH,
    MODEL_ACTIONS_FILE,
    MODEL_PATH,
    MODEL_REGISTRY_FILE,
    MODELS_DIR,
)


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    model_path: Path
    actions_path: Path
    results_dir: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_REGISTRY_FILE.exists():
        return {"models": {}, "best_model": None, "active_model": None}
    registry = json.loads(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    registry.setdefault("models", {})
    registry.setdefault("best_model", None)
    registry.setdefault("active_model", None)
    return registry


def save_registry(registry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def next_model_version() -> ModelVersion:
    registry = load_registry()
    existing_numbers = []
    for version_id in registry.get("models", {}):
        suffix = version_id.removeprefix("model_v")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    version_id = f"model_v{next_number:03d}"
    return ModelVersion(
        version_id=version_id,
        model_path=MODELS_DIR / f"{version_id}.keras",
        actions_path=MODELS_DIR / f"{version_id}_actions.json",
        results_dir=MODELS_DIR / version_id,
    )


def register_model(
    version: ModelVersion,
    actions: list[str],
    metadata: dict,
    make_active: bool = True,
    make_best: bool = True,
) -> None:
    registry = load_registry()
    registry["models"][version.version_id] = {
        "created": utc_now(),
        "model_path": str(version.model_path),
        "actions_path": str(version.actions_path),
        "results_dir": str(version.results_dir),
        "classes": len(actions),
        "actions": actions,
        **metadata,
    }
    if make_best:
        registry["best_model"] = version.version_id
    if make_active:
        registry["active_model"] = version.version_id
    save_registry(registry)


def write_actions_file(path: Path, actions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(actions, indent=2), encoding="utf-8")


def update_compatibility_files(model_path: Path, actions: list[str], save_legacy_h5: bool = True) -> None:
    shutil.copy2(model_path, MODEL_PATH)
    shutil.copy2(model_path, BEST_MODEL_PATH)
    write_actions_file(MODEL_ACTIONS_FILE, actions)
    if save_legacy_h5:
        try:
            from tensorflow.keras.models import load_model

            model = load_model(model_path)
            model.save(LEGACY_MODEL_PATH)
        except Exception as exc:
            print(f"Could not save legacy h5 compatibility model: {exc}")


def activate_model(version_id: str) -> dict:
    registry = load_registry()
    info = registry.get("models", {}).get(version_id)
    if not info:
        raise ValueError(f"Unknown model version: {version_id}")
    model_path = Path(info["model_path"])
    actions_path = Path(info["actions_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not actions_path.exists():
        raise FileNotFoundError(actions_path)
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    update_compatibility_files(model_path, actions, save_legacy_h5=False)
    registry["active_model"] = version_id
    save_registry(registry)
    return info

