"""Optional metadata for static/dynamic sign handling."""

from __future__ import annotations

import json

from sign_ai.config import ACTIONS_METADATA_FILE
from sign_language_common import sanitize_action_name


VALID_SIGN_TYPES = {"static", "dynamic"}


def load_actions_metadata() -> dict:
    if not ACTIONS_METADATA_FILE.exists():
        return {}
    data = json.loads(ACTIONS_METADATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{ACTIONS_METADATA_FILE.name} must contain a JSON object.")
    return data


def save_actions_metadata(metadata: dict) -> None:
    ACTIONS_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def set_action_type(label: str, sign_type: str) -> None:
    safe_label = sanitize_action_name(label)
    sign_type = sign_type.strip().lower()
    if not safe_label:
        raise ValueError("Action label cannot be empty.")
    if sign_type not in VALID_SIGN_TYPES:
        raise ValueError(f"Sign type must be one of: {sorted(VALID_SIGN_TYPES)}")
    metadata = load_actions_metadata()
    metadata.setdefault(safe_label, {})
    metadata[safe_label]["type"] = sign_type
    save_actions_metadata(metadata)


def get_action_type(label: str, default: str = "dynamic") -> str:
    metadata = load_actions_metadata()
    value = metadata.get(label, {}).get("type", default)
    return value if value in VALID_SIGN_TYPES else default
