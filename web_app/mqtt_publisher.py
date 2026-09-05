"""Optional, failure-isolated MQTT output for recognition state.

MQTT is deliberately disabled by default.  This module does not subscribe to
commands and never handles camera frames or landmark arrays; it only publishes
small JSON summaries produced by the existing recognition bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4


LOGGER = logging.getLogger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_HISTORY_UNSET = object()
_MAX_PENDING_EVENTS = 100


def _is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _read_int(value: str | None, default: int, name: str, minimum: int, maximum: int) -> int:
    raw = str(value if value is not None else default).strip()
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _device_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "sign-ai-pc"


def _topic_prefix(value: str) -> str:
    cleaned = value.strip().strip("/")
    if not cleaned:
        raise ValueError("SIGN_AI_MQTT_TOPIC_PREFIX cannot be empty")
    if any(character in cleaned for character in ("+", "#", "\x00")):
        raise ValueError("SIGN_AI_MQTT_TOPIC_PREFIX cannot contain MQTT wildcards")
    return cleaned


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _reason_failed(reason_code: Any) -> bool:
    failure = getattr(reason_code, "is_failure", None)
    if failure is not None:
        return bool(failure)
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class MqttConfig:
    """Connection settings loaded from environment variables."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    username: str = ""
    password: str = ""
    use_tls: bool = False
    device_id: str = "sign-ai-pc"
    topic_prefix: str = "signai/v1"
    keepalive: int = 60

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "MqttConfig":
        env = os.environ if environ is None else environ
        enabled = _is_enabled(env.get("SIGN_AI_MQTT_ENABLED"))
        if not enabled:
            return cls()

        host = str(env.get("SIGN_AI_MQTT_HOST", "127.0.0.1")).strip()
        if not host:
            raise ValueError("SIGN_AI_MQTT_HOST cannot be empty")

        username = str(env.get("SIGN_AI_MQTT_USERNAME", "")).strip()
        password = str(env.get("SIGN_AI_MQTT_PASSWORD", ""))
        if password and not username:
            raise ValueError("SIGN_AI_MQTT_USERNAME is required when a password is set")

        return cls(
            enabled=True,
            host=host,
            port=_read_int(env.get("SIGN_AI_MQTT_PORT"), 1883, "SIGN_AI_MQTT_PORT", 1, 65535),
            username=username,
            password=password,
            use_tls=_is_enabled(env.get("SIGN_AI_MQTT_TLS")),
            device_id=_device_id(str(env.get("SIGN_AI_MQTT_DEVICE_ID", "sign-ai-pc"))),
            topic_prefix=_topic_prefix(str(env.get("SIGN_AI_MQTT_TOPIC_PREFIX", "signai/v1"))),
            keepalive=_read_int(
                env.get("SIGN_AI_MQTT_KEEPALIVE"),
                60,
                "SIGN_AI_MQTT_KEEPALIVE",
                5,
                3600,
            ),
        )


class MqttPublisher:
    """Publish current recognition state without affecting recognition flow."""

    def __init__(
        self,
        config: MqttConfig | None = None,
        client_factory: Callable[[MqttConfig], Any] | None = None,
        configuration_error: str = "",
    ):
        self.config = config or MqttConfig()
        self._client_factory = client_factory
        self._client = None
        self._lock = threading.RLock()
        self._started = False
        self._connected = False
        self._availability_announced = False
        self._connection_generation = 0
        self._last_error = configuration_error
        self._last_state_key: str | None = None
        self._last_history: object | list[str] = _HISTORY_UNSET
        self._pending_events: list[dict] = []

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        client_factory: Callable[[MqttConfig], Any] | None = None,
    ) -> "MqttPublisher":
        try:
            config = MqttConfig.from_env(environ)
        except (TypeError, ValueError) as exc:
            message = f"Invalid MQTT configuration; MQTT disabled: {exc}"
            LOGGER.warning(message)
            return cls(MqttConfig(), client_factory=client_factory, configuration_error=message)
        return cls(config, client_factory=client_factory)

    @property
    def base_topic(self) -> str:
        return f"{_topic_prefix(self.config.topic_prefix)}/{_device_id(self.config.device_id)}"

    def topic(self, suffix: str) -> str:
        cleaned = str(suffix).strip().strip("/")
        if not cleaned or any(character in cleaned for character in ("+", "#", "\x00")):
            raise ValueError("MQTT topic suffix must be non-empty and cannot contain wildcards")
        return f"{self.base_topic}/{cleaned}"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "started": self._started,
                "connected": self._connected,
                "host": self.config.host,
                "port": self.config.port,
                "device_id": self.config.device_id,
                "topic_prefix": self.config.topic_prefix,
                "tls": self.config.use_tls,
                "error": self._last_error,
            }

    def _set_error(self, message: str) -> None:
        safe_message = str(message)
        with self._lock:
            changed = safe_message != self._last_error
            self._last_error = safe_message
        if changed and safe_message:
            LOGGER.warning("MQTT: %s", safe_message)

    def _set_callback_error(self, client, message: str) -> bool:
        """Record a callback error only if it belongs to the active client."""
        safe_message = str(message)
        with self._lock:
            if not self._started or self._client is not client:
                return False
            self._connected = False
            self._availability_announced = False
            self._connection_generation += 1
            changed = safe_message != self._last_error
            self._last_error = safe_message
        if changed:
            LOGGER.warning("MQTT: %s", safe_message)
        return True

    def _create_client(self):
        if self._client_factory is not None:
            return self._client_factory(self.config)

        import paho.mqtt.client as mqtt

        return mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"sign-ai-{self.config.device_id}"[:128],
            protocol=mqtt.MQTTv311,
        )

    def start(self) -> bool:
        """Start the MQTT network thread; failures remain local to MQTT."""
        if not self.config.enabled:
            return False
        with self._lock:
            if self._started:
                return True

        client = None
        try:
            client = self._create_client()
            client.on_connect = self._on_connect
            client.on_connect_fail = self._on_connect_fail
            client.on_disconnect = self._on_disconnect
            client.reconnect_delay_set(min_delay=1, max_delay=30)
            client.max_queued_messages_set(100)
            if self.config.username:
                client.username_pw_set(self.config.username, self.config.password or None)
            if self.config.use_tls:
                client.tls_set()
            client.will_set(
                self.topic("availability"),
                self._json_payload({"online": False}, include_observed_at=False),
                qos=1,
                retain=True,
            )
            result = client.connect_async(self.config.host, self.config.port, self.config.keepalive)
            if _reason_failed(result):
                raise RuntimeError(f"connect_async returned {result}")
            with self._lock:
                self._client = client
                self._started = True
                self._connected = False
                self._availability_announced = False
                self._connection_generation += 1
            result = client.loop_start()
            if _reason_failed(result):
                raise RuntimeError(f"loop_start returned {result}")
            return True
        except Exception as exc:  # MQTT must never prevent the web app from starting.
            with self._lock:
                self._client = None
                self._started = False
                self._connected = False
                self._availability_announced = False
                self._connection_generation += 1
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass
                try:
                    client.loop_stop()
                except Exception:
                    pass
            self._set_error(f"Could not start MQTT: {exc}")
            return False

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        try:
            failed = _reason_failed(reason_code)
            failure_message = f"Broker rejected the MQTT connection: {reason_code}"
            with self._lock:
                if not self._started or self._client is not client:
                    return
                self._connected = not failed
                self._availability_announced = False
                self._connection_generation += 1
                if failed:
                    changed = failure_message != self._last_error
                    self._last_error = failure_message
                else:
                    changed = False
                    self._last_error = ""
                    self._last_state_key = None
            if failed:
                if changed:
                    LOGGER.warning("MQTT: %s", failure_message)
                return
            self._publish_online_if_needed()
        except Exception as exc:
            self._set_callback_error(client, f"MQTT connect callback failed: {exc}")

    def _on_connect_fail(self, client, _userdata) -> None:
        self._set_callback_error(
            client,
            f"Cannot reach MQTT broker at {self.config.host}:{self.config.port}",
        )

    def _on_disconnect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        if _reason_failed(reason_code):
            self._set_callback_error(client, f"MQTT connection lost: {reason_code}")
            return
        with self._lock:
            if not self._started or self._client is not client:
                return
            self._connected = False
            self._availability_announced = False
            self._connection_generation += 1

    def _json_payload(self, payload: Mapping[str, Any], *, include_observed_at: bool = True) -> str:
        message = {
            "schema_version": 1,
            "device_id": self.config.device_id,
        }
        if include_observed_at:
            message["observed_at"] = _utc_now()
        message.update(payload)
        return json.dumps(
            message,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def publish(self, suffix: str, payload: Mapping[str, Any], *, retain: bool) -> bool:
        """Publish JSON without blocking; return False on any MQTT failure."""
        if not self.config.enabled:
            return False
        with self._lock:
            client = self._client
            ready = self._started and self._connected and client is not None
        if not ready:
            return False

        try:
            info = client.publish(
                self.topic(suffix),
                self._json_payload(payload),
                qos=1,
                retain=retain,
            )
            if _reason_failed(getattr(info, "rc", 0)):
                self._set_error(f"MQTT publish returned {info.rc}")
                return False
            with self._lock:
                self._last_error = ""
            return True
        except Exception as exc:
            self._set_error(f"MQTT publish failed: {exc}")
            return False

    def _publish_online_if_needed(self) -> bool:
        with self._lock:
            needed = self._started and self._connected and not self._availability_announced
            client = self._client
            generation = self._connection_generation
        if not needed or not self.publish("availability", {"online": True}, retain=True):
            return False
        with self._lock:
            if (
                self._started
                and self._connected
                and self._client is client
                and self._connection_generation == generation
            ):
                self._availability_announced = True
        return True

    @staticmethod
    def _state_payload(state: Mapping[str, Any]) -> dict:
        model = state.get("model") if isinstance(state.get("model"), Mapping) else {}
        memory = state.get("memory") if isinstance(state.get("memory"), Mapping) else {}
        return {
            "running": bool(state.get("running")),
            "system_online": bool(state.get("system_online")),
            "camera_connected": bool(state.get("camera_connected")),
            "mediapipe_active": bool(state.get("mediapipe_active")),
            "recognition_state": str(state.get("recognition_state", "")),
            "detected_sign": str(state.get("detected_sign", "")),
            "confidence": round(_finite_float(state.get("confidence", 0.0)), 4),
            "prediction_source": str(state.get("prediction_source", "")),
            "sentence": [str(item) for item in state.get("sentence", [])],
            "translated_text": str(state.get("translated_text", "")),
            "model_loaded": bool(model.get("loaded")),
            "memory_loaded": bool(memory.get("loaded")),
            "has_error": bool(state.get("error")),
        }

    @staticmethod
    def _history_entries(state: Mapping[str, Any]) -> tuple[list[dict], list[str]]:
        entries: list[dict] = []
        keys: list[str] = []
        history = state.get("history", [])
        if not isinstance(history, list):
            return entries, keys
        for item in history:
            if not isinstance(item, Mapping):
                continue
            entry = {
                "time": str(item.get("time", "")),
                "sign": str(item.get("sign", "")),
                "display_sign": str(item.get("display_sign", item.get("sign", ""))),
                "confidence": round(_finite_float(item.get("confidence", 0.0)), 4),
                "source": str(item.get("source", "")),
            }
            entries.append(entry)
            keys.append(json.dumps(entry, sort_keys=True, separators=(",", ":")))
        return entries, keys

    def _new_history_entries(self, entries: list[dict], keys: list[str]) -> list[dict]:
        with self._lock:
            previous = self._last_history
            self._last_history = list(keys)

        if previous is _HISTORY_UNSET or keys == previous or not keys:
            return []
        if not previous:
            return list(reversed(entries))

        previous_keys = list(previous)
        for added_count in range(1, len(keys)):
            remaining = keys[added_count:]
            if len(remaining) <= len(previous_keys) and remaining == previous_keys[: len(remaining)]:
                return list(reversed(entries[:added_count]))

        # History was replaced rather than extended. Publish only its newest item
        # to avoid replaying old actuator events.
        return [entries[0]]

    def publish_state(self, state: Mapping[str, Any]) -> bool:
        """Publish a deduplicated state and newly accepted sign events."""
        if not self.config.enabled:
            return False

        published = self._publish_online_if_needed()
        state_payload = self._state_payload(state)
        state_key = json.dumps(state_payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        entries, history_keys = self._history_entries(state)
        new_entries = self._new_history_entries(entries, history_keys)
        with self._lock:
            connected = self._started and self._connected and self._client is not None
            if connected:
                for entry in new_entries:
                    event = dict(entry)
                    event["event_id"] = uuid4().hex
                    self._pending_events.append(event)
                if len(self._pending_events) > _MAX_PENDING_EVENTS:
                    overflow = len(self._pending_events) - _MAX_PENDING_EVENTS
                    del self._pending_events[:overflow]
                    LOGGER.warning("MQTT event queue full; dropped %d oldest event(s)", overflow)

        with self._lock:
            state_changed = state_key != self._last_state_key
        if state_changed and self.publish("state", state_payload, retain=True):
            with self._lock:
                self._last_state_key = state_key
            published = True

        while True:
            with self._lock:
                if not self._pending_events:
                    break
                event = dict(self._pending_events[0])
            if not self.publish("events/sign", event, retain=False):
                break
            with self._lock:
                if self._pending_events and self._pending_events[0].get("event_id") == event["event_id"]:
                    self._pending_events.pop(0)
            published = True
        return published

    def stop(self) -> bool:
        """Publish offline availability and stop once; safe to call repeatedly."""
        with self._lock:
            if not self._started or self._client is None:
                return False
            client = self._client
            connected = self._connected
            self._started = False
            self._connected = False
            self._availability_announced = False
            self._connection_generation += 1
            self._client = None
            self._pending_events.clear()

        if connected:
            try:
                info = client.publish(
                    self.topic("availability"),
                    self._json_payload({"online": False}),
                    qos=1,
                    retain=True,
                )
                wait_for_publish = getattr(info, "wait_for_publish", None)
                if callable(wait_for_publish):
                    wait_for_publish(timeout=1.0)
                is_published = getattr(info, "is_published", None)
                if _reason_failed(getattr(info, "rc", 0)):
                    self._set_error(f"MQTT shutdown publish returned {info.rc}")
                elif callable(is_published) and not is_published():
                    self._set_error("MQTT shutdown status was not confirmed before timeout")
            except Exception as exc:
                self._set_error(f"Could not publish MQTT shutdown status: {exc}")
        try:
            client.disconnect()
        except Exception as exc:
            self._set_error(f"Could not disconnect MQTT: {exc}")
        try:
            client.loop_stop()
        except Exception as exc:
            self._set_error(f"Could not stop MQTT network loop: {exc}")
        return True
