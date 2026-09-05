from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import Mock, patch

from web_app.app import create_app, main, socketio, start_status_publisher
from web_app.mqtt_publisher import MqttConfig, MqttPublisher


def mqtt_state() -> dict:
    return {
        "running": True,
        "system_online": True,
        "camera_connected": True,
        "mediapipe_active": True,
        "recognition_state": "RECOGNIZING",
        "mode": "WORDS",
        "available_modes": [
            {"name": "WORDS", "available": True, "reason": "Available"},
        ],
        "sequence_mode": "FIXED",
        "available_sequence_modes": ["FIXED", "MOTION"],
        "detected_sign": "hello",
        "confidence": 0.91234,
        "prediction_source": "model",
        "prediction_reason": "accepted",
        "prediction_time_ms": 12.5,
        "top_predictions": [],
        "memory_matches": [],
        "fps": 30.0,
        "ai_fps": 6.0,
        "sentence": [],
        "translated_text": "",
        "history": [],
        "model": {"loaded": True, "path": "private/local/model.keras"},
        "memory": {"loaded": True},
        "signdetr": {"loaded": False},
        "error": "",
    }


class FakeMessageInfo:
    def __init__(self, rc: int = 0, published: bool = True, call_log: list | None = None):
        self.rc = rc
        self.published = published
        self.call_log = call_log
        self.wait_timeouts: list[float] = []

    def wait_for_publish(self, timeout: float | None = None):
        self.wait_timeouts.append(timeout)
        if self.call_log is not None:
            self.call_log.append(("wait_for_publish", timeout))

    def is_published(self):
        return self.published


class FakeClient:
    def __init__(self, *, connect_error: Exception | None = None):
        self.connect_error = connect_error
        self.publish_rc = 0
        self.publish_confirmed = True
        self.publish_error: Exception | None = None
        self.publish_hook = None
        self.on_connect = None
        self.on_connect_fail = None
        self.on_disconnect = None
        self.reconnect_delays = None
        self.max_queued = None
        self.credentials = None
        self.tls_calls = 0
        self.will = None
        self.connect_args = None
        self.loop_start_calls = 0
        self.loop_stop_calls = 0
        self.disconnect_calls = 0
        self.published: list[dict] = []
        self.calls: list[tuple] = []

    def reconnect_delay_set(self, min_delay, max_delay):
        self.reconnect_delays = (min_delay, max_delay)

    def max_queued_messages_set(self, count):
        self.max_queued = count

    def username_pw_set(self, username, password):
        self.credentials = (username, password)

    def tls_set(self):
        self.tls_calls += 1

    def will_set(self, topic, payload, qos, retain):
        self.will = {"topic": topic, "payload": payload, "qos": qos, "retain": retain}

    def connect_async(self, host, port, keepalive):
        if self.connect_error:
            raise self.connect_error
        self.connect_args = (host, port, keepalive)
        return 0

    def loop_start(self):
        self.loop_start_calls += 1
        return 0

    def publish(self, topic, payload, qos, retain):
        if self.publish_error:
            raise self.publish_error
        result_code = self.publish_rc
        hook = self.publish_hook
        self.publish_hook = None
        if hook is not None:
            hook()
        self.calls.append(("publish", topic))
        info = FakeMessageInfo(result_code, self.publish_confirmed, self.calls)
        self.published.append(
            {"topic": topic, "payload": payload, "qos": qos, "retain": retain, "info": info}
        )
        return info

    def disconnect(self):
        self.calls.append(("disconnect",))
        self.disconnect_calls += 1
        return 0

    def loop_stop(self):
        self.calls.append(("loop_stop",))
        self.loop_stop_calls += 1
        return 0


class MqttPublisherTests(unittest.TestCase):
    def setUp(self):
        self.config = MqttConfig(enabled=True, device_id="test-device")
        self.client = FakeClient()
        self.publisher = MqttPublisher(self.config, client_factory=lambda _config: self.client)

    def connect(self):
        self.assertTrue(self.publisher.start())
        self.client.on_connect(self.client, None, None, 0, None)
        self.assertTrue(self.publisher.snapshot()["connected"])

    def test_disabled_publisher_is_a_noop(self):
        calls = []
        publisher = MqttPublisher(MqttConfig(), client_factory=lambda config: calls.append(config))
        self.assertFalse(publisher.start())
        self.assertFalse(publisher.publish_state(mqtt_state()))
        self.assertFalse(publisher.stop())
        self.assertEqual(calls, [])
        self.assertEqual(
            {key: publisher.snapshot()[key] for key in ("enabled", "started", "connected", "error")},
            {"enabled": False, "started": False, "connected": False, "error": ""},
        )

    def test_invalid_configuration_and_missing_dependency_are_non_fatal(self):
        invalid = MqttPublisher.from_env(
            {"SIGN_AI_MQTT_ENABLED": "1", "SIGN_AI_MQTT_PORT": "invalid"}
        )
        self.assertFalse(invalid.snapshot()["enabled"])
        self.assertIn("must be an integer", invalid.snapshot()["error"])

        missing = MqttPublisher(
            self.config,
            client_factory=lambda _config: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'paho'")),
        )
        self.assertFalse(missing.start())
        self.assertIn("paho", missing.snapshot()["error"])
        self.assertFalse(missing.publish_state(mqtt_state()))

    def test_start_applies_connection_security_and_queue_settings(self):
        config = MqttConfig(
            enabled=True,
            host="broker.local",
            port=8883,
            username="signai",
            password="secret",
            use_tls=True,
            device_id="test-device",
            keepalive=45,
        )
        publisher = MqttPublisher(config, client_factory=lambda _config: self.client)
        self.assertTrue(publisher.start())
        self.assertEqual(self.client.connect_args, ("broker.local", 8883, 45))
        self.assertEqual(self.client.credentials, ("signai", "secret"))
        self.assertEqual(self.client.tls_calls, 1)
        self.assertEqual(self.client.reconnect_delays, (1, 30))
        self.assertEqual(self.client.max_queued, 100)
        self.assertEqual(self.client.will["topic"], "signai/v1/test-device/availability")
        self.assertTrue(self.client.will["retain"])
        self.assertNotIn("observed_at", json.loads(self.client.will["payload"]))

    def test_broker_start_and_publish_failures_do_not_escape(self):
        failing_client = FakeClient(connect_error=OSError("connection refused"))
        publisher = MqttPublisher(self.config, client_factory=lambda _config: failing_client)
        self.assertFalse(publisher.start())
        self.assertEqual(failing_client.loop_start_calls, 0)
        self.assertIn("connection refused", publisher.snapshot()["error"])

        self.connect()
        self.client.publish_error = OSError("network down")
        self.assertFalse(self.publisher.publish_state(mqtt_state()))
        self.assertIn("network down", self.publisher.snapshot()["error"])

    def test_async_connection_failure_is_non_fatal(self):
        self.assertTrue(self.publisher.start())
        self.client.on_connect_fail(self.client, None)
        self.assertFalse(self.publisher.snapshot()["connected"])
        self.assertIn("Cannot reach MQTT broker", self.publisher.snapshot()["error"])
        self.assertFalse(self.publisher.publish_state(mqtt_state()))

    def test_failed_online_availability_is_retried_before_state(self):
        self.assertTrue(self.publisher.start())
        self.client.publish_rc = 4
        self.client.on_connect(self.client, None, None, 0, None)
        self.client.publish_rc = 0

        self.assertTrue(self.publisher.publish_state(mqtt_state()))
        availability = [
            item for item in self.client.published if item["topic"].endswith("/availability")
        ]
        self.assertEqual([item["info"].rc for item in availability], [4, 0])
        successful_topics = [
            item["topic"] for item in self.client.published if item["info"].rc == 0
        ]
        self.assertLess(
            successful_topics.index("signai/v1/test-device/availability"),
            successful_topics.index("signai/v1/test-device/state"),
        )

    def test_reconnect_during_availability_publish_cannot_suppress_retry(self):
        self.assertTrue(self.publisher.start())

        def reconnect_during_publish():
            self.client.on_disconnect(self.client, None, None, 1, None)
            self.client.publish_rc = 4
            self.client.on_connect(self.client, None, None, 0, None)
            self.client.publish_rc = 0

        self.client.publish_hook = reconnect_during_publish
        self.client.on_connect(self.client, None, None, 0, None)
        self.assertTrue(self.publisher.snapshot()["connected"])
        self.assertTrue(self.publisher.publish_state(mqtt_state()))

        availability = [
            item for item in self.client.published if item["topic"].endswith("/availability")
        ]
        self.assertEqual([item["info"].rc for item in availability], [4, 0, 0])
        self.assertEqual(self.client.published[-1]["topic"], "signai/v1/test-device/state")

    def test_state_is_deduplicated_and_events_are_oldest_first(self):
        self.connect()
        self.client.published.clear()
        state = mqtt_state()

        self.assertTrue(self.publisher.publish_state(state))
        self.assertFalse(self.publisher.publish_state(state))
        state["fps"] = 1.0
        state["ai_fps"] = 1.0
        state["prediction_time_ms"] = 999.0
        self.assertFalse(self.publisher.publish_state(state))

        first = {"time": "10:00:00", "sign": "hello", "display_sign": "Hello", "confidence": 0.9, "source": "model"}
        second = {"time": "10:00:01", "sign": "thanks", "display_sign": "Thanks", "confidence": 0.8, "source": "memory"}
        state["history"] = [second, first]
        state["sentence"] = ["hello", "thanks"]
        state["translated_text"] = "Hello Thanks"
        self.assertTrue(self.publisher.publish_state(state))

        events = [item for item in self.client.published if item["topic"].endswith("/events/sign")]
        self.assertEqual([json.loads(item["payload"])["sign"] for item in events], ["hello", "thanks"])
        self.assertTrue(all(item["topic"] == "signai/v1/test-device/events/sign" for item in events))
        self.assertTrue(all(item["qos"] == 1 and not item["retain"] for item in events))
        first_payload = json.loads(events[0]["payload"])
        self.assertEqual(
            {key: first_payload[key] for key in ("time", "sign", "display_sign", "confidence", "source")},
            first,
        )
        self.assertEqual(first_payload["schema_version"], 1)
        self.assertEqual(first_payload["device_id"], "test-device")
        self.assertNotEqual(
            first_payload["event_id"],
            json.loads(events[1]["payload"])["event_id"],
        )

        states = [item for item in self.client.published if item["topic"].endswith("/state")]
        payload = json.loads(states[-1]["payload"])
        self.assertEqual(payload["device_id"], "test-device")
        self.assertNotIn("fps", payload)
        self.assertNotIn("prediction_time_ms", payload)
        self.assertNotIn("path", payload)
        self.assertNotIn("frame", payload)

    def test_failed_state_publish_is_retried(self):
        self.connect()
        self.client.published.clear()
        self.client.publish_rc = 4
        state = mqtt_state()
        self.assertFalse(self.publisher.publish_state(state))
        self.client.publish_rc = 0
        self.assertTrue(self.publisher.publish_state(state))
        states = [item for item in self.client.published if item["topic"].endswith("/state")]
        self.assertEqual(len(states), 2)

    def test_failed_event_publish_retries_with_the_same_event_id(self):
        self.connect()
        state = mqtt_state()
        self.assertTrue(self.publisher.publish_state(state))
        self.client.published.clear()

        state["history"] = [
            {"time": "10:00:00", "sign": "hello", "display_sign": "Hello", "confidence": 0.9, "source": "model"}
        ]
        self.client.publish_rc = 4
        self.assertFalse(self.publisher.publish_state(state))
        self.client.publish_rc = 0
        self.assertTrue(self.publisher.publish_state(state))

        attempts = [item for item in self.client.published if item["topic"].endswith("/events/sign")]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            json.loads(attempts[0]["payload"])["event_id"],
            json.loads(attempts[1]["payload"])["event_id"],
        )

    def test_replaced_history_publishes_only_the_newest_item(self):
        self.connect()
        state = mqtt_state()
        state["history"] = [
            {"time": "10:00:01", "sign": "a", "confidence": 0.9},
            {"time": "10:00:00", "sign": "b", "confidence": 0.8},
        ]
        self.publisher.publish_state(state)
        self.client.published.clear()

        state["history"] = [
            {"time": "11:00:01", "sign": "x", "confidence": 0.7},
            {"time": "11:00:00", "sign": "y", "confidence": 0.6},
        ]
        self.assertTrue(self.publisher.publish_state(state))
        events = [item for item in self.client.published if item["topic"].endswith("/events/sign")]
        self.assertEqual([json.loads(item["payload"])["sign"] for item in events], ["x"])

    def test_non_finite_confidence_is_valid_json(self):
        self.connect()
        state = mqtt_state()
        state["confidence"] = float("nan")
        self.assertTrue(self.publisher.publish_state(state))
        state_message = next(item for item in reversed(self.client.published) if item["topic"].endswith("/state"))
        self.assertEqual(json.loads(state_message["payload"])["confidence"], 0.0)

    def test_stale_callbacks_cannot_change_a_restarted_publisher(self):
        first_client = self.client
        second_client = FakeClient()
        clients = iter((first_client, second_client))
        publisher = MqttPublisher(self.config, client_factory=lambda _config: next(clients))
        self.assertTrue(publisher.start())
        self.assertTrue(publisher.stop())
        self.assertTrue(publisher.start())
        second_client.on_connect(second_client, None, None, 0, None)
        self.assertTrue(publisher.snapshot()["connected"])

        first_client.on_connect(first_client, None, None, 0, None)
        first_client.on_disconnect(first_client, None, None, 1, None)
        first_client.on_connect_fail(first_client, None)
        self.assertTrue(publisher.snapshot()["connected"])

    def test_stop_racing_with_connect_callback_cannot_restore_connected_state(self):
        self.assertTrue(self.publisher.start())
        reason_started = threading.Event()
        allow_reason_to_finish = threading.Event()

        def blocking_reason(_reason_code):
            reason_started.set()
            self.assertTrue(allow_reason_to_finish.wait(timeout=2))
            return False

        with patch("web_app.mqtt_publisher._reason_failed", side_effect=blocking_reason):
            callback = threading.Thread(
                target=self.client.on_connect,
                args=(self.client, None, None, 0, None),
            )
            callback.start()
            self.assertTrue(reason_started.wait(timeout=2))
            self.assertTrue(self.publisher.stop())
            allow_reason_to_finish.set()
            callback.join(timeout=2)

        self.assertFalse(callback.is_alive())
        self.assertFalse(self.publisher.snapshot()["started"])
        self.assertFalse(self.publisher.snapshot()["connected"])

    def test_events_accepted_while_disconnected_are_not_replayed(self):
        self.connect()
        state = mqtt_state()
        self.publisher.publish_state(state)
        self.client.published.clear()
        self.client.on_disconnect(self.client, None, None, 1, None)

        state["history"] = [
            {"time": "10:00:00", "sign": "hello", "confidence": 0.9, "source": "model"}
        ]
        self.assertFalse(self.publisher.publish_state(state))
        self.client.on_connect(self.client, None, None, 0, None)
        self.publisher.publish_state(state)
        events = [item for item in self.client.published if item["topic"].endswith("/events/sign")]
        self.assertEqual(events, [])

    def test_stop_is_idempotent_and_publishes_offline_first(self):
        self.connect()
        self.client.published.clear()
        self.client.calls.clear()
        self.assertTrue(self.publisher.stop())
        self.assertFalse(self.publisher.stop())
        self.assertEqual(self.client.disconnect_calls, 1)
        self.assertEqual(self.client.loop_stop_calls, 1)
        self.assertEqual(self.client.published[-1]["topic"], "signai/v1/test-device/availability")
        self.assertFalse(json.loads(self.client.published[-1]["payload"])["online"])
        self.assertTrue(self.client.published[-1]["retain"])
        self.assertEqual(
            self.client.calls,
            [
                ("publish", "signai/v1/test-device/availability"),
                ("wait_for_publish", 1.0),
                ("disconnect",),
                ("loop_stop",),
            ],
        )
        self.assertFalse(self.publisher.snapshot()["started"])

    def test_shutdown_publish_timeout_is_reported_but_still_stops(self):
        self.connect()
        self.client.publish_confirmed = False
        self.assertTrue(self.publisher.stop())
        self.assertIn("not confirmed", self.publisher.snapshot()["error"])
        self.assertEqual(self.client.disconnect_calls, 1)
        self.assertEqual(self.client.loop_stop_calls, 1)


class LoopBridge:
    def snapshot(self):
        return mqtt_state()


class RaisingMqttPublisher:
    def __init__(self):
        self.calls = 0
        self.start_calls = 0

    def start(self):
        self.start_calls += 1

    def snapshot(self):
        return {"enabled": True, "started": False, "connected": False, "error": "test"}

    def publish_state(self, _state):
        self.calls += 1
        raise OSError("test MQTT failure")


class FalseyMqttPublisher(RaisingMqttPublisher):
    def __bool__(self):
        return False


class MqttWebIntegrationTests(unittest.TestCase):
    def test_app_factory_injects_without_starting_mqtt(self):
        mqtt = FalseyMqttPublisher()
        app = create_app(LoopBridge(), {"TESTING": True, "SECRET_KEY": "test"}, mqtt_publisher=mqtt)
        self.assertIs(app.extensions["mqtt_publisher"], mqtt)
        self.assertEqual(mqtt.start_calls, 0)
        self.assertEqual(app.test_client().get("/api/status").status_code, 200)
        mqtt_response = app.test_client().get("/api/mqtt")
        self.assertEqual(mqtt_response.status_code, 200)
        self.assertNotIn("password", mqtt_response.get_json())

    def test_mqtt_exception_does_not_stop_socketio_updates(self):
        mqtt = RaisingMqttPublisher()
        app = create_app(LoopBridge(), {"TESTING": True, "SECRET_KEY": "test"}, mqtt_publisher=mqtt)
        stop_event = app.extensions["publisher_stop"]

        with (
            patch.object(socketio, "start_background_task", side_effect=lambda target: target),
            patch.object(socketio, "emit") as emit,
            patch.object(socketio, "sleep", side_effect=lambda _seconds: stop_event.set()),
        ):
            publish_loop = start_status_publisher(app)
            publish_loop()

        emitted_names = {call.args[0] for call in emit.call_args_list}
        self.assertEqual(
            emitted_names,
            {"prediction_update", "sentence_update", "fps_update", "mode_update", "status_update"},
        )
        self.assertEqual(mqtt.calls, 1)

    def test_main_cleans_up_if_status_publisher_cannot_start(self):
        bridge = Mock()
        mqtt = Mock()
        with (
            patch("web_app.app.RecognitionBridge", return_value=bridge),
            patch("web_app.app.MqttPublisher.from_env", return_value=mqtt),
            patch("web_app.app.start_status_publisher", side_effect=RuntimeError("publisher failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "publisher failed"):
                main()

        mqtt.start.assert_called_once_with()
        bridge.stop.assert_called_once_with()
        mqtt.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
