"""Local Flask-SocketIO server for the Sign AI browser dashboard."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, Response, jsonify, render_template, request
    from flask_socketio import SocketIO, emit
except ImportError as exc:
    raise SystemExit(
        "Missing web dependency. Run: python -m pip install -r web_app/requirements.txt"
    ) from exc

from web_app.recognition_bridge import RecognitionBridge, UnsupportedModeError
from web_app.mqtt_publisher import MqttPublisher


socketio = SocketIO(async_mode="threading")
LOGGER = logging.getLogger(__name__)


def _prediction_payload(state: dict) -> dict:
    return {
        "detected_sign": state["detected_sign"],
        "confidence": state["confidence"],
        "prediction_source": state["prediction_source"],
        "prediction_reason": state["prediction_reason"],
        "prediction_time_ms": state["prediction_time_ms"],
        "top_predictions": state["top_predictions"],
        "memory_matches": state["memory_matches"],
    }


def _sentence_payload(state: dict) -> dict:
    return {
        "sentence": state["sentence"],
        "translated_text": state["translated_text"],
        "history": state["history"],
    }


def _fps_payload(state: dict) -> dict:
    return {
        "fps": state["fps"],
        "ai_fps": state["ai_fps"],
        "prediction_time_ms": state["prediction_time_ms"],
    }


def _mode_payload(state: dict) -> dict:
    return {
        "mode": state["mode"],
        "available_modes": state["available_modes"],
        "sequence_mode": state["sequence_mode"],
        "available_sequence_modes": state["available_sequence_modes"],
    }


def create_app(
    bridge: RecognitionBridge | None = None,
    test_config: dict | None = None,
    mqtt_publisher: MqttPublisher | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(SECRET_KEY=os.getenv("SIGN_AI_WEB_SECRET", "sign-ai-local-dashboard"))
    if test_config:
        app.config.update(test_config)

    active_bridge = bridge or RecognitionBridge(load_models=True)
    app.extensions["recognition_bridge"] = active_bridge
    app.extensions["mqtt_publisher"] = (
        mqtt_publisher if mqtt_publisher is not None else MqttPublisher.from_env()
    )
    app.extensions["publisher_stop"] = threading.Event()
    socketio.init_app(app)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(
            active_bridge.mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/status")
    def api_status():
        return jsonify(active_bridge.snapshot())

    @app.get("/api/mqtt")
    def api_mqtt():
        return jsonify(app.extensions["mqtt_publisher"].snapshot())

    @app.post("/api/start")
    def api_start():
        started = active_bridge.start()
        return jsonify({"ok": True, "started": started, "state": active_bridge.snapshot()}), (202 if started else 200)

    @app.post("/api/stop")
    def api_stop():
        stopped = active_bridge.stop()
        return jsonify({"ok": True, "stopped": stopped, "state": active_bridge.snapshot()})

    @app.post("/api/clear")
    def api_clear():
        active_bridge.clear_sentence()
        return jsonify({"ok": True, "state": active_bridge.snapshot()})

    @app.post("/api/mode")
    def api_mode():
        payload = request.get_json(silent=True) or {}
        try:
            mode = active_bridge.set_interface_mode(payload.get("mode", ""))
        except UnsupportedModeError as exc:
            return jsonify({"ok": False, "error": str(exc), "state": active_bridge.snapshot()}), 409
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "state": active_bridge.snapshot()}), 400
        return jsonify({"ok": True, "mode": mode, "state": active_bridge.snapshot()})

    @app.post("/api/sequence-mode")
    def api_sequence_mode():
        payload = request.get_json(silent=True) or {}
        try:
            mode = active_bridge.set_sequence_mode(payload.get("mode", ""))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "state": active_bridge.snapshot()}), 400
        return jsonify({"ok": True, "sequence_mode": mode, "state": active_bridge.snapshot()})

    @socketio.on("connect")
    def socket_connect(auth=None):
        del auth
        state = active_bridge.snapshot()
        emit("status_update", state)
        emit("prediction_update", _prediction_payload(state))
        emit("sentence_update", _sentence_payload(state))
        emit("fps_update", _fps_payload(state))
        emit("mode_update", _mode_payload(state))

    return app


def start_status_publisher(app: Flask):
    """Broadcast changed recognition state without sending camera frame bytes."""
    bridge: RecognitionBridge = app.extensions["recognition_bridge"]
    mqtt_publisher: MqttPublisher = app.extensions["mqtt_publisher"]
    stop_event: threading.Event = app.extensions["publisher_stop"]

    def publish_loop():
        previous_prediction = None
        previous_sentence = None
        previous_fps = None
        previous_mode = None
        previous_status = None
        last_full_status = 0.0
        mqtt_error_logged = False

        while not stop_event.is_set():
            state = bridge.snapshot()
            try:
                mqtt_publisher.publish_state(state)
                mqtt_error_logged = False
            except Exception as exc:  # A third-party MQTT error must not stop Socket.IO.
                if not mqtt_error_logged:
                    LOGGER.warning("MQTT state publishing failed: %s", exc)
                    mqtt_error_logged = True
            prediction = _prediction_payload(state)
            sentence = _sentence_payload(state)
            fps = _fps_payload(state)
            mode = _mode_payload(state)
            status = {
                "running": state["running"],
                "system_online": state["system_online"],
                "camera_connected": state["camera_connected"],
                "mediapipe_active": state["mediapipe_active"],
                "recognition_state": state["recognition_state"],
                "model": state["model"],
                "memory": state["memory"],
                "signdetr": state["signdetr"],
                "error": state["error"],
            }

            if prediction != previous_prediction:
                socketio.emit("prediction_update", prediction)
                previous_prediction = prediction
            if sentence != previous_sentence:
                socketio.emit("sentence_update", sentence)
                previous_sentence = sentence
            if fps != previous_fps:
                socketio.emit("fps_update", fps)
                previous_fps = fps
            if mode != previous_mode:
                socketio.emit("mode_update", mode)
                previous_mode = mode

            now = time.monotonic()
            if status != previous_status or now - last_full_status >= 1.0:
                socketio.emit("status_update", state)
                previous_status = status
                last_full_status = now
            socketio.sleep(0.2)

    return socketio.start_background_task(publish_loop)


def print_startup_status(state: dict) -> None:
    def ready(value: bool) -> str:
        return "READY" if value else "NOT AVAILABLE"

    print("=" * 45)
    print("SIGN AI WEB DASHBOARD")
    print("=" * 45)
    print(f"LSTM       : {ready(state['model']['loaded'])}")
    print(f"SignDETR   : {ready(state['signdetr']['loaded'])}")
    print(f"MediaPipe  : {ready(state['mediapipe_active'])}")
    print(f"Camera     : {ready(state['camera_connected'])}")
    if state["error"]:
        print(f"Status     : {state['error']}")
    print("\nWeb Dashboard:")
    print("http://127.0.0.1:5000")
    print("=" * 45)


def main() -> None:
    bridge = RecognitionBridge(load_models=True)
    mqtt_publisher = MqttPublisher.from_env()
    app = create_app(bridge=bridge, mqtt_publisher=mqtt_publisher)
    publisher_task = None
    try:
        mqtt_publisher.start()
        publisher_task = start_status_publisher(app)

        auto_start = os.getenv("SIGN_AI_WEB_AUTOSTART", "1").strip().lower() not in {"0", "false", "no"}
        if auto_start:
            bridge.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                state = bridge.snapshot()
                if state["camera_connected"] or state["error"]:
                    break
                time.sleep(0.05)

        print_startup_status(bridge.snapshot())
      socketio.run(
    app,
    host="0.0.0.0",
    port=int(os.getenv("PORT", "10000")),
    debug=False,
    use_reloader=False,
    allow_unsafe_werkzeug=True,
)
        )
    finally:
        app.extensions["publisher_stop"].set()
        join = getattr(publisher_task, "join", None)
        if callable(join):
            try:
                join(timeout=1.0)
            except Exception as exc:
                LOGGER.warning("Could not join status publisher during shutdown: %s", exc)
        try:
            bridge.stop()
        finally:
            mqtt_publisher.stop()


if __name__ == "__main__":
    main()
