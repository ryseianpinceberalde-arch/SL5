from __future__ import annotations

import time
import unittest

from web_app.app import create_app, socketio
from web_app.recognition_bridge import RecognitionBridge, UnsupportedModeError


def sample_state() -> dict:
    return {
        "running": False,
        "system_online": False,
        "camera_connected": False,
        "mediapipe_active": False,
        "recognition_state": "IDLE",
        "mode": "WORDS",
        "available_modes": [
            {"name": "HYBRID", "available": False, "reason": "Unavailable"},
            {"name": "LETTERS", "available": False, "reason": "Unavailable"},
            {"name": "WORDS", "available": True, "reason": "Available"},
        ],
        "sequence_mode": "FIXED",
        "available_sequence_modes": ["FIXED", "MOTION"],
        "fps": 0.0,
        "ai_fps": 0.0,
        "prediction_time_ms": 0.0,
        "detected_sign": "IDLE",
        "confidence": 0.0,
        "prediction_source": "none",
        "prediction_reason": "test",
        "top_predictions": [],
        "memory_matches": [],
        "sentence": [],
        "translated_text": "",
        "history": [],
        "sequence_length": 30,
        "model": {"loaded": True, "status": "Loaded", "name": "test", "type": "LSTM", "path": "", "labels": ["hello"], "error": ""},
        "memory": {"loaded": True, "status": "Loaded", "examples": 1, "error": ""},
        "signdetr": {"loaded": False, "status": "Not available", "error": "Unavailable"},
        "camera": {"index": 1, "width": 1280, "height": 720, "target_fps": 30},
        "error": "",
        "updated_at": 0.0,
    }


class StubBridge:
    def __init__(self):
        self.state = sample_state()

    def snapshot(self):
        return dict(self.state)

    def start(self):
        changed = not self.state["running"]
        self.state["running"] = True
        self.state["system_online"] = True
        return changed

    def stop(self):
        changed = self.state["running"]
        self.state["running"] = False
        self.state["system_online"] = False
        return changed

    def clear_sentence(self):
        self.state["sentence"] = []
        self.state["translated_text"] = ""

    def set_interface_mode(self, mode):
        requested = str(mode).upper()
        if requested not in {"HYBRID", "LETTERS", "WORDS"}:
            raise ValueError("Unknown mode")
        if requested != "WORDS":
            raise UnsupportedModeError("Mode unavailable")
        self.state["mode"] = requested
        return requested

    def set_sequence_mode(self, mode):
        requested = str(mode).upper()
        if requested not in {"FIXED", "MOTION"}:
            raise ValueError("Invalid sequence mode")
        self.state["sequence_mode"] = requested
        return requested

    def mjpeg_stream(self):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ntest\r\n"


class WebApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = StubBridge()
        cls.app = create_app(cls.bridge, {"TESTING": True, "SECRET_KEY": "test"})
        cls.client = cls.app.test_client()

    def setUp(self):
        self.bridge.state = sample_state()

    def test_template_and_static_assets_load(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Real-Time Sign Language Recognition", response.data)
        response.close()
        css_response = self.client.get("/static/css/style.css")
        js_response = self.client.get("/static/js/app.js")
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(js_response.status_code, 200)
        css_response.close()
        js_response.close()

    def test_status_and_control_endpoints(self):
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertEqual(self.client.post("/api/start").status_code, 202)
        self.assertTrue(self.client.get("/api/status").get_json()["running"])
        self.assertEqual(self.client.post("/api/clear").status_code, 200)
        self.assertEqual(self.client.post("/api/sequence-mode", json={"mode": "MOTION"}).status_code, 200)
        self.assertEqual(self.client.post("/api/stop").status_code, 200)

    def test_unavailable_and_invalid_modes_are_rejected(self):
        self.assertEqual(self.client.post("/api/mode", json={"mode": "LETTERS"}).status_code, 409)
        self.assertEqual(self.client.post("/api/mode", json={"mode": "missing"}).status_code, 400)

    def test_video_feed_is_multipart(self):
        response = self.client.get("/video_feed", buffered=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response.content_type)
        self.assertTrue(next(response.response).startswith(b"--frame"))
        response.close()

    def test_socketio_client_receives_initial_state(self):
        socket_client = socketio.test_client(self.app, flask_test_client=self.client)
        self.assertTrue(socket_client.is_connected())
        events = {item["name"] for item in socket_client.get_received()}
        self.assertTrue({"status_update", "prediction_update", "sentence_update", "fps_update", "mode_update"}.issubset(events))
        socket_client.disconnect()


class RecognitionBridgeTests(unittest.TestCase):
    def test_bridge_state_and_modes_without_opening_camera(self):
        bridge = RecognitionBridge(load_models=False)
        self.assertEqual(bridge.snapshot()["sequence_length"], 30)
        self.assertEqual(bridge.set_interface_mode("words"), "WORDS")
        with self.assertRaises(UnsupportedModeError):
            bridge.set_interface_mode("letters")
        self.assertEqual(bridge.set_sequence_mode("motion"), "MOTION")

    def test_camera_frame_error_clears_after_recovery(self):
        bridge = RecognitionBridge(load_models=False)
        bridge._update_camera_frame_health(frame_received=False)
        self.assertEqual(bridge.snapshot()["error"], "Camera frame not received.")
        self.assertFalse(bridge.snapshot()["camera_connected"])
        bridge._update_camera_frame_health(frame_received=True)
        self.assertEqual(bridge.snapshot()["error"], "")
        self.assertTrue(bridge.snapshot()["camera_connected"])

    def test_camera_owner_is_exclusive(self):
        first = RecognitionBridge(load_models=False)
        second = RecognitionBridge(load_models=False)
        try:
            self.assertTrue(first._claim_camera())
            self.assertTrue(second.start())
            deadline = time.monotonic() + 2.0
            state = second.snapshot()
            while state["running"] and time.monotonic() < deadline:
                time.sleep(0.01)
                state = second.snapshot()
            self.assertIn("already owned", state["error"].lower())
            self.assertFalse(state["camera_connected"])
        finally:
            second.stop()
            first._release_camera()
            second._release_camera()


if __name__ == "__main__":
    unittest.main()
