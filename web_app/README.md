# Sign AI Web Dashboard

This folder is an add-on around the existing optimized recognition runtime. It
does not change training data, model files, labels, thresholds, camera settings,
or the original Python scripts.

## Install

From the project root:

```powershell
python -m pip install -r web_app/requirements.txt
```

## Run

```powershell
python web_app/app.py
```

Open <http://127.0.0.1:5000/api/mqtt> to check the connection state. This
status response never includes the MQTT password.

Open <http://127.0.0.1:5000>.

The web process is the single camera owner. Do not run an original realtime
camera script at the same time.

Set `SIGN_AI_WEB_AUTOSTART=0` before launching if the server should start with
recognition stopped. The dashboard Start button can then open the camera.

## Current capabilities

- WORDS uses the existing LSTM model and DTW memory fallback.
- FIXED and MOTION use the existing sequence-mode behavior.
- HYBRID and LETTERS are visible but disabled because this inspected project
  contains no SignDETR implementation, letter adapter, or checkpoint.
- History is stored in memory for the current server session only.
- Socket.IO updates use the JavaScript client CDN. If it cannot load, the
  dashboard automatically falls back to `/api/status` polling.

## Optional MQTT output

MQTT is disabled by default and does not change camera recognition, models,
memory, HTTP routes, or Socket.IO. When enabled, it publishes small JSON status
messages and accepted signs. It never publishes camera frames or landmarks and
does not subscribe to remote commands.

Install and run an MQTT broker separately. For a local Windows setup, Eclipse
Mosquitto is a common choice. Then enable MQTT before starting the dashboard:

```powershell
$env:SIGN_AI_MQTT_ENABLED="1"
$env:SIGN_AI_MQTT_HOST="127.0.0.1"
$env:SIGN_AI_MQTT_PORT="1883"
$env:SIGN_AI_MQTT_DEVICE_ID="sign-ai-pc"
python web_app/app.py
```

If the broker requires credentials, also set `SIGN_AI_MQTT_USERNAME` and
`SIGN_AI_MQTT_PASSWORD`. Set `SIGN_AI_MQTT_TLS=1` for a TLS-enabled broker
(normally port 8883). Credentials are read only from environment variables.

Published topics use the following defaults:

- `signai/v1/sign-ai-pc/availability` - retained online/offline state
- `signai/v1/sign-ai-pc/state` - retained recognition summary
- `signai/v1/sign-ai-pc/events/sign` - each newly accepted sign

The sign-event topic is live-only: signs accepted while the broker is
disconnected are not replayed later. The retained state still provides the
latest recognized sentence after reconnection.

To watch all messages with Mosquitto's command-line subscriber:

```powershell
mosquitto_sub -h 127.0.0.1 -t "signai/v1/#" -v
```

The optional settings `SIGN_AI_MQTT_TOPIC_PREFIX` and
`SIGN_AI_MQTT_KEEPALIVE` can change the topic root and keepalive interval. If
the package is missing, configuration is invalid, or the broker is offline,
the web dashboard and recognition continue running normally.

Port 1883 is unencrypted and should be limited to this computer or a trusted
local network. For any untrusted network, use broker authentication, topic
access controls, and TLS with certificate verification (normally port 8883).
Add the matching username, password, and TLS options to `mosquitto_sub` when
testing a secured broker.
