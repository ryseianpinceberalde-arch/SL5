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
