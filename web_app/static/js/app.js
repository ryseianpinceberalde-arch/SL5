"use strict";

const elements = {};
let currentState = {};
let toastTimer = null;

const byId = (id) => document.getElementById(id);
const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, Number(value) || 0));
const percent = (value) => `${(clamp(value) * 100).toFixed(1)}%`;
const labelText = (value) => String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function cacheElements() {
  [
    "sidebar", "menuToggle", "sidebarStatusDot", "sidebarStatusText", "systemPill", "systemStatus",
    "errorBanner", "errorText", "liveBadge", "cameraMessage", "cameraResolution", "cameraFps",
    "detectedSign", "predictionSource", "confidenceValue", "confidenceTrack", "confidenceBar",
    "predictionTime", "aiFps", "startButton", "stopButton", "clearControlButton", "translatedText",
    "wordCount", "clearButton", "copyButton", "speakButton", "historyCount", "historyBody",
    "modelName", "lstmStatus", "lstmDetail", "lstmDot", "detrStatus", "detrDetail", "detrDot",
    "mediapipeStatus", "mediapipeDot", "cameraStatus", "cameraDetail", "cameraDot",
    "recognitionStatus", "recognitionDetail", "recognitionDot", "memoryStatus", "memoryDetail",
    "memoryDot", "sequenceLength", "modelLabelCount", "connectionStatus", "toast"
  ].forEach((id) => { elements[id] = byId(id); });
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 2800);
}

function setHealth(dot, ready, warning = false) {
  dot.classList.toggle("ready", Boolean(ready));
  dot.classList.toggle("warning", !ready && Boolean(warning));
}

function updateSystemStatus(state) {
  const online = Boolean(state.system_online && state.running);
  elements.systemPill.classList.toggle("online", online);
  elements.systemPill.classList.toggle("offline", !online);
  elements.systemStatus.textContent = online ? "System online" : "System offline";
  elements.sidebarStatusDot.classList.toggle("online", online);
  elements.sidebarStatusText.textContent = online ? "Online" : "Offline";
  elements.liveBadge.classList.toggle("live", online && state.camera_connected);
  elements.liveBadge.classList.toggle("idle", !online || !state.camera_connected);
  elements.liveBadge.querySelector("span").textContent = online && state.camera_connected ? "Live" : "Idle";

  const cameraReady = Boolean(state.camera_connected);
  elements.cameraMessage.classList.toggle("hidden", cameraReady);
  elements.cameraMessage.querySelector("span").textContent = state.error || (state.running ? "Connecting to camera…" : "Recognition is stopped");
  elements.startButton.disabled = Boolean(state.running);
  elements.stopButton.disabled = !state.running;

  const error = String(state.error || "");
  elements.errorBanner.hidden = !error;
  elements.errorText.textContent = error;
}

function updateDetection(state) {
  const reserved = new Set(["", "IDLE", "TRANSITION", "UNKNOWN"]);
  const rawLabel = String(state.detected_sign || "");
  elements.detectedSign.textContent = reserved.has(rawLabel) ? (rawLabel === "UNKNOWN" ? "Unknown" : "Waiting") : labelText(rawLabel);
  elements.predictionSource.textContent = state.prediction_source && state.prediction_source !== "none" ? state.prediction_source : "—";
  const confidence = reserved.has(rawLabel) ? 0 : clamp(state.confidence);
  elements.confidenceValue.textContent = percent(confidence);
  elements.confidenceBar.style.width = percent(confidence);
  elements.confidenceTrack.setAttribute("aria-valuenow", String(Math.round(confidence * 100)));
  elements.predictionTime.textContent = `${Math.round(Number(state.prediction_time_ms) || 0)} ms`;
}

function updatePredictions(predictions = []) {
  const cards = document.querySelectorAll(".prediction-card");
  cards.forEach((card, index) => {
    const item = predictions[index];
    const confidence = item ? clamp(item.confidence) : 0;
    card.querySelector("[data-prediction-label]").textContent = item ? (item.display_label || labelText(item.label)) : "Waiting";
    card.querySelector("[data-prediction-confidence]").textContent = percent(confidence);
    card.querySelector("[data-prediction-bar]").style.width = percent(confidence);
  });
}

function updateSentence(state) {
  const words = Array.isArray(state.sentence) ? state.sentence : [];
  const text = String(state.translated_text || "").trim();
  elements.translatedText.textContent = text || "Recognized words will appear here.";
  elements.translatedText.classList.toggle("has-text", Boolean(text));
  elements.wordCount.textContent = `${words.length} ${words.length === 1 ? "word" : "words"}`;
  elements.copyButton.disabled = !text;
  elements.speakButton.disabled = !text;
}

function updateHistory(history = []) {
  elements.historyCount.textContent = `${history.length} ${history.length === 1 ? "event" : "events"}`;
  elements.historyBody.replaceChildren();
  if (!history.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "No recognized signs yet.";
    row.appendChild(cell);
    elements.historyBody.appendChild(row);
    return;
  }
  history.forEach((item) => {
    const row = document.createElement("tr");
    [item.time || "—", item.display_sign || labelText(item.sign), percent(item.confidence), item.source || "—"].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    elements.historyBody.appendChild(row);
  });
}

function updateModes(state) {
  const capabilities = new Map((state.available_modes || []).map((item) => [item.name, item]));
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const capability = capabilities.get(button.dataset.mode);
    button.disabled = !capability?.available;
    button.title = capability?.reason || "Mode unavailable";
    button.classList.toggle("active", button.dataset.mode === state.mode);
  });
  document.querySelectorAll("[data-sequence-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.sequenceMode === state.sequence_mode);
  });
}

function updateAiStatus(state) {
  const model = state.model || {};
  const detr = state.signdetr || {};
  const memory = state.memory || {};
  const camera = state.camera || {};

  elements.modelName.textContent = model.name || "Model unavailable";
  elements.lstmStatus.textContent = model.loaded ? "Loaded" : "Not loaded";
  elements.lstmDetail.textContent = model.loaded ? `${model.type || "LSTM"} · ${(model.labels || []).length} labels` : (model.error || "Model unavailable");
  setHealth(elements.lstmDot, model.loaded, !model.loaded);

  elements.detrStatus.textContent = detr.loaded ? "Loaded" : "Not available";
  elements.detrDetail.textContent = detr.loaded ? "Letter recognition ready" : "No implementation found";
  setHealth(elements.detrDot, detr.loaded, !detr.loaded);

  elements.mediapipeStatus.textContent = state.mediapipe_active ? "Active" : "Inactive";
  setHealth(elements.mediapipeDot, state.mediapipe_active);

  elements.cameraStatus.textContent = state.camera_connected ? "Connected" : "Disconnected";
  const width = camera.reported_width || camera.width || 0;
  const height = camera.reported_height || camera.height || 0;
  elements.cameraDetail.textContent = `Camera ${camera.index ?? "—"} · ${width} × ${height}`;
  elements.cameraResolution.textContent = `${width} × ${height}`;
  setHealth(elements.cameraDot, state.camera_connected, Boolean(state.error));

  elements.recognitionStatus.textContent = state.running ? "Running" : "Idle";
  elements.recognitionDetail.textContent = `${labelText(state.mode)} · ${labelText(state.sequence_mode)}`;
  setHealth(elements.recognitionDot, state.running);

  elements.memoryStatus.textContent = memory.loaded ? "Loaded" : "Not loaded";
  elements.memoryDetail.textContent = `${memory.examples || 0} examples`;
  setHealth(elements.memoryDot, memory.loaded, !memory.loaded);

  elements.sequenceLength.textContent = `${state.sequence_length ?? "—"} frames`;
  elements.modelLabelCount.textContent = `${(model.labels || []).length} labels`;
}

function applyState(nextState) {
  currentState = { ...currentState, ...nextState };
  updateSystemStatus(currentState);
  updateDetection(currentState);
  updatePredictions(currentState.top_predictions || []);
  updateSentence(currentState);
  updateHistory(currentState.history || []);
  updateModes(currentState);
  updateAiStatus(currentState);
  elements.cameraFps.textContent = (Number(currentState.fps) || 0).toFixed(1);
  elements.aiFps.textContent = `${(Number(currentState.ai_fps) || 0).toFixed(1)} FPS`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function runAction(button, url, body = null) {
  button.classList.add("loading");
  button.disabled = true;
  try {
    const payload = await fetchJson(url, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined
    });
    if (payload.state) applyState(payload.state);
    return payload;
  } catch (error) {
    showToast(error.message, true);
    throw error;
  } finally {
    button.classList.remove("loading");
    button.disabled = button === elements.startButton ? Boolean(currentState.running) : (button === elements.stopButton ? !currentState.running : false);
  }
}

function bindControls() {
  elements.startButton.addEventListener("click", async () => {
    try { await runAction(elements.startButton, "/api/start"); showToast("Recognition is starting"); } catch (_) { /* handled */ }
  });
  elements.stopButton.addEventListener("click", async () => {
    try { await runAction(elements.stopButton, "/api/stop"); showToast("Recognition stopped"); } catch (_) { /* handled */ }
  });
  [elements.clearButton, elements.clearControlButton].forEach((button) => {
    button.addEventListener("click", async () => {
      try { await runAction(button, "/api/clear"); showToast("Translated text cleared"); } catch (_) { /* handled */ }
    });
  });
  elements.copyButton.addEventListener("click", async () => {
    const text = String(currentState.translated_text || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast("Translation copied");
    } catch (_) {
      showToast("Clipboard permission was denied", true);
    }
  });
  elements.speakButton.addEventListener("click", () => {
    const text = String(currentState.translated_text || "").trim();
    if (!text || !("speechSynthesis" in window)) {
      showToast("Speech synthesis is unavailable", true);
      return;
    }
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  });

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", async () => {
      try { await runAction(button, "/api/mode", { mode: button.dataset.mode }); } catch (_) { /* handled */ }
    });
  });
  document.querySelectorAll("[data-sequence-mode]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await runAction(button, "/api/sequence-mode", { mode: button.dataset.sequenceMode });
        showToast(`${labelText(button.dataset.sequenceMode)} sequence mode selected`);
      } catch (_) { /* handled */ }
    });
  });
}

function bindNavigation() {
  elements.menuToggle.addEventListener("click", () => {
    const open = elements.sidebar.classList.toggle("open");
    elements.menuToggle.setAttribute("aria-expanded", String(open));
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((nav) => nav.classList.remove("active"));
      item.classList.add("active");
      elements.sidebar.classList.remove("open");
      elements.menuToggle.setAttribute("aria-expanded", "false");
    });
  });
  document.addEventListener("click", (event) => {
    if (window.innerWidth <= 780 && elements.sidebar.classList.contains("open") && !elements.sidebar.contains(event.target) && !elements.menuToggle.contains(event.target)) {
      elements.sidebar.classList.remove("open");
      elements.menuToggle.setAttribute("aria-expanded", "false");
    }
  });
}

async function refreshStatus() {
  try {
    const state = await fetchJson("/api/status");
    applyState(state);
    if (!window.io) elements.connectionStatus.textContent = "Connected via status polling";
  } catch (error) {
    elements.connectionStatus.textContent = "Backend disconnected";
    showToast("Cannot reach the recognition backend", true);
  }
}

function connectSocket() {
  if (!window.io) {
    elements.connectionStatus.textContent = "Live socket unavailable · using status polling";
    return;
  }
  const socket = window.io({ transports: ["websocket", "polling"] });
  socket.on("connect", () => { elements.connectionStatus.textContent = "Live backend connected"; });
  socket.on("disconnect", () => { elements.connectionStatus.textContent = "Live socket disconnected · polling active"; });
  socket.on("connect_error", () => { elements.connectionStatus.textContent = "Socket unavailable · polling active"; });
  socket.on("status_update", applyState);
  socket.on("prediction_update", (payload) => applyState(payload));
  socket.on("sentence_update", (payload) => applyState(payload));
  socket.on("fps_update", (payload) => applyState(payload));
  socket.on("mode_update", (payload) => applyState(payload));
}

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindControls();
  bindNavigation();
  refreshStatus();
  connectSocket();
  window.setInterval(refreshStatus, 1500);
});
