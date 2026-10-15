/**
 * Wires the live-camera page: capture -> /api/predict/ -> UI + voice.
 */
document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("video");
  if (!video) return;

  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnPredict = document.getElementById("btnPredict");
  const btnSwitch = document.getElementById("btnSwitch");
  const autoPredict = document.getElementById("autoPredict");
  const statusBadge = document.getElementById("cameraStatus");
  const errorBox = document.getElementById("cameraError");
  const overlay = document.getElementById("videoOverlay");

  const activityLabel = document.getElementById("activityLabel");
  const confidenceBar = document.getElementById("confidenceBar");
  const processingTime = document.getElementById("processingTime");
  const modelName = document.getElementById("modelName");
  const uncertainBadge = document.getElementById("uncertainBadge");

  const voiceEnabled = document.getElementById("voiceEnabled");
  const voiceVolume = document.getElementById("voiceVolume");
  const voiceRate = document.getElementById("voiceRate");

  const camera = new CameraController(video);
  let autoTimer = null;
  let busy = false;

  // ---- Voice controls --------------------------------------------------
  voiceEnabled.addEventListener("change", () => {
    VoiceFeedback.setEnabled(voiceEnabled.checked);
    if (voiceEnabled.checked) VoiceFeedback.speak("Voice feedback enabled.", true);
  });
  voiceVolume.addEventListener("input", () => VoiceFeedback.setVolume(voiceVolume.value));
  voiceRate.addEventListener("input", () => VoiceFeedback.setRate(voiceRate.value));
  VoiceFeedback.setVolume(voiceVolume.value);
  VoiceFeedback.setRate(voiceRate.value);

  // ---- Helpers ---------------------------------------------------------
  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("d-none");
  }
  function clearError() {
    errorBox.classList.add("d-none");
    errorBox.textContent = "";
  }
  function setStatus(text, cls) {
    statusBadge.textContent = text;
    statusBadge.className = `badge ${cls}`;
  }

  function renderResult(data) {
    activityLabel.textContent = data.activity.replace(/_/g, " ").toUpperCase();

    const pct = Math.round((data.confidence || 0) * 100);
    confidenceBar.style.width = `${pct}%`;
    confidenceBar.textContent = `${pct}%`;
    confidenceBar.className =
      "progress-bar " + (data.uncertain ? "bg-warning" : pct >= 80 ? "bg-success" : "bg-info");

    processingTime.textContent = `${Math.round(data.processing_time_ms)} ms`;
    modelName.textContent = data.model || "—";

    uncertainBadge.classList.toggle("d-none", !data.uncertain);

    if (data.speech) VoiceFeedback.speak(data.speech);
  }

  // ---- Actions ---------------------------------------------------------
  async function startCamera() {
    clearError();
    overlay.classList.remove("d-none");
    try {
      await camera.start();
      setStatus("Running", "bg-success");
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnPredict.disabled = false;
      btnSwitch.disabled = false;
    } catch (err) {
      showError(err.message);
      setStatus("Error", "bg-danger");
    } finally {
      overlay.classList.add("d-none");
    }
  }

  function stopCamera() {
    camera.stop();
    stopAuto();
    setStatus("Stopped", "bg-secondary");
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnPredict.disabled = true;
    btnSwitch.disabled = true;
  }

  async function predictOnce() {
    if (busy || !camera.isActive) return;
    busy = true;
    btnPredict.disabled = true;
    try {
      const dataUrl = camera.captureDataUrl();
      const result = await window.apiFetch("/api/predict/", {
        method: "POST",
        body: JSON.stringify({ image: dataUrl, source: "camera" }),
      });
      renderResult(result);
      clearError();
    } catch (err) {
      showError(`Prediction failed: ${err.message}`);
    } finally {
      busy = false;
      btnPredict.disabled = false;
    }
  }

  function startAuto() {
    stopAuto();
    autoTimer = setInterval(predictOnce, 2000);
  }
  function stopAuto() {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = null;
  }

  // ---- Event listeners -------------------------------------------------
  btnStart.addEventListener("click", startCamera);
  btnStop.addEventListener("click", stopCamera);
  btnPredict.addEventListener("click", predictOnce);
  btnSwitch.addEventListener("click", async () => {
    try {
      const mode = await camera.switchCamera();
      VoiceFeedback.speak(mode === "user" ? "Front camera." : "Rear camera.", true);
    } catch (err) {
      showError(err.message);
    }
  });

  autoPredict.addEventListener("change", () => {
    if (autoPredict.checked && camera.isActive) startAuto();
    else stopAuto();
  });

  // Restart auto-prediction once the camera is running
  btnStart.addEventListener("click", () => {
    setTimeout(() => { if (autoPredict.checked && camera.isActive) startAuto(); }, 500);
  });

  // Clean up when leaving the page
  window.addEventListener("beforeunload", () => {
    stopAuto();
    camera.stop();
  });
});