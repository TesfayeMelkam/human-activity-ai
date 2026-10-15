document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("imageInput");
  if (!input) return;

  const dropZone = document.getElementById("dropZone");
  const preview = document.getElementById("preview");
  const btn = document.getElementById("btnUploadPredict");
  const errorBox = document.getElementById("uploadError");
  const resultBox = document.getElementById("uploadResult");
  const activityEl = document.getElementById("uploadActivity");
  const bar = document.getElementById("uploadConfidenceBar");
  const timeEl = document.getElementById("uploadTime");

  let file = null;

  function setFile(selected) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      showError("Please choose an image file.");
      return;
    }
    if (selected.size > 8 * 1024 * 1024) {
      showError("The image is larger than 8 MB.");
      return;
    }
    file = selected;
    clearError();
    btn.disabled = false;
    preview.src = URL.createObjectURL(selected);
    preview.classList.remove("d-none");
    resultBox.classList.add("d-none");
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("d-none");
  }
  function clearError() {
    errorBox.classList.add("d-none");
  }

  input.addEventListener("change", () => setFile(input.files[0]));

  dropZone.addEventListener("click", () => input.click());
  ["dragenter", "dragover"].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove("dragover"); })
  );
  dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  btn.addEventListener("click", async () => {
    if (!file) return;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analysing...';

    const formData = new FormData();
    formData.append("image", file);

    try {
      const data = await window.apiFetch("/api/upload/", { method: "POST", body: formData });

      activityEl.textContent = data.activity.replace(/_/g, " ").toUpperCase();
      const pct = Math.round((data.confidence || 0) * 100);
      bar.style.width = `${pct}%`;
      bar.textContent = `${pct}%`;
      bar.className = "progress-bar " + (data.uncertain ? "bg-warning" : "bg-success");
      timeEl.textContent = `${Math.round(data.processing_time_ms)} ms`;
      resultBox.classList.remove("d-none");
      clearError();

      if (data.speech) VoiceFeedback.speak(data.speech, true);
    } catch (err) {
      showError(err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="bi bi-magic"></i> Predict activity';
    }
  });
});