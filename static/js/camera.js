/**
 * Thin wrapper around navigator.mediaDevices.getUserMedia().
 * Works on desktop browsers and on Android / iOS Safari.
 */
class CameraController {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.facingMode = "user"; // "user" = front, "environment" = rear
    this.deviceIndex = 0;
  }

  get isActive() {
    return Boolean(this.stream) && this.stream.getVideoTracks().some(t => t.readyState === "live");
  }

  async start(facingMode) {
    if (facingMode) this.facingMode = facingMode;
    this.stop();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Your browser does not support camera access. " +
        "Use a modern browser and make sure the page is served over http://localhost or https://."
      );
    }

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "SecurityError") {
        throw new Error("Camera permission was denied. Allow camera access in your browser settings.");
      }
      if (err.name === "NotFoundError" || err.name === "OverconstrainedError") {
        throw new Error("No compatible camera was found on this device.");
      }
      if (err.name === "NotReadableError") {
        throw new Error("The camera is already in use by another application.");
      }
      throw new Error(`Camera error: ${err.message}`);
    }

    this.video.srcObject = this.stream;
    await this.video.play().catch(() => {});
    return this.stream;
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
  }

  async switchCamera() {
    // Flip between front and rear camera
    this.facingMode = this.facingMode === "user" ? "environment" : "user";
    await this.start(this.facingMode);
    return this.facingMode;
  }

  /** Grab the current frame as a JPEG data URL. */
  captureDataUrl(maxWidth = 640, quality = 0.85) {
    if (!this.isActive) throw new Error("The camera is not running.");

    const canvas = document.createElement("canvas");
    const scale = Math.min(1, maxWidth / (this.video.videoWidth || maxWidth));
    canvas.width = Math.round((this.video.videoWidth || 640) * scale);
    canvas.height = Math.round((this.video.videoHeight || 480) * scale);

    const ctx = canvas.getContext("2d");
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  }
}