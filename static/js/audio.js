/**
 * Browser voice feedback built on window.speechSynthesis.
 * Fully client side - nothing is sent to the server.
 */
const VoiceFeedback = (function () {
  let enabled = false;
  let volume = 1.0;
  let rate = 1.0;
  let lastSpoken = "";
  let lastSpokenAt = 0;
  const MIN_INTERVAL_MS = 2500;

  function supported() {
    return "speechSynthesis" in window;
  }

  function speak(text, force = false) {
    if (!enabled || !supported() || !text) return;

    const now = Date.now();
    // Do not repeat the same sentence over and over within a short window
    if (!force && text === lastSpoken && now - lastSpokenAt < MIN_INTERVAL_MS) return;

    lastSpoken = text;
    lastSpokenAt = now;

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.volume = volume;
      utterance.rate = rate;
      utterance.pitch = 1;
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn("Speech synthesis failed:", err);
    }
  }

  function setEnabled(value) {
    enabled = Boolean(value);
    if (!enabled && supported()) window.speechSynthesis.cancel();
  }

  function setVolume(value) { volume = Math.min(Math.max(parseFloat(value) || 0, 0), 1); }
  function setRate(value) { rate = Math.min(Math.max(parseFloat(value) || 1, 0.5), 2); }

  function isEnabled() { return enabled; }

  return { supported, speak, setEnabled, setVolume, setRate, isEnabled };
})();