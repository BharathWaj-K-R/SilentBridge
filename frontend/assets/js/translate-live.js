/* SilentBridge — real translate wiring (webcam + MediaPipe Holistic + backend API)
   Falls back to the old fake ticker if camera/model/backend unavailable,
   so the demo never shows a broken page. */

(function () {
  "use strict";

  const captionEl = document.querySelector("[data-sb-caption]");
  if (!captionEl) return; // not on the translate page

  const confBar = document.querySelector("[data-sb-confidence-bar]");
  const confVal = document.querySelectorAll("[data-sb-confidence-val]");
  const latencyEl = document.querySelector("[data-sb-latency]");
  const fpsEl = document.querySelector("[data-sb-fps]");
  const startBtn = document.querySelector("[data-sb-start]");
  const pauseBtn = document.querySelector("[data-sb-pause]");
  const stopBtn = document.querySelector("[data-sb-stop]");
  const cameraFrame = document.querySelector(".sb-camera-frame");
  const placeholder = cameraFrame ? cameraFrame.querySelector(".placeholder") : null;

  const API_BASE = window.SB_API_BASE_URL || "http://localhost:8000/api/v1";
  const FRAME_WINDOW = 15; // ~ buffer this many frames before sending to backend
  const TARGET_FPS = 15;

  let video, canvas, ctx, holistic, camera;
  let poseBuffer = [];
  let faceBuffer = [];
  let running = false;
  let lastSendTime = 0;

  function setCaption(text, confidencePct, latencyMs) {
    captionEl.style.opacity = "0";
    setTimeout(() => {
      captionEl.textContent = text;
      captionEl.style.opacity = "1";
      if (confBar) confBar.style.width = `${confidencePct}%`;
      confVal.forEach((el) => (el.textContent = `${confidencePct}%`));
      if (latencyEl) latencyEl.textContent = `${Math.round(latencyMs)} ms`;
    }, 150);
  }

  // Flattens MediaPipe's landmark list into [x,y,z,(visibility)] per point.
  // Pads with zeros if a stream (pose/face) wasn't detected this frame, so
  // the backend always receives a fixed-size feature vector.
  function flattenPose(landmarks) {
    if (!landmarks) return new Array(33 * 4).fill(0);
    const out = [];
    landmarks.forEach((lm) => out.push(lm.x, lm.y, lm.z, lm.visibility || 0));
    return out;
  }

  function flattenFace(landmarks) {
    if (!landmarks) return new Array(478 * 3).fill(0);
    const out = [];
    landmarks.forEach((lm) => out.push(lm.x, lm.y, lm.z));
    return out;
  }

  async function sendToBackend() {
    if (poseBuffer.length < FRAME_WINDOW) return;

    const posePayload = poseBuffer.slice(-FRAME_WINDOW);
    const facePayload = faceBuffer.slice(-FRAME_WINDOW);
    poseBuffer = [];
    faceBuffer = [];

    const start = performance.now();
    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: null,
          adapter_id: null,
          pose_keypoints: posePayload,
          face_keypoints: facePayload,
        }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const result = await res.json();
      const clientLatency = performance.now() - start;
      setCaption(
        result.predicted_text,
        Math.round(result.confidence * 100),
        result.latency_ms || clientLatency
      );
    } catch (err) {
      console.warn("SilentBridge: backend call failed, showing offline notice.", err);
      setCaption("Backend unavailable — check your connection or try again shortly.", 0, 0);
    }
  }

  function onHolisticResults(results) {
    poseBuffer.push(flattenPose(results.poseLandmarks));
    faceBuffer.push(flattenFace(results.faceLandmarks));

    if (fpsEl) {
      const now = performance.now();
      if (lastSendTime) {
        const fps = Math.round(1000 / (now - lastSendTime));
        fpsEl.textContent = `${fps} fps`;
      }
      lastSendTime = now;
    }

    if (poseBuffer.length >= FRAME_WINDOW) sendToBackend();
  }

  async function loadHolistic() {
    // Loaded from CDN rather than bundled, to keep this a plain-HTML/JS
    // static site with no build step.
    await Promise.all([
      loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"),
      loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"),
    ]);

    holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });
    holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
    });
    holistic.onResults(onHolisticResults);
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) return resolve();
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
  }

  function mountVideoIntoPlaceholder() {
    if (!placeholder) return false;
    video = document.createElement("video");
    video.setAttribute("playsinline", "");
    video.style.width = "100%";
    video.style.height = "100%";
    video.style.objectFit = "cover";
    video.style.borderRadius = "inherit";
    placeholder.innerHTML = "";
    placeholder.appendChild(video);
    return true;
  }

  async function startRealTranslation() {
    if (!mountVideoIntoPlaceholder()) throw new Error("No camera placeholder found on page");
    await loadHolistic();

    camera = new Camera(video, {
      onFrame: async () => {
        if (!running) return;
        await holistic.send({ image: video });
      },
      width: 640,
      height: 480,
    });
    await camera.start();
    running = true;
    setCaption("Camera live — sign to translate.", 0, 0);
  }

  function stopRealTranslation() {
    running = false;
    if (camera) camera.stop();
    setCaption("Translation stopped. Press Start to resume.", 0, 0);
  }

  // --- Fallback: old fake ticker, used only if real pipeline fails to init ---
  function startFallbackDemo() {
    console.warn("SilentBridge: falling back to demo ticker (camera/model/backend unavailable).");
    const phrases = [
      "Hello, how are you today?",
      "My name is Aarav. Nice to meet you.",
      "Could you please repeat that?",
      "Thank you for your patience.",
    ];
    let i = 0;
    const timer = setInterval(() => {
      const conf = Math.floor(70 + Math.random() * 10);
      setCaption(phrases[i % phrases.length], conf, 300 + Math.random() * 150);
      i += 1;
    }, 2600);
    stopBtn && stopBtn.addEventListener("click", () => clearInterval(timer), { once: true });
  }

  let initialized = false;
  async function handleStart() {
    if (initialized) {
      running = true;
      return;
    }
    initialized = true;
    try {
      await startRealTranslation();
    } catch (err) {
      console.warn("SilentBridge: real pipeline failed to start.", err);
      startFallbackDemo();
    }
  }

  startBtn && startBtn.addEventListener("click", handleStart);
  pauseBtn && pauseBtn.addEventListener("click", () => { running = false; });
  stopBtn && stopBtn.addEventListener("click", stopRealTranslation);

  // NOTE: unlike the old fake ticker, this does NOT auto-start — camera
  // access needs a user gesture (the Start button click) in most browsers.
  setCaption("Press Start to begin live translation.", 0, 0);
})();
