/* SilentBridge — real translate wiring (webcam + MediaPipe Holistic + backend API)
   FIXED VERSION: Properly pads face and pose keypoints to expected dimensions
   Falls back to the old fake ticker if camera/model/backend unavailable. */

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
  const FRAME_WINDOW = 50; // buffer this many frames before sending to backend
  const TARGET_FPS = 15;

  // Expected dimensions for the model — MUST match backend/app/models/base_model.py's
  // POSE_INPUT_DIM / FACE_INPUT_DIM exactly, which are themselves derived from legacy
  // mp.solutions.holistic (no iris refinement): 33 pose landmarks, 468 face landmarks.
  const EXPECTED_POSE_DIM = 33 * 4;   // 132 (33 landmarks × 4 coords: x,y,z,visibility)
  const EXPECTED_FACE_DIM = 468 * 3;  // 1404 (468 landmarks × 3 coords: x,y,z)
  // NEVER 478 * 3 (1434) here — that landmark count only applies when MediaPipe's
  // refineFaceLandmarks (iris refinement) is enabled, which this page does not use
  // (see holistic.setOptions below: refineFaceLandmarks: false). Sending 1434-dim
  // face vectors to a model trained on 1404-dim features silently misaligns every
  // feature after landmark 468 and produces garbage predictions.

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

  // FIXED: Flattens MediaPipe's pose landmarks into a fixed-size vector
  // Always returns exactly 132 dimensions, padding with zeros if needed
  function flattenPose(landmarks) {
    const out = [];
    
    if (!landmarks || landmarks.length === 0) {
      // No pose detected → return all zeros
      return new Array(EXPECTED_POSE_DIM).fill(0);
    }
    
    // Flatten all detected landmarks: (x, y, z, visibility)
    landmarks.forEach((lm) => {
      out.push(
        lm.x || 0,
        lm.y || 0,
        lm.z || 0,
        lm.visibility || 0
      );
    });
    
    // Pad with zeros if we got fewer landmarks than expected
    while (out.length < EXPECTED_POSE_DIM) {
      out.push(0);
    }
    
    // Truncate if somehow we got more (shouldn't happen, but defensive)
    return out.slice(0, EXPECTED_POSE_DIM);
  }

  // Flattens MediaPipe's face landmarks into a fixed-size vector.
  // Always returns exactly 1404 dimensions (EXPECTED_FACE_DIM), padding with
  // zeros if needed — this must stay in lockstep with base_model.py's
  // FACE_INPUT_DIM (468 landmarks × 3 coords), never 478 × 3 (1434).
  function flattenFace(landmarks) {
    const out = [];
    
    if (!landmarks || landmarks.length === 0) {
      // No face detected → return all zeros
      return new Array(EXPECTED_FACE_DIM).fill(0);
    }
    
    // Flatten all detected landmarks: (x, y, z)
    landmarks.forEach((lm) => {
      out.push(
        lm.x || 0,
        lm.y || 0,
        lm.z || 0
      );
    });
    
    // Pad with zeros if we got fewer landmarks than expected (e.g. MediaPipe
    // only detected a subset of the usual 468 for this frame).
    while (out.length < EXPECTED_FACE_DIM) {
      out.push(0);
    }
    
    // Truncate if somehow we got more (shouldn't happen, but defensive)
    return out.slice(0, EXPECTED_FACE_DIM);
  }

  // Validates a pose/face payload against the model's feature contract
  // before it ever leaves the browser. Returns null if valid, or a short
  // reason string if not — callers should drop the payload rather than
  // send it and let the backend guess.
  function validatePayload(posePayload, facePayload) {
    if (posePayload.length === 0 || facePayload.length === 0) {
      return "empty pose/face buffer";
    }
    if (posePayload.length !== facePayload.length) {
      return `pose/face frame count mismatch: ${posePayload.length} vs ${facePayload.length}`;
    }
    if (posePayload.length < FRAME_WINDOW) {
      return `not enough frames buffered: ${posePayload.length} < ${FRAME_WINDOW}`;
    }
    for (let i = 0; i < posePayload.length; i++) {
      if (posePayload[i].length !== EXPECTED_POSE_DIM) {
        return `pose frame ${i} has ${posePayload[i].length} dims, expected ${EXPECTED_POSE_DIM}`;
      }
      if (facePayload[i].length !== EXPECTED_FACE_DIM) {
        return `face frame ${i} has ${facePayload[i].length} dims, expected ${EXPECTED_FACE_DIM}`;
      }
    }
    return null;
  }

  async function sendToBackend() {
    if (poseBuffer.length < FRAME_WINDOW) return;

    const posePayload = poseBuffer.slice(-FRAME_WINDOW);
    const facePayload = faceBuffer.slice(-FRAME_WINDOW);
    poseBuffer = [];
    faceBuffer = [];

    const validationError = validatePayload(posePayload, facePayload);
    if (validationError) {
      // Never send a malformed payload — flattenPose/flattenFace already
      // pad/truncate to a fixed size, so this only fires on a genuine bug
      // (e.g. a future refactor changing EXPECTED_*_DIM without updating
      // the flatteners). Fail loudly in the console instead of sending
      // shapes the model wasn't trained on and getting a confusing 500
      // or silently garbled predictions back.
      console.error(`SilentBridge: refusing to send invalid payload — ${validationError}`);
      return;
    }

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
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Backend returned ${res.status}: ${errorText}`);
      }
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