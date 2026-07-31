/**
 * Thin fetch wrappers around the SilentBridge backend API.
 * Depends on config.js being loaded first (defines API_BASE_URL).
 */

async function apiRegister(username, password) {
  const res = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(`Register failed: ${res.status}`);
  return res.json();
}

async function apiLogin(username, password) {
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(`Login failed: ${res.status}`);
  return res.json(); // { access_token, token_type }
}

/**
 * poseKeypoints / faceKeypoints: arrays shaped (frames, feature_dim),
 * produced by whatever keypoint extractor runs client-side (e.g. MediaPipe
 * Holistic via a JS/WASM build) or sent to a server-side extraction step —
 * that extraction pipeline isn't implemented yet, this just wires the call.
 */
async function apiTranslate({ userId = null, adapterId = null, poseKeypoints, faceKeypoints }) {
  const res = await fetch(`${API_BASE_URL}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      adapter_id: adapterId,
      pose_keypoints: poseKeypoints,
      face_keypoints: faceKeypoints,
    }),
  });
  if (!res.ok) throw new Error(`Translate failed: ${res.status}`);
  return res.json(); // { predicted_text, confidence, latency_ms, used_adapter }
}

async function apiCalibrate({ userId, calibrationSeconds, poseKeypoints, faceKeypoints, labelIds }) {
  const res = await fetch(`${API_BASE_URL}/calibration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      calibration_seconds: calibrationSeconds,
      pose_keypoints: poseKeypoints,
      face_keypoints: faceKeypoints,
      label_ids: labelIds,
    }),
  });
  if (!res.ok) throw new Error(`Calibration failed: ${res.status}`);
  return res.json(); // { adapter_id, calibration_seconds, param_count, accuracy_gain_pct }
}
