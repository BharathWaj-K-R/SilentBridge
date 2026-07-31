/**
 * Backend API base URL.
 * Local dev: FastAPI dev server. Production: the backend's Render URL.
 * Update the production fallback below once you know your actual
 * silentbridge-backend Render URL (shown on its dashboard page).
 */
window.SB_API_BASE_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/api/v1"
    : "https://silentbridge-backend-qpsn.onrender.com/api/v1";
