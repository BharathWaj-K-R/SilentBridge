/**
 * Central place for the backend API base URL.
 * Local dev: your FastAPI dev server (uvicorn app.main:app --reload).
 * Production: the backend's Render Web Service URL.
 *
 * Drop your existing Lovable.dev-generated HTML/CSS/JS into this frontend/
 * folder alongside this file, and import API_BASE_URL wherever fetch calls
 * are made (see api.js for ready-made helper functions).
 */
const API_BASE_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/api/v1"
    : "https://YOUR-BACKEND-SERVICE.onrender.com/api/v1";
