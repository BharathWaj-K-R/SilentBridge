# SilentBridge Frontend

This folder is a placeholder for the existing UI built separately (HTML5 /
CSS3 / Bootstrap 5 / vanilla JS, originally scaffolded via Lovable.dev).

**To wire it up:**
1. Copy your built HTML/CSS/JS files into this folder (keep `index.html` at
   the root of `frontend/`).
2. Include `js/config.js` then `js/api.js` in your HTML, before your own
   scripts:
   ```html
   <script src="js/config.js"></script>
   <script src="js/api.js"></script>
   <script src="js/app.js"></script> <!-- your existing app logic -->
   ```
3. Call `apiLogin`, `apiTranslate`, `apiCalibrate` etc. from your existing
   UI event handlers instead of hardcoded/mock data.
4. Update `API_BASE_URL` in `config.js` once the backend is deployed on
   Render.

## Local preview
Any static file server works, e.g.:
```bash
npx serve frontend
```

## Deploying on Render
Deploy this folder as a **Static Site**:
- Build command: (none needed for plain HTML/CSS/JS)
- Publish directory: `frontend`
