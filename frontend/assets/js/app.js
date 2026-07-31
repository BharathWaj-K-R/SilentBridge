/* SilentBridge — shared UI behavior
   No jQuery, no packages. Vanilla JS only. */

(function () {
  "use strict";

  // Shared nav markup so every page has a consistent header without duplication.
  const NAV_LINKS = [
    { href: "/index.html", label: "Home" },
    { href: "/pages/dashboard.html", label: "Dashboard" },
    { href: "/pages/translate.html", label: "Translate" },
    { href: "/pages/calibration.html", label: "Calibration" },
    { href: "/pages/history.html", label: "History" },
    { href: "/pages/users.html", label: "Users" },
    { href: "/pages/evaluation.html", label: "Evaluation" },
    { href: "/pages/settings.html", label: "Settings" },
  ];

  function renderNav() {
    const mount = document.querySelector("[data-sb-nav]");
    if (!mount) return;
    const currentPath = window.location.pathname.replace(/\/$/, "");
    const links = NAV_LINKS.map((link) => {
      const isActive = currentPath.endsWith(link.href.split("/").pop());
      return `
        <li class="nav-item">
          <a class="nav-link ${isActive ? "active" : ""}" href="${link.href}">${link.label}</a>
        </li>`;
    }).join("");

    mount.innerHTML = `
      <nav class="sb-nav" aria-label="Primary">
        <div class="container-sb d-flex align-items-center justify-content-between py-3">
          <a class="navbar-brand" href="/index.html" aria-label="SilentBridge home">
            <span class="sb-logo-mark" aria-hidden="true"><i class="bi bi-soundwave"></i></span>
            SilentBridge
          </a>
          <button class="btn btn-ghost-sb d-lg-none btn-icon" type="button"
            aria-expanded="false" aria-controls="sbNavList" data-sb-nav-toggle>
            <i class="bi bi-list" aria-hidden="true"></i>
            <span class="visually-hidden">Toggle navigation</span>
          </button>
          <ul id="sbNavList" class="nav align-items-center gap-1 d-none d-lg-flex" role="menubar">
            ${links}
          </ul>
          <div class="d-none d-lg-flex align-items-center gap-2">
            <a href="/pages/translate.html" class="btn btn-primary-sb btn-icon btn-ripple">
              <i class="bi bi-broadcast" aria-hidden="true"></i> Start Translation
            </a>
          </div>
        </div>
        <div class="container-sb d-lg-none pb-3 d-none" data-sb-nav-mobile>
          <ul class="nav flex-column gap-1">${links}</ul>
          <a href="/pages/translate.html" class="btn btn-primary-sb w-100 mt-2">Start Translation</a>
        </div>
      </nav>
    `;

    const toggle = mount.querySelector("[data-sb-nav-toggle]");
    const mobile = mount.querySelector("[data-sb-nav-mobile]");
    if (toggle && mobile) {
      toggle.addEventListener("click", () => {
        const open = mobile.classList.toggle("d-none") === false;
        toggle.setAttribute("aria-expanded", String(open));
      });
    }

    const navEl = mount.querySelector(".sb-nav");
    if (navEl) {
      const onScroll = () => navEl.classList.toggle("scrolled", window.scrollY > 4);
      onScroll();
      window.addEventListener("scroll", onScroll, { passive: true });
    }
  }

  function renderFooter() {
    const mount = document.querySelector("[data-sb-footer]");
    if (!mount) return;
    mount.innerHTML = `
      <footer class="sb-footer">
        <div class="container-sb d-flex flex-wrap gap-3 justify-content-between align-items-center">
          <div class="d-flex align-items-center gap-2">
            <span class="sb-logo-mark" aria-hidden="true"><i class="bi bi-soundwave"></i></span>
            <span>&copy; ${new Date().getFullYear()} SilentBridge. Designed for accessible communication.</span>
          </div>
          <div class="d-flex gap-3">
            <a href="/pages/settings.html">Settings</a>
            <a href="#" aria-label="Privacy policy">Privacy</a>
            <a href="#" aria-label="Terms of service">Terms</a>
          </div>
        </div>
      </footer>
    `;
  }

  // Button ripple micro-interaction
  function initRipples() {
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".btn-ripple");
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      btn.style.setProperty("--x", `${((e.clientX - rect.left) / rect.width) * 100}%`);
      btn.style.setProperty("--y", `${((e.clientY - rect.top) / rect.height) * 100}%`);
      btn.classList.remove("is-rippling");
      // Force reflow so the animation can restart
      void btn.offsetWidth;
      btn.classList.add("is-rippling");
      setTimeout(() => btn.classList.remove("is-rippling"), 500);
    });
  }

  // Reveal on scroll for elements marked [data-sb-reveal]
  function initReveal() {
    const items = document.querySelectorAll("[data-sb-reveal]");
    if (!items.length || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("sb-fade-up");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    items.forEach((el) => io.observe(el));
  }

  // Circular progress helper — usage: <div class="sb-progress-ring" data-value="72">
  function initRings() {
    document.querySelectorAll(".sb-progress-ring[data-value]").forEach((el) => {
      const val = Math.max(0, Math.min(100, Number(el.dataset.value) || 0));
      el.style.setProperty("--value", val);
      const valNode = el.querySelector(".val");
      if (valNode && !valNode.dataset.static) valNode.textContent = `${val}%`;
    });
  }

  // Simple copy-to-clipboard
  function initCopy() {
    document.querySelectorAll("[data-sb-copy]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const targetSel = btn.getAttribute("data-sb-copy");
        const target = targetSel ? document.querySelector(targetSel) : null;
        const text = target ? target.innerText : btn.getAttribute("data-copy-text") || "";
        try {
          await navigator.clipboard.writeText(text);
          const original = btn.innerHTML;
          btn.innerHTML = '<i class="bi bi-check2" aria-hidden="true"></i> Copied';
          setTimeout(() => (btn.innerHTML = original), 1400);
        } catch (err) {
          console.warn("Copy failed", err);
        }
      });
    });
  }

  // Fake live translation ticker used on the Translate page
  function initFakeTranslation() {
    const captionEl = document.querySelector("[data-sb-caption]");
    const confBar = document.querySelector("[data-sb-confidence-bar]");
    const confVal = document.querySelector("[data-sb-confidence-val]");
    const latency = document.querySelector("[data-sb-latency]");
    const fps = document.querySelector("[data-sb-fps]");
    if (!captionEl) return;

    const phrases = [
      "Hello, how are you today?",
      "My name is Aarav. Nice to meet you.",
      "Could you please repeat that?",
      "Thank you for your patience.",
      "I would like some water, please.",
      "Where is the nearest metro station?",
    ];
    let i = 0;

    const startBtn = document.querySelector("[data-sb-start]");
    const pauseBtn = document.querySelector("[data-sb-pause]");
    const stopBtn = document.querySelector("[data-sb-stop]");
    let timer = null;

    function tick() {
      captionEl.style.opacity = "0";
      setTimeout(() => {
        captionEl.textContent = phrases[i % phrases.length];
        captionEl.style.opacity = "1";
        const conf = Math.floor(88 + Math.random() * 10);
        if (confBar) confBar.style.width = `${conf}%`;
        if (confVal) confVal.textContent = `${conf}%`;
        if (latency) latency.textContent = `${Math.floor(220 + Math.random() * 180)} ms`;
        if (fps) fps.textContent = `${Math.floor(28 + Math.random() * 4)} fps`;
        i += 1;
      }, 200);
    }

    function start() { if (!timer) { tick(); timer = setInterval(tick, 2400); } }
    function pause() { if (timer) { clearInterval(timer); timer = null; } }
    function stop() { pause(); captionEl.textContent = "Translation stopped. Press Start to resume."; }

    startBtn && startBtn.addEventListener("click", start);
    pauseBtn && pauseBtn.addEventListener("click", pause);
    stopBtn && stopBtn.addEventListener("click", stop);

    // Auto-start after a short delay for demo polish
    setTimeout(start, 600);
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderNav();
    renderFooter();
    initRipples();
    initReveal();
    initRings();
    initCopy();
    // initFakeTranslation() removed: translate.html now uses
    // translate-live.js for real webcam + backend wiring, with its own
    // internal fallback if camera/model/backend are unavailable.
  });
})();
