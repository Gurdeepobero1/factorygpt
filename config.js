// ──────────────────────────────────────────────────────────────────────────────
// FactoryGPT Frontend Config
// Edit these values (or inject at deploy time) — no rebuild required.
// ──────────────────────────────────────────────────────────────────────────────
(function () {
  const isLocalhost =
    location.hostname === "localhost" || location.hostname === "127.0.0.1";

  if (isLocalhost) {
    // Local development — talk to the local Main API (app.py)
    window.BACKEND_API_BASE = "http://localhost:8000";
    window.BACKEND_WS_URL   = "ws://localhost:8000/ws/iot";
    window.FACTORYGPT_API_KEY = "";   // blank = dev mode
  } else {
    // Production — pointing at the live Render broker
    window.BACKEND_API_BASE   = "https://factorygpt-cloud.onrender.com";
    window.BACKEND_WS_URL     = "wss://factorygpt-cloud.onrender.com/ws/iot";
    window.FACTORYGPT_API_KEY = "d6oorxSgmiSZqM0cp+JcxcSeI3gc8Y+MsQD07LnJXDQ=";
  }
})();
