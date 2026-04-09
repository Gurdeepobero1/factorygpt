# FactoryGPT — Deployment Guide

Three-tier deployment:

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  Frontend       │──────▶│  Cloud Broker    │◀──────│  Edge Vision    │
│  (Vercel)       │  WSS  │  (Render)        │  WSS  │  (Your machine) │
│  index.html     │       │  cloud_backend.py│       │  edge_vision.py │
└─────────────────┘       └──────────────────┘       └─────────────────┘
                                   │
                                   │ HTTPS
                                   ▼
                          ┌──────────────────┐
                          │  Sarvam AI       │
                          │  (LLM provider)  │
                          └──────────────────┘
```

---

## 1  Prerequisites

| You need | Where to get it |
|---|---|
| GitHub account | github.com |
| Render account | render.com (free) |
| Vercel account | vercel.com (free) |
| Sarvam AI API key | dashboard.sarvam.ai |
| A webcam (for edge) | any USB / built-in camera |

---

## 2  Push to GitHub

```bash
cd ~/Desktop/FactoryGPT
git init
git add .
git commit -m "Initial FactoryGPT deployment"
git branch -M main
git remote add origin https://github.com/<you>/factorygpt.git
git push -u origin main
```

The `.gitignore` already excludes `venv/`, `.env`, `*.db`, `*.pt`, and `.claude/launch.json`.

---

## 3  Deploy Cloud Broker → Render

### Option A — Blueprint (recommended, uses `render.yaml`)

1. Go to **render.com → New → Blueprint**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` and provisions:
   - Web service `factorygpt-cloud`
   - Build cmd: `pip install -r requirements-cloud.txt`
   - Start cmd: `uvicorn cloud_backend:app --host 0.0.0.0 --port $PORT`
4. In the **Environment** tab, set:
   - `SARVAM_API_KEY` = your key
   - `ALLOWED_ORIGINS` = `https://your-vercel-app.vercel.app` (update after step 4)
5. Render generates `FACTORYGPT_API_KEY` automatically — **copy it**, you'll paste it into `config.js` next.
6. Wait for "Live" status → note the URL (e.g. `https://factorygpt-cloud.onrender.com`).

Verify: `curl https://factorygpt-cloud.onrender.com/health` → should return `{"status":"ok", ...}`

### Option B — Docker on Fly.io / Railway / Cloud Run

```bash
docker build -t factorygpt-cloud .
docker run -p 8000:8000 \
  -e SARVAM_API_KEY=xxx \
  -e FACTORYGPT_API_KEY=yyy \
  -e ALLOWED_ORIGINS=https://yourapp.vercel.app \
  factorygpt-cloud
```

---

## 4  Deploy Frontend → Vercel

1. Edit `config.js` — paste your Render URL **and** API key:

   ```js
   window.BACKEND_API_BASE = "https://factorygpt-cloud.onrender.com";
   window.BACKEND_WS_URL   = "wss://factorygpt-cloud.onrender.com/ws/iot";
   window.FACTORYGPT_API_KEY = "paste-the-render-generated-key-here";
   ```

2. Commit + push:
   ```bash
   git add config.js && git commit -m "Point frontend at Render" && git push
   ```

3. **vercel.com → Add New → Project → Import Git Repository**
4. Framework: **Other** (Vercel reads `vercel.json` automatically)
5. Click **Deploy**
6. Copy your Vercel URL → go back to Render and set `ALLOWED_ORIGINS` to that URL → trigger a redeploy so CORS updates.

---

## 5  Run the Edge Vision Node (locally)

The edge node runs on **your machine** (the one with the camera) and pushes frames to the cloud broker.

```bash
cd ~/Desktop/FactoryGPT
source venv/bin/activate
pip install -r requirements-edge.txt
```

Create `.env` in the project root:

```env
CLOUD_WS_URL=wss://factorygpt-cloud.onrender.com/ws/iot
FACTORYGPT_API_KEY=paste-the-render-generated-key-here
STREAM_FPS=10
```

Run it:

```bash
python edge_vision.py
```

Open your Vercel URL → click **Bind & Start** on Node 0 → you should see the live feed with multi-zone PPE detection.

---

## 6  Smoke test checklist

- [ ] `curl https://<render>/health` returns ok
- [ ] Vercel site loads, WebSocket status shows **Nodes Connected**
- [ ] Edge terminal prints `Connected. Waiting for frontend init commands...`
- [ ] Clicking **Bind & Start** shows a live camera feed on the dashboard
- [ ] Standing without a hard hat / hi-vis vest triggers an alert badge
- [ ] The AI Brain panel responds to a test prompt
- [ ] Refreshing the page keeps the alert history (localStorage persistence)

---

## 7  Ongoing operations

| Task | Where |
|---|---|
| View alert history | `GET /api/alerts?limit=100` with `X-API-Key` header |
| System stats | `GET /api/stats` |
| Resolve an alert | `POST /api/alerts/{id}/resolve` |
| Change CORS | Render → Environment → `ALLOWED_ORIGINS` |
| Rotate API key | Render → Environment → regenerate `FACTORYGPT_API_KEY` → update `config.js` + edge `.env` |
| Free tier cold starts | Upgrade Render plan to Starter ($7/mo) |

---

## 8  Troubleshooting

| Symptom | Fix |
|---|---|
| CORS error in browser console | `ALLOWED_ORIGINS` on Render doesn't match Vercel URL exactly (no trailing slash) |
| WebSocket stays "Reconnecting…" | Check Render logs, free tier spins down after 15 min idle — first request wakes it |
| 403 on every request | `FACTORYGPT_API_KEY` in `config.js` doesn't match Render env var |
| Edge: `cv2.VideoCapture` fails | Wrong camera index — try 0, 1, 2 or grant camera permission to Terminal |
| AI returns 502 | `SARVAM_API_KEY` missing or expired on Render |
