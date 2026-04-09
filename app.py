import cv2
import numpy as np
import base64
import asyncio
import json
import pandas as pd
import io
import os
import sqlite3
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import httpx
from ultralytics import YOLO
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")
FACTORYGPT_KEY   = os.getenv("FACTORYGPT_API_KEY")          # optional; if unset → dev open access
ALLOWED_ORIGINS  = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # e.g. "https://app.example.com"
DB_PATH          = os.getenv("DB_PATH", "factorygpt.db")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FactoryGPT Production API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Vision model ──────────────────────────────────────────────────────────────
print("Initializing YOLOv8 Neural Edge Node...")
vision_model = None
try:
    vision_model = YOLO("yolov8n.pt")
    print("YOLOv8 loaded.")
except Exception as e:
    print(f"[WARN] Failed to load vision model: {e} — vision endpoints disabled.")

# ── Auth ──────────────────────────────────────────────────────────────────────
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: Optional[str] = Security(_key_header)):
    """If FACTORYGPT_API_KEY is set in env, every request must supply it."""
    if FACTORYGPT_KEY and key != FACTORYGPT_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")
    return True

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id     INTEGER,
                alert_type  TEXT,
                message     TEXT,
                timestamp   TEXT,
                resolved    INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT UNIQUE,
                uploaded_at TEXT,
                row_count   INTEGER
            );
            CREATE TABLE IF NOT EXISTS production_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                line_name   TEXT,
                oee         REAL,
                output      INTEGER,
                target      INTEGER,
                downtime    REAL,
                defects     REAL,
                logged_at   TEXT
            );
        """)
        conn.commit()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def log_alert(node_id: int, alert_type: str, message: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO alerts (node_id, alert_type, message, timestamp) VALUES (?, ?, ?, ?)",
            (node_id, alert_type, message, datetime.utcnow().isoformat())
        )
        conn.commit()

# ── In-memory data store (uploaded CSV context for AI) ───────────────────────
factory_data_store: dict[str, str] = {}

# ── PPE Detection helpers ─────────────────────────────────────────────────────
def check_hard_hat(head_region: np.ndarray) -> bool:
    """Detect helmet presence by colour in the head region (top 25 % of bbox)."""
    if head_region.size == 0:
        return True   # can't assess → assume OK
    hsv  = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    # Yellow/lime hard hat
    m1 = cv2.inRange(hsv, np.array([18, 80,  80]), np.array([42, 255, 255]))
    # Orange hard hat
    m2 = cv2.inRange(hsv, np.array([0,  140, 80]), np.array([18, 255, 255]))
    # White hard hat (low saturation, high value)
    m3 = cv2.inRange(hsv, np.array([0,   0, 175]), np.array([180, 50, 255]))
    total   = head_region.shape[0] * head_region.shape[1] + 1
    ratio   = (cv2.countNonZero(m1) + cv2.countNonZero(m2) + cv2.countNonZero(m3)) / total
    return ratio > 0.08

def check_hi_vis_vest(torso_region: np.ndarray) -> bool:
    """Detect high-vis safety vest in the torso region (20–70 % of bbox)."""
    if torso_region.size == 0:
        return True   # can't assess → assume OK
    hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
    # Neon yellow-green vest
    m1 = cv2.inRange(hsv, np.array([25, 100, 100]), np.array([75, 255, 255]))
    # High-vis orange vest
    m2 = cv2.inRange(hsv, np.array([5,  120, 120]), np.array([22, 255, 255]))
    total = torso_region.shape[0] * torso_region.shape[1] + 1
    ratio = (cv2.countNonZero(m1) + cv2.countNonZero(m2)) / total
    return ratio > 0.05

def annotate_ppe(frame: np.ndarray, results) -> tuple[np.ndarray, list[str]]:
    """
    Run per-person PPE checks and return annotated frame + list of violation strings.
    Checks both hard hat AND high-vis vest independently.
    """
    violations: list[str] = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.40:          # skip low-confidence detections
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = y2 - y1

            head_y1  = y1
            head_y2  = y1 + max(int(h * 0.25), 1)
            torso_y1 = y1 + int(h * 0.20)
            torso_y2 = y1 + int(h * 0.70)

            head_region  = frame[head_y1:head_y2,  x1:x2]
            torso_region = frame[torso_y1:torso_y2, x1:x2]

            has_hat  = check_hard_hat(head_region)
            has_vest = check_hi_vis_vest(torso_region)

            issues = []
            if not has_hat:
                issues.append("No Hard Hat")
            if not has_vest:
                issues.append("No Hi-Vis Vest")

            if issues:
                violations.extend(issues)
                label  = " | ".join(issues)
                colour = (0, 0, 255)
            else:
                label  = f"PPE OK ({conf:.0%})"
                colour = (0, 200, 60)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return frame, violations

# ── WebSocket manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()
active_tasks: dict[int, asyncio.Task] = {}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "vision_model": vision_model is not None, "ts": datetime.utcnow().isoformat()}

@app.post("/api/upload", dependencies=[Depends(verify_api_key)])
async def upload_factory_data(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV / Excel files are accepted.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    factory_data_store[file.filename] = df.to_json(orient="records")

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO datasets (filename, uploaded_at, row_count) VALUES (?, ?, ?)",
            (file.filename, datetime.utcnow().isoformat(), len(df))
        )
        conn.commit()

    return {"status": "success", "rows_processed": len(df), "filename": file.filename}

class Message(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    system_prompt: str
    messages: List[Message]

@app.post("/api/ai", dependencies=[Depends(verify_api_key)])
async def process_ai_request(req: AIRequest):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured.")

    context_str = "\n\n".join(
        [f"DATASET ({k}):\n{v}" for k, v in factory_data_store.items()]
    ) or "No historical CSV data provided."

    enhanced_system = f"{req.system_prompt}\n\nHISTORICAL & WORKER CONTEXT:\n{context_str}"

    payload = {
        "model": "sarvam-30b",
        "messages": [{"role": "system", "content": enhanced_system}]
                  + [{"role": m.role, "content": m.content} for m in req.messages],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                "https://api.sarvam.ai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"},
            )
            r.raise_for_status()
            return {"status": "success", "response": r.json()["choices"][0]["message"]["content"]}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Sarvam API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts(limit: int = 100, unresolved_only: bool = False):
    with get_db() as conn:
        query = "SELECT * FROM alerts"
        if unresolved_only:
            query += " WHERE resolved = 0"
        query += f" ORDER BY id DESC LIMIT {limit}"
        rows = conn.execute(query).fetchall()
    return {"alerts": [dict(r) for r in rows]}

@app.post("/api/alerts/{alert_id}/resolve", dependencies=[Depends(verify_api_key)])
async def resolve_alert(alert_id: int):
    with get_db() as conn:
        conn.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
        conn.commit()
    return {"status": "resolved", "id": alert_id}

@app.get("/api/stats", dependencies=[Depends(verify_api_key)])
async def get_stats():
    with get_db() as conn:
        total_alerts   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        open_alerts    = conn.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]
        datasets_count = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        recent_alerts  = conn.execute(
            "SELECT alert_type, COUNT(*) as cnt FROM alerts GROUP BY alert_type ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
    return {
        "total_alerts":   total_alerts,
        "open_alerts":    open_alerts,
        "datasets_loaded": datasets_count,
        "top_alert_types": [dict(r) for r in recent_alerts],
    }

# ── WebSocket / Camera streams ────────────────────────────────────────────────
@app.websocket("/ws/iot")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                cmd = payload.get("command")
                if cmd == "init_node":
                    node_id  = int(payload.get("node", 0))
                    cam_idx  = int(payload.get("camera_index", 0))
                    if node_id in active_tasks:
                        active_tasks[node_id].cancel()
                    fn = process_ppe_stream if node_id == 0 else process_motion_stream
                    active_tasks[node_id] = asyncio.create_task(fn(node_id, cam_idx))
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)

async def process_ppe_stream(node_id: int, camera_index: int):
    """Node 0 — multi-zone PPE detection (hard hat + hi-vis vest)."""
    if vision_model is None:
        return
    cap = cv2.VideoCapture(camera_index)
    last_alert_msg = ""
    try:
        while True:
            await asyncio.sleep(0.1)
            if not manager.active:
                continue
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1.0)
                continue

            frame = cv2.resize(frame, (320, 240))
            results = vision_model(frame, classes=[0], verbose=False)

            human_detected = any(len(r.boxes) > 0 for r in results)
            frame, violations = annotate_ppe(frame, results)

            alert_msg = None
            if violations:
                alert_msg = "PPE violation: " + "; ".join(set(violations))
                if alert_msg != last_alert_msg:
                    last_alert_msg = alert_msg
                    log_alert(node_id, "ppe_violation", alert_msg)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            await manager.broadcast(json.dumps({
                "node_id":    node_id,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "motion_pct": 100 if human_detected else 0,
                "image":      base64.b64encode(buf).decode(),
                "alert":      alert_msg,
            }))
    finally:
        cap.release()

async def process_motion_stream(node_id: int, camera_index: int):
    """Node 1 — motion activity percentage."""
    cap = cv2.VideoCapture(camera_index)
    prev_gray = None
    try:
        while True:
            await asyncio.sleep(0.1)
            if not manager.active:
                continue
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1.0)
                continue

            frame     = cv2.resize(frame, (320, 240))
            gray      = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            motion_pct = 0.0

            if prev_gray is not None:
                delta     = cv2.absdiff(prev_gray, gray)
                thresh    = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_pct = round(cv2.countNonZero(thresh) / (320 * 240) * 100, 2)
                colour    = (0, 200, 60) if motion_pct > 2.0 else (0, 80, 200)
                cv2.putText(frame, f"Activity: {motion_pct}%", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

            prev_gray = gray
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            await manager.broadcast(json.dumps({
                "node_id":    node_id,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "motion_pct": motion_pct,
                "image":      base64.b64encode(buf).decode(),
                "alert":      None,
            }))
    finally:
        cap.release()
