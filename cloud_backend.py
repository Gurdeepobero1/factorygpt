"""
FactoryGPT — Cloud Broker  (v3.0)
================================================================================
Deployed to Render / Railway / Fly.io.
Bridges browser-based edge vision and the Frontend dashboard.
Features: multi-provider AI Brain, SQLite alert persistence, API-key auth,
health endpoint, browser-reported alert logging, WebSocket relay.
================================================================================
"""

import asyncio
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from ai_brain import call_brain, compose_system_prompt

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
FACTORYGPT_KEY  = os.getenv("FACTORYGPT_API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
DB_PATH         = os.getenv("DB_PATH", "factorygpt.db")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FactoryGPT Cloud Broker", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: Optional[str] = Security(_key_header)):
    if FACTORYGPT_KEY and key != FACTORYGPT_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key.")
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
                severity    TEXT DEFAULT 'critical',
                timestamp   TEXT,
                resolved    INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT UNIQUE,
                uploaded_at TEXT,
                row_count   INTEGER
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

def log_alert(node_id: int, alert_type: str, message: str, severity: str = "critical"):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO alerts (node_id, alert_type, message, severity, timestamp) VALUES (?, ?, ?, ?, ?)",
            (node_id, alert_type, message, severity, datetime.utcnow().isoformat()),
        )
        conn.commit()

# ── In-memory CSV store (for AI context) ─────────────────────────────────────
factory_data_store: dict[str, str] = {}

# ── WebSocket manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, message: str, sender: WebSocket | None = None):
        dead = []
        for ws in self.active:
            if ws is sender:
                continue
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    from ai_brain import _available_providers
    return {
        "status": "ok",
        "version": "3.0.0",
        "connections": len(manager.active),
        "datasets": list(factory_data_store.keys()),
        "llm_providers_available": _available_providers(),
        "ts": datetime.utcnow().isoformat(),
    }

@app.post("/api/upload", dependencies=[Depends(verify_api_key)])
async def upload_factory_data(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV / Excel accepted.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    factory_data_store[file.filename] = df.to_json(orient="records")
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO datasets (filename, uploaded_at, row_count) VALUES (?, ?, ?)",
            (file.filename, datetime.utcnow().isoformat(), len(df)),
        )
        conn.commit()
    return {"status": "success", "rows_processed": len(df), "filename": file.filename}


class Message(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    system_prompt: str
    messages: List[Message]
    live_context: Optional[str] = None

@app.post("/api/ai", dependencies=[Depends(verify_api_key)])
async def process_ai_request(req: AIRequest):
    """
    Main AI endpoint — routes to the best available LLM provider with
    auto-fallback, smart context assembly, and master prompt injection.
    """
    try:
        full_system = compose_system_prompt(
            base_user_prompt=req.system_prompt,
            factory_data=factory_data_store,
            live_context=req.live_context,
        )
        text, provider = await call_brain(
            system_prompt=full_system,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=0.3,
        )
        return {"status": "success", "response": text, "provider": provider}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


class AlertIn(BaseModel):
    node_id: int
    alert_type: str
    message: str
    severity: str = "critical"

@app.post("/api/alerts/log", dependencies=[Depends(verify_api_key)])
async def log_browser_alert(alert: AlertIn):
    """Endpoint for browser-based edge vision to report PPE violations."""
    log_alert(alert.node_id, alert.alert_type, alert.message, alert.severity)
    # Broadcast to any connected dashboards in real-time
    await manager.broadcast(json.dumps({
        "type": "alert",
        "node_id": alert.node_id,
        "alert_type": alert.alert_type,
        "message": alert.message,
        "severity": alert.severity,
        "timestamp": datetime.utcnow().isoformat(),
    }))
    return {"status": "logged"}

@app.get("/api/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts(limit: int = 100, unresolved_only: bool = False):
    with get_db() as conn:
        query = "SELECT * FROM alerts"
        if unresolved_only:
            query += " WHERE resolved = 0"
        query += f" ORDER BY id DESC LIMIT {int(limit)}"
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
        total   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        open_   = conn.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]
        ds_cnt  = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        by_type = conn.execute(
            "SELECT alert_type, COUNT(*) as cnt FROM alerts GROUP BY alert_type ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
    return {
        "total_alerts":    total,
        "open_alerts":     open_,
        "datasets_loaded": ds_cnt,
        "top_alert_types": [dict(r) for r in by_type],
    }

# ── WebSocket relay ───────────────────────────────────────────────────────────
@app.websocket("/ws/iot")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    print(f"[WS] Client connected — total: {len(manager.active)}")
    try:
        while True:
            data = await ws.receive_text()
            await manager.broadcast(data, sender=ws)
    except WebSocketDisconnect:
        manager.disconnect(ws)
        print(f"[WS] Client disconnected — total: {len(manager.active)}")
