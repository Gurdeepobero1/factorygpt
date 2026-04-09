"""
FactoryGPT — Cloud Broker  (v2.0)
===================================
Deployed to Render / Railway / Fly.io.
Bridges Edge Vision nodes and the Frontend dashboard over WebSocket.
Includes API-key auth, health endpoint, and graceful broadcast.
"""

import asyncio
import json
import pandas as pd
import io
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
FACTORYGPT_KEY  = os.getenv("FACTORYGPT_API_KEY", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="FactoryGPT Cloud Broker", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: Optional[str] = Security(_key_header)):
    if FACTORYGPT_KEY and key != FACTORYGPT_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key.")
    return True

# ── In-memory CSV store ────────────────────────────────────────────────────────
factory_data_store: dict[str, str] = {}

# ── WebSocket manager ──────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, message: str, sender: WebSocket | None = None):
        """Relay to all connections except the sender."""
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

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "connections": len(manager.active),
        "datasets": list(factory_data_store.keys()),
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
    return {"status": "success", "rows_processed": len(df), "filename": file.filename}


class Message(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    system_prompt: str
    messages: list[Message]

@app.post("/api/ai", dependencies=[Depends(verify_api_key)])
async def process_ai_request(req: AIRequest):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured.")

    context_str = "\n\n".join(
        [f"DATASET ({k}):\n{v}" for k, v in factory_data_store.items()]
    ) or "No historical CSV data provided."

    payload = {
        "model": "sarvam-30b",
        "messages": [{"role": "system", "content": f"{req.system_prompt}\n\nCONTEXT:\n{context_str}"}]
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

# ── WebSocket relay ────────────────────────────────────────────────────────────
@app.websocket("/ws/iot")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    print(f"[WS] Client connected — total: {len(manager.active)}")
    try:
        while True:
            data = await ws.receive_text()
            # Relay everything (video frames from edge, commands from frontend)
            await manager.broadcast(data, sender=ws)
    except WebSocketDisconnect:
        manager.disconnect(ws)
        print(f"[WS] Client disconnected — total: {len(manager.active)}")
