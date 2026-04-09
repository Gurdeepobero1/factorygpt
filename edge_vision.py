"""
FactoryGPT — Edge Vision Node  (v2.0)
======================================
Runs on-device (Raspberry Pi / laptop).
Performs multi-zone PPE detection and streams results to the Cloud Broker.
Reconnects automatically with exponential back-off on connection failure.
"""

import cv2
import numpy as np
import base64
import asyncio
import json
import os
from datetime import datetime

import websockets
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
CLOUD_WS_URL   = os.getenv("CLOUD_WS_URL", "ws://localhost:8000/ws/iot")
FACTORY_API_KEY = os.getenv("FACTORYGPT_API_KEY", "")   # must match server if auth is on
STREAM_FPS     = float(os.getenv("STREAM_FPS", "10"))    # frames per second
JPEG_QUALITY   = int(os.getenv("JPEG_QUALITY", "72"))    # 1-100

RECONNECT_BASE = 1    # seconds
RECONNECT_MAX  = 60   # seconds

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading YOLOv8 on Edge Device...")
vision_model = YOLO("yolov8n.pt")
print("Model ready.")

# ── PPE helpers ────────────────────────────────────────────────────────────────
def _ratio(mask: np.ndarray, region: np.ndarray) -> float:
    total = region.shape[0] * region.shape[1] + 1
    return cv2.countNonZero(mask) / total

def has_hard_hat(head: np.ndarray) -> bool:
    """
    Yellow / orange / white helmet detection in the head bounding-box region.
    Returns True if PPE is present (or region is too small to assess).
    """
    if head.size < 200:
        return True
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([18, 80,  80]), np.array([42, 255, 255]))
    orange = cv2.inRange(hsv, np.array([0,  140, 80]), np.array([18, 255, 255]))
    white  = cv2.inRange(hsv, np.array([0,  0,  175]), np.array([180, 50, 255]))
    return _ratio(yellow + orange + white, head) > 0.08

def has_hi_vis_vest(torso: np.ndarray) -> bool:
    """
    Neon yellow-green / high-vis orange vest detection in the torso region.
    Returns True if PPE is present (or region is too small to assess).
    """
    if torso.size < 400:
        return True
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    neon   = cv2.inRange(hsv, np.array([25, 100, 100]), np.array([75, 255, 255]))
    orange = cv2.inRange(hsv, np.array([5,  120, 120]), np.array([22, 255, 255]))
    return _ratio(neon + orange, torso) > 0.05

def analyse_ppe(frame: np.ndarray, results) -> tuple[np.ndarray, list[str]]:
    """
    For each detected person run multi-zone PPE analysis.
    Returns annotated frame and list of unique violation strings.
    """
    violations: list[str] = []
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < 0.40:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = y2 - y1

            head  = frame[y1            : y1 + max(int(h * 0.25), 1), x1:x2]
            torso = frame[y1 + int(h * 0.20) : y1 + int(h * 0.70),   x1:x2]

            hat_ok  = has_hard_hat(head)
            vest_ok = has_hi_vis_vest(torso)

            issues = []
            if not hat_ok:
                issues.append("No Hard Hat")
            if not vest_ok:
                issues.append("No Hi-Vis Vest")

            if issues:
                violations.extend(issues)
                label  = " | ".join(issues)
                colour = (0, 0, 255)
            else:
                label  = f"PPE OK ({float(box.conf[0]):.0%})"
                colour = (0, 200, 60)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return frame, violations

# ── Streaming ──────────────────────────────────────────────────────────────────
active_streams: dict[int, asyncio.Task] = {}

async def stream_node(websocket, node_id: int, cam_idx: int):
    """Stream a single camera; cancelled by reconnect/re-bind."""
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_idx}")
        return

    prev_gray      = None
    last_violation = ""
    interval       = 1.0 / STREAM_FPS
    print(f"[Node {node_id}] Streaming camera {cam_idx} at {STREAM_FPS} FPS")

    try:
        while True:
            await asyncio.sleep(interval)
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1.0)
                continue

            frame      = cv2.resize(frame, (320, 240))
            motion_pct = 0.0
            alert_msg  = None

            if node_id == 0:
                # ── PPE mode ──────────────────────────────────────────────────
                results        = vision_model(frame, classes=[0], verbose=False)
                human_detected = any(len(r.boxes) > 0 for r in results)
                frame, violations = analyse_ppe(frame, results)

                motion_pct = 100.0 if human_detected else 0.0
                if violations:
                    alert_msg = "PPE violation: " + "; ".join(set(violations))
                    if alert_msg != last_violation:
                        last_violation = alert_msg
                        print(f"[ALERT] {alert_msg}")
            else:
                # ── Motion mode ───────────────────────────────────────────────
                gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if prev_gray is not None:
                    delta      = cv2.absdiff(prev_gray, gray)
                    thresh     = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                    motion_pct = round(cv2.countNonZero(thresh) / (320 * 240) * 100, 2)
                    colour     = (0, 200, 60) if motion_pct > 2.0 else (0, 80, 200)
                    cv2.putText(frame, f"Activity: {motion_pct}%", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
                prev_gray = gray

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            payload = json.dumps({
                "node_id":    node_id,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "motion_pct": motion_pct,
                "image":      base64.b64encode(buf).decode(),
                "alert":      alert_msg,
            })
            await websocket.send(payload)

    except asyncio.CancelledError:
        pass
    finally:
        cap.release()
        print(f"[Node {node_id}] Stream stopped.")

# ── Main loop with reconnection ────────────────────────────────────────────────
async def edge_client():
    backoff = RECONNECT_BASE
    extra_headers = {}
    if FACTORY_API_KEY:
        extra_headers["X-API-Key"] = FACTORY_API_KEY

    while True:
        print(f"Connecting to Cloud Broker at {CLOUD_WS_URL} ...")
        try:
            async with websockets.connect(CLOUD_WS_URL, extra_headers=extra_headers) as ws:
                backoff = RECONNECT_BASE   # reset on successful connect
                print("Connected. Waiting for frontend init commands...")

                async for raw in ws:
                    try:
                        data    = json.loads(raw)
                        cmd     = data.get("command")
                        if cmd == "init_node":
                            node_id = int(data.get("node", 0))
                            cam_idx = int(data.get("camera_index", 0))
                            print(f"[CMD] init_node={node_id} cam={cam_idx}")

                            # Cancel existing stream for this node
                            if node_id in active_streams:
                                active_streams[node_id].cancel()

                            active_streams[node_id] = asyncio.create_task(
                                stream_node(ws, node_id, cam_idx)
                            )
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            # Cancel all running streams before reconnecting
            for task in active_streams.values():
                task.cancel()
            active_streams.clear()

            print(f"Connection error: {e}. Retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX)

if __name__ == "__main__":
    asyncio.run(edge_client())
