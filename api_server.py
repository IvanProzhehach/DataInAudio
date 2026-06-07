"""
AcouSteg v6 — Decode API for Render.

  GET  /health
  POST /api/decode
  WS   /ws/decode

Local: uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from decoder import CFG, StreamSession, configure_decoder, decode_bytes

configure_decoder(airplay=True)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

app = FastAPI(title="AcouSteg v6 Decode API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecodeResponse(BaseModel):
    ok: bool
    frames: list[dict]
    message: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "decoder": "acousteg-v6",
        "sample_rate": CFG.sample_rate,
        "airplay_band_hz": [CFG.band_low, CFG.band_high],
    }


@app.post("/api/decode", response_model=DecodeResponse)
async def decode_upload(file: UploadFile = File(...)) -> DecodeResponse:
    data = await file.read()
    if not data:
        return DecodeResponse(ok=False, frames=[], message="empty file")
    try:
        frames = decode_bytes(data, realtime=True)
    except Exception as exc:
        return DecodeResponse(ok=False, frames=[], message=str(exc))
    if not frames:
        return DecodeResponse(ok=True, frames=[], message="no v6 frame found")
    return DecodeResponse(ok=True, frames=frames)


@app.websocket("/ws/decode")
async def ws_decode(ws: WebSocket) -> None:
    await ws.accept()
    session = StreamSession(airplay=True)
    await ws.send_json({
        "type": "ready",
        "sample_rate": CFG.sample_rate,
        "buffer_seconds": session.buffer_seconds,
        "chunk_format": "float32_le_mono",
    })

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "invalid json"})
                    continue

                msg_type = payload.get("type")
                if msg_type == "start":
                    session.reset()
                    sr = int(payload.get("sample_rate", CFG.sample_rate))
                    session.set_input_sample_rate(sr)
                    await ws.send_json({"type": "started", "sample_rate": sr})
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong", **session.status()})
                elif msg_type == "status":
                    await ws.send_json({"type": "status", **session.status()})
                else:
                    await ws.send_json({"type": "error", "message": f"unknown type: {msg_type}"})
                continue

            data = message.get("bytes")
            if not data:
                continue

            if len(data) % 4 != 0:
                await ws.send_json({
                    "type": "error",
                    "message": "pcm chunk must be float32 (multiple of 4 bytes)",
                })
                continue

            chunk = np.frombuffer(data, dtype=np.float32)
            session.push_pcm(chunk)

            for hit in session.scan():
                await ws.send_json({"type": "decoded", **hit})

    except WebSocketDisconnect:
        pass
