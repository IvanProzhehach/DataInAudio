"""
AcouSteg v6 — Decode API for Render.

  GET  /health
  POST /api/decode
  POST /api/encode-duration
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

from codec import CFG
from decoder import StreamSession, configure_decoder, decode_bytes
from encode_duration import check_encode_fit, estimate_encode_duration

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


class EncodeDurationRequest(BaseModel):
    text: str
    audio_duration_s: Optional[float] = None


class EncodeDurationResponse(BaseModel):
    ok: bool
    duration_s: float
    symbols: int
    payload_bytes: int
    utf8_bytes: int
    fits: Optional[bool] = None
    audio_duration_s: Optional[float] = None
    shortfall_s: Optional[float] = None
    tiles: Optional[int] = None
    message: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "decoder": "acousteg-v6",
        "sample_rate": CFG.sample_rate,
        "airplay_band_hz": [CFG.band_low, CFG.band_high],
    }


@app.post("/api/encode-duration", response_model=EncodeDurationResponse)
def encode_duration(req: EncodeDurationRequest) -> EncodeDurationResponse:
    text = req.text.strip()
    if not text:
        return EncodeDurationResponse(
            ok=False,
            duration_s=0.0,
            symbols=0,
            payload_bytes=0,
            utf8_bytes=0,
            message="text must not be empty",
        )
    try:
        if req.audio_duration_s is not None:
            info = check_encode_fit(text, req.audio_duration_s)
        else:
            info = estimate_encode_duration(text)
    except Exception as exc:
        return EncodeDurationResponse(
            ok=False,
            duration_s=0.0,
            symbols=0,
            payload_bytes=0,
            utf8_bytes=0,
            message=str(exc),
        )
    return EncodeDurationResponse(
        ok=True,
        message=None if info.get("fits", True) else (
            f"audio too short: need {info['duration_s']}s, got {info['audio_duration_s']}s"
        ),
        **info,
    )


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
