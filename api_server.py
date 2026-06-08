"""
AcouSteg v6 — Decode API for Render.

  GET  /health
  POST /api/decode
  POST /api/encode-duration
  WS   /ws/decode

Local: uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("acousteg.ws")

from codec import CFG
from decoder import (
    API_BUILD,
    CLOUD_FAST,
    MAX_DECODE_SECONDS,
    MAX_UPLOAD_BYTES,
    StreamSession,
    configure_decoder,
    decode_bytes,
)
from encode_duration import check_encode_fit, estimate_encode_duration

configure_decoder(airplay=True)

_decode_pool = ThreadPoolExecutor(max_workers=2)

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
    duration_s: Optional[float] = None
    used_seconds: Optional[float] = None
    truncated: Optional[bool] = None


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


@app.get("/")
def root() -> dict:
    return {
        "service": "AcouSteg v6 Decode API",
        "endpoints": {
            "health": "GET /health",
            "decode": f"POST /api/decode  (multipart field: file, max {MAX_UPLOAD_BYTES // 1024 // 1024}MB, "
                      f"first {MAX_DECODE_SECONDS:.0f}s decoded)",
            "encode_duration": "POST /api/encode-duration  (json: {text, audio_duration_s?})",
            "live_decode": "WS /ws/decode",
            "docs": "GET /docs",
        },
        "limits": {
            "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
            "max_decode_sec": MAX_DECODE_SECONDS,
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "decoder": "acousteg-v6",
        "build": API_BUILD,
        "cloud_fast": CLOUD_FAST,
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
async def decode_upload(
    file: UploadFile = File(...),
    airplay: bool = Query(True, description="Speaker→mic decode (same as decoder_v6 --airplay)"),
) -> DecodeResponse:
    data = await file.read()
    if not data:
        return DecodeResponse(ok=False, frames=[], message="empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // 1024 // 1024
        return DecodeResponse(
            ok=False, frames=[],
            message=f"file too large ({len(data) / 1024 / 1024:.1f} MB, max {mb} MB)",
        )
    try:
        loop = asyncio.get_running_loop()
        frames, meta = await asyncio.wait_for(
            loop.run_in_executor(
                _decode_pool,
                lambda: decode_bytes(data, realtime=False, airplay=airplay),
            ),
            timeout=float(os.getenv("ACOUSTEG_DECODE_TIMEOUT_SEC", "55")),
        )
    except asyncio.TimeoutError:
        return DecodeResponse(
            ok=False, frames=[],
            message=f"decode timeout — use first {MAX_DECODE_SECONDS:.0f}s of audio or shorter file",
        )
    except MemoryError:
        return DecodeResponse(ok=False, frames=[], message="out of memory — file too long for free tier")
    except Exception as exc:
        return DecodeResponse(ok=False, frames=[], message=str(exc))

    note = None
    if meta.get("truncated"):
        note = (
            f"decoded first {meta['used_seconds']}s of {meta['duration_s']}s "
            f"(limit {MAX_DECODE_SECONDS:.0f}s on cloud)"
        )
    if not frames:
        return DecodeResponse(
            ok=True, frames=[], message=note or "no v6 frame found",
            duration_s=meta.get("duration_s"), used_seconds=meta.get("used_seconds"),
            truncated=meta.get("truncated"),
        )
    return DecodeResponse(
        ok=True, frames=frames, message=note,
        duration_s=meta.get("duration_s"), used_seconds=meta.get("used_seconds"),
        truncated=meta.get("truncated"),
    )


@app.websocket("/ws/decode")
async def ws_decode(ws: WebSocket) -> None:
    await ws.accept()
    session = StreamSession(airplay=True)
    loop = asyncio.get_running_loop()
    pcm_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
    stop = asyncio.Event()

    await ws.send_json({
        "type": "ready",
        "sample_rate": CFG.sample_rate,
        "buffer_seconds": session.buffer_seconds,
        "chunk_format": "float32_le_mono",
        "build": API_BUILD,
    })

    scan_timeout = float(os.getenv("ACOUSTEG_SCAN_TIMEOUT_SEC", "30"))

    async def run_scan() -> None:
        """Heavy DSP. Runs as its own task so draining never blocks on it."""
        window = session.snapshot()
        try:
            hits = await asyncio.wait_for(
                loop.run_in_executor(_decode_pool, session.scan_window, window),
                timeout=scan_timeout,
            )
            for hit in hits:
                await ws.send_json({"type": "decoded", **hit})
            try:
                await ws.send_json({"type": "scan_status", **session.status(light=True)})
            except Exception:  # noqa: BLE001
                pass
        except asyncio.TimeoutError:
            logger.warning("scan_window exceeded %.0fs — skipped", scan_timeout)
        except Exception as exc:  # noqa: BLE001 — a bad scan must not kill the loop
            logger.exception("scan_window failed")
            try:
                await ws.send_json({"type": "scan_error", "message": str(exc)})
            except Exception:  # noqa: BLE001 — client may already be gone
                pass

    async def scanner_loop() -> None:
        """Drain PCM into the rolling buffer; trigger scans without ever
        blocking on them. The buffer keeps filling no matter how slow or
        broken a scan is.
        """
        scan_task: Optional[asyncio.Task] = None
        while not stop.is_set():
            while not pcm_queue.empty():
                session.push_pcm(pcm_queue.get_nowait())
            scan_running = scan_task is not None and not scan_task.done()
            if not scan_running and session.should_scan():
                scan_task = asyncio.create_task(run_scan())
                scan_task.add_done_callback(_log_task_error)
            await asyncio.sleep(0.05)

    def _log_task_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("background task died", exc_info=exc)

    scanner_task = asyncio.create_task(scanner_loop())
    scanner_task.add_done_callback(_log_task_error)

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
                    while not pcm_queue.empty():
                        pcm_queue.get_nowait()
                    sr = int(payload.get("sample_rate", CFG.sample_rate))
                    session.set_input_sample_rate(sr)
                    await ws.send_json({"type": "started", "sample_rate": sr, "build": API_BUILD})
                elif msg_type in ("ping", "status"):
                    await ws.send_json({
                        "type": "pong" if msg_type == "ping" else "status",
                        **session.status(light=True),
                    })
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
            try:
                pcm_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Drop oldest chunk rather than block the receive loop.
                try:
                    pcm_queue.get_nowait()
                    pcm_queue.put_nowait(chunk)
                except asyncio.QueueEmpty:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        scanner_task.cancel()
        try:
            await scanner_task
        except asyncio.CancelledError:
            pass
