"""
AcouSteg v6 — Decode API for Render.

  GET  /health
  POST /api/decode          — full-file decode (upload WAV)
  POST /api/decode-mic      — fast chunked mic decode (~20 s)
  POST /api/encode-duration
  POST /api/shorten         — (optional proxy, unused by default frontend)

Local: uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from codec import CFG
from decoder import (
    API_BUILD,
    CLOUD_FAST,
    MAX_DECODE_SECONDS,
    MAX_UPLOAD_BYTES,
    configure_decoder,
    decode_bytes,
    decode_mic_chunk,
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
    sync_score: Optional[float] = None


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
            "decode": f"POST /api/decode  (multipart: file, max {MAX_UPLOAD_BYTES // 1024 // 1024}MB)",
            "decode_mic": "POST /api/decode-mic  (multipart: file, ~20s mic chunk, fast airplay)",
            "encode_duration": "POST /api/encode-duration  (json: {text, audio_duration_s?})",
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


async def _run_decode(fn, timeout: float) -> tuple[list[dict], dict]:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_decode_pool, fn),
        timeout=timeout,
    )


@app.post("/api/decode", response_model=DecodeResponse)
async def decode_upload(
    file: UploadFile = File(...),
    airplay: bool = Query(True, description="Speaker→mic decode (decoder_v6 --airplay)"),
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
        frames, meta = await _run_decode(
            lambda: decode_bytes(data, realtime=False, airplay=airplay),
            timeout=float(os.getenv("ACOUSTEG_DECODE_TIMEOUT_SEC", "55")),
        )
    except asyncio.TimeoutError:
        return DecodeResponse(
            ok=False, frames=[],
            message=f"decode timeout — use first {MAX_DECODE_SECONDS:.0f}s of audio or shorter file",
        )
    except MemoryError:
        return DecodeResponse(ok=False, frames=[], message="out of memory — file too long")
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
            sync_score=meta.get("sync_score"),
        )
    return DecodeResponse(
        ok=True, frames=frames, message=note,
        duration_s=meta.get("duration_s"), used_seconds=meta.get("used_seconds"),
        truncated=meta.get("truncated"),
    )


@app.post("/api/decode-mic", response_model=DecodeResponse)
async def decode_mic_upload(file: UploadFile = File(...)) -> DecodeResponse:
    """Fast path for browser mic chunks (~20 s WAV)."""
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
        frames, meta = await _run_decode(
            lambda: decode_mic_chunk(data),
            timeout=float(os.getenv("ACOUSTEG_MIC_TIMEOUT_SEC", "40")),
        )
    except asyncio.TimeoutError:
        return DecodeResponse(
            ok=False, frames=[],
            message="mic decode timeout — shorten the recording chunk",
        )
    except Exception as exc:
        return DecodeResponse(ok=False, frames=[], message=str(exc))

    sync_score = meta.get("sync_score")
    if not frames:
        msg = f"no frame (sync={sync_score}, need ≥{0.2})" if sync_score is not None else "no v6 frame found"
        return DecodeResponse(
            ok=True, frames=[], message=msg,
            duration_s=meta.get("duration_s"), used_seconds=meta.get("used_seconds"),
            truncated=meta.get("truncated"), sync_score=sync_score,
        )
    return DecodeResponse(
        ok=True, frames=frames,
        duration_s=meta.get("duration_s"), used_seconds=meta.get("used_seconds"),
        truncated=meta.get("truncated"), sync_score=sync_score,
    )
