"""Estimate how long a v6 acoustic frame is for a given payload text."""

from __future__ import annotations

import math

from codec import CFG, frame_duration_s, frame_symbol_count

try:
    from reedsolo import RSCodec
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False


def payload_bytes_for_text(text: str) -> int:
    raw = text.encode("utf-8")
    if _RS_AVAILABLE:
        raw = bytes(RSCodec(CFG.rs_nsym).encode(raw))
    return len(raw)


def estimate_encode_duration(text: str) -> dict:
    """Return the minimum audio length (seconds) for one complete v6 frame."""
    if not text:
        raise ValueError("text must not be empty")
    payload_b = payload_bytes_for_text(text)
    return {
        "duration_s": round(frame_duration_s(payload_b), 3),
        "symbols": frame_symbol_count(payload_b),
        "payload_bytes": payload_b,
        "utf8_bytes": len(text.encode("utf-8")),
    }


def check_encode_fit(text: str, audio_duration_s: float) -> dict:
    """
    Check whether an audio track is long enough to embed the payload.

    The encoder tiles one frame across the carrier; at least one full frame
    must fit inside the audio (audio_duration_s >= duration_s).
    """
    if audio_duration_s <= 0:
        raise ValueError("audio_duration_s must be positive")
    info = estimate_encode_duration(text)
    block_s = info["duration_s"]
    fits = audio_duration_s >= block_s
    shortfall = max(0.0, block_s - audio_duration_s)
    tiles = math.ceil(audio_duration_s / block_s) if block_s > 0 else 0
    return {
        **info,
        "fits": fits,
        "audio_duration_s": round(audio_duration_s, 3),
        "shortfall_s": round(shortfall, 3),
        "tiles": tiles,
    }
