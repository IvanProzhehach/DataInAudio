"""AcouSteg v6 frame format — shared by encoder, decoder, and duration estimation."""

from __future__ import annotations

import os
from dataclasses import dataclass

PREAMBLE = b"V6"
FOOTER = b"6V"
SYMBOL_CHARS = "01234567"


def bytes_to_octal(data: bytes) -> str:
    bits = "".join(f"{b:08b}" for b in data)
    pad = (3 - len(bits) % 3) % 3
    bits += "0" * pad
    return "".join(str(int(bits[i: i + 3], 2)) for i in range(0, len(bits), 3))


def octal_symbols_for_bytes(nbytes: int) -> int:
    return (nbytes * 8 + 2) // 3


PREAMBLE_SYM = bytes_to_octal(PREAMBLE)
FOOTER_SYM = bytes_to_octal(FOOTER)
PREAMBLE_LEN = len(PREAMBLE_SYM)
FOOTER_LEN = len(FOOTER_SYM)
LENGTH_SYM = octal_symbols_for_bytes(2)
CRC_SYM = octal_symbols_for_bytes(1)


@dataclass
class Config:
    sample_rate:     int   = 44_100
    symbol_duration: float = 0.12
    guard_duration:  float = 0.015
    base_freq:       int   = 18_000
    freq_step:       int   = 200
    num_tones:       int   = 8
    sync_freq:       int   = 17_500
    sync_duration:   float = 0.35
    rs_nsym:         int   = int(os.getenv("ACOUSTEG_RS_NSYM", "12"))

    @property
    def band_low(self) -> int:
        return self.sync_freq - 300

    @property
    def band_high(self) -> int:
        return self.base_freq + (self.num_tones - 1) * self.freq_step + 300


CFG = Config()


def frame_symbol_count(payload_bytes: int) -> int:
    return (PREAMBLE_LEN + LENGTH_SYM + octal_symbols_for_bytes(payload_bytes)
            + CRC_SYM + FOOTER_LEN)


def frame_samples(payload_bytes: int = 32) -> int:
    n_sym = frame_symbol_count(payload_bytes)
    return int(CFG.sample_rate * (
        CFG.sync_duration + CFG.guard_duration
        + n_sym * (CFG.symbol_duration + CFG.guard_duration)
    ))


def frame_duration_s(payload_bytes: int) -> float:
    return frame_samples(payload_bytes) / CFG.sample_rate


def max_frame_samples() -> int:
    return frame_samples(64)
