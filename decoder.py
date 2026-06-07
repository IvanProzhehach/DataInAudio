"""AcouSteg v6 decoder core for cloud API (no CLI / no sounddevice)."""

from __future__ import annotations

import io
import os
import time
import wave
from typing import Optional

import numpy as np
from scipy.signal import butter, resample, sosfilt, sosfiltfilt

try:
    from pydub import AudioSegment
except ImportError as exc:
    raise ImportError("pip install pydub") from exc

try:
    from reedsolo import RSCodec
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

from codec import (
    CFG,
    CRC_SYM,
    Config,
    FOOTER_LEN,
    FOOTER_SYM,
    LENGTH_SYM,
    PREAMBLE_LEN,
    PREAMBLE_SYM,
    SYMBOL_CHARS,
    frame_samples,
    octal_symbols_for_bytes,
)
SYNC_HOP = 64
SYNC_THRESH = 0.30
AIRPLAY_MODE = False
MIN_PAYLOAD_LEN = 12
REALTIME_SCAN_SEC = 30.0
REALTIME_SCAN_INTERVAL = 1.0
MAX_DECODE_SECONDS = float(os.getenv("ACOUSTEG_MAX_DECODE_SEC", "90"))
MAX_UPLOAD_BYTES = int(os.getenv("ACOUSTEG_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

_SOS: Optional[np.ndarray] = None


def configure_decoder(*, airplay: bool = True) -> None:
    global SYNC_THRESH, AIRPLAY_MODE
    AIRPLAY_MODE = airplay
    SYNC_THRESH = 0.20 if airplay else 0.30


def _octal_to_bytes(symbols: str, nbytes: Optional[int] = None) -> bytes:
    bits = "".join(f"{int(c):03b}" for c in symbols)
    if nbytes is not None:
        bits = bits[:nbytes * 8]
    else:
        bits = bits[: len(bits) // 8 * 8]
    return bytes(int(bits[i: i + 8], 2) for i in range(0, len(bits), 8))


def crc8_raw(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
            crc &= 0xFF
    return crc


def _plausible_payload(text: str) -> bool:
    if len(text) < MIN_PAYLOAD_LEN:
        return False
    if not all(c.isprintable() and c != "\x00" for c in text):
        return False
    if text.startswith("http") and "://" not in text:
        return False
    return True


class ToneCache:
    def __init__(self, cfg: Config):
        n_sym = int(cfg.sample_rate * cfg.symbol_duration)
        n_sync = int(cfg.sample_rate * cfg.sync_duration)
        win_sym = np.hanning(n_sym)
        win_sync = np.hanning(n_sync)
        t_sym = np.linspace(0, cfg.symbol_duration, n_sym, endpoint=False)
        t_sync = np.linspace(0, cfg.sync_duration, n_sync, endpoint=False)

        self._tones_norm: dict[str, np.ndarray] = {}
        for i in range(cfg.num_tones):
            freq = cfg.base_freq + i * cfg.freq_step
            t = (np.sin(2 * np.pi * freq * t_sym) * win_sym).astype(np.float64)
            self._tones_norm[str(i)] = t / (np.linalg.norm(t) + 1e-12)

        sync = (np.sin(2 * np.pi * cfg.sync_freq * t_sync) * win_sync).astype(np.float64)
        self._sync_norm = sync / (np.linalg.norm(sync) + 1e-12)
        self._guard = np.zeros(int(cfg.sample_rate * cfg.guard_duration))

    def tone_norm(self, ch: str) -> np.ndarray:
        return self._tones_norm[ch]

    def sync_norm(self) -> np.ndarray:
        return self._sync_norm

    def tone_len(self) -> int:
        return len(next(iter(self._tones_norm.values())))

    def sync_len(self) -> int:
        return len(self._sync_norm)

    def guard_len(self) -> int:
        return len(self._guard)


_CACHE: Optional[ToneCache] = None


def _get_cache() -> ToneCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ToneCache(CFG)
    return _CACHE


def _tone_freq(idx: int) -> float:
    return CFG.base_freq + idx * CFG.freq_step


def _goertzel_power(chunk: np.ndarray, freq: float, sr: int) -> float:
    n = len(chunk)
    if n == 0:
        return 0.0
    k = int(0.5 + n * freq / sr)
    w = 2 * np.pi * k / n
    c = 2 * np.cos(w)
    s0 = s1 = s2 = 0.0
    for x in chunk:
        s0 = x + c * s1 - s2
        s2, s1 = s1, s0
    return max(s2 * s2 + s1 * s1 - c * s1 * s2, 1e-20)


def _get_sos() -> np.ndarray:
    global _SOS
    if _SOS is None:
        nyq = 0.5 * CFG.sample_rate
        _SOS = butter(6, [CFG.band_low / nyq, CFG.band_high / nyq], btype="band", output="sos")
    return _SOS


def bandpass(audio: np.ndarray, *, fast: bool = False) -> np.ndarray:
    """fast=True: single-pass sosfilt (API / long files). Default: sosfiltfilt (higher quality)."""
    x = audio.astype(np.float64, copy=False)
    sos = _get_sos()
    if fast or len(x) > CFG.sample_rate * 45:
        return sosfilt(sos, x)
    return sosfiltfilt(sos, x)


def wav_info(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        return {
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "frames": frames,
            "duration_s": round(frames / sr, 3) if sr else 0.0,
        }


def _ncc_dot(chunk: np.ndarray, ref_norm: np.ndarray) -> float:
    n = min(len(chunk), len(ref_norm))
    c = chunk[:n]
    nc = np.linalg.norm(c)
    if nc < 1e-12:
        return 0.0
    return float(abs(np.dot(c / nc, ref_norm[:n])))


def max_sync_score(audio: np.ndarray) -> tuple[float, int]:
    cache = _get_cache()
    sync_len = cache.sync_len()
    ref = cache.sync_norm()
    if len(audio) < sync_len:
        return 0.0, -1
    best_score, best_pos = 0.0, 0
    for start in range(0, len(audio) - sync_len, SYNC_HOP):
        score = _ncc_dot(audio[start: start + sync_len], ref)
        if score > best_score:
            best_score, best_pos = score, start
    return best_score, best_pos


def _room_samples() -> int:
    """Min audio tail after sync for typical payloads (mic / airplay)."""
    return frame_samples(48)


def find_sync_positions(audio: np.ndarray, *, require_room: bool = False) -> list[int]:
    cache = _get_cache()
    sync_len = cache.sync_len()
    ref = cache.sync_norm()
    need = _room_samples() if require_room else 0
    positions: list[int] = []
    best_pos, best_score = -sync_len, 0.0
    for start in range(0, len(audio) - sync_len, SYNC_HOP):
        score = _ncc_dot(audio[start: start + sync_len], ref)
        if score > best_score:
            best_score, best_pos = score, start
        if start - best_pos > sync_len // 2:
            if best_score >= SYNC_THRESH and (not require_room or best_pos + need <= len(audio)):
                positions.append(best_pos)
            best_score, best_pos = 0.0, start
    if best_score >= SYNC_THRESH and (not require_room or best_pos + need <= len(audio)):
        positions.append(best_pos)
    return positions


def _estimate_eq(work: np.ndarray, sync_start: int) -> np.ndarray:
    cache = _get_cache()
    sr = CFG.sample_rate
    sym_len = cache.tone_len()
    step = sym_len + cache.guard_len()
    p0 = sync_start + cache.sync_len() + cache.guard_len()
    measured = np.zeros(CFG.num_tones, dtype=np.float64)
    counts = np.zeros(CFG.num_tones, dtype=np.float64)
    for si, ch in enumerate(PREAMBLE_SYM):
        idx = int(ch)
        chunk = work[p0 + si * step: p0 + si * step + sym_len]
        measured[idx] += _goertzel_power(chunk, _tone_freq(idx), sr)
        counts[idx] += 1.0
    known = [i for i in range(CFG.num_tones) if counts[i] > 0]
    if not known:
        return np.ones(CFG.num_tones)
    xs = np.array(known, dtype=np.float64)
    ys = measured[known] / counts[known]
    for i in range(CFG.num_tones):
        if counts[i] == 0:
            measured[i] = float(np.interp(i, xs, ys))
        else:
            measured[i] /= counts[i]
    ref = float(np.median(measured[known]))
    return ref / np.maximum(measured, ref * 0.01)


def decode_symbols(work: np.ndarray, sync_start: int, eq: Optional[np.ndarray]) -> str:
    cache = _get_cache()
    sym_len = cache.tone_len()
    step = sym_len + cache.guard_len()
    pos = sync_start + cache.sync_len() + cache.guard_len()
    out: list[str] = []
    max_sym = PREAMBLE_LEN + LENGTH_SYM + octal_symbols_for_bytes(256) + CRC_SYM + FOOTER_LEN
    for _ in range(max_sym):
        if pos + sym_len > len(work):
            break
        chunk = work[pos: pos + sym_len]
        if AIRPLAY_MODE and eq is not None:
            best_ch, best = "0", -1.0
            for i in range(CFG.num_tones):
                pw = _goertzel_power(chunk, _tone_freq(i), CFG.sample_rate) * eq[i]
                if pw > best:
                    best, best_ch = pw, str(i)
            out.append(best_ch)
        else:
            best_ch, best = "0", 0.0
            for ch in SYMBOL_CHARS:
                s = _ncc_dot(chunk, cache.tone_norm(ch))
                if s > best:
                    best, best_ch = s, ch
            out.append(best_ch)
        pos += step
    return "".join(out)


def verify_symbols(symbols: str) -> Optional[str]:
    if not symbols.startswith(PREAMBLE_SYM):
        return None
    if len(symbols) < PREAMBLE_LEN + LENGTH_SYM + CRC_SYM + FOOTER_LEN:
        return None
    if not symbols.endswith(FOOTER_SYM):
        idx = symbols.find(FOOTER_SYM, PREAMBLE_LEN)
        if idx == -1:
            return None
        symbols = symbols[: idx + FOOTER_LEN]

    p = PREAMBLE_LEN
    length_bytes = _octal_to_bytes(symbols[p: p + LENGTH_SYM], 2)
    if len(length_bytes) != 2:
        return None
    plen = int.from_bytes(length_bytes, "big")
    if plen <= 0 or plen > 512:
        return None

    p += LENGTH_SYM
    payload_sym_len = octal_symbols_for_bytes(plen)
    if len(symbols) < p + payload_sym_len + CRC_SYM + FOOTER_LEN:
        return None
    payload = _octal_to_bytes(symbols[p: p + payload_sym_len], plen)
    if len(payload) != plen:
        return None
    p += payload_sym_len
    crc_recv = _octal_to_bytes(symbols[p: p + CRC_SYM], 1)
    if len(crc_recv) != 1 or crc_recv[0] != crc8_raw(payload):
        if not AIRPLAY_MODE:
            return None
        if abs(crc_recv[0] - crc8_raw(payload)) > 1:
            return None

    attempts: list[bytes] = [payload]
    if _RS_AVAILABLE:
        try:
            attempts.insert(0, bytes(RSCodec(CFG.rs_nsym).decode(payload)[0]))
        except Exception:
            pass
    for raw in attempts:
        if not raw:
            continue
        try:
            text = raw.decode("utf-8")
            if _plausible_payload(text):
                return text
        except Exception:
            continue
    return None


def decode_at(work: np.ndarray, sync_start: int) -> Optional[str]:
    cache = _get_cache()
    sym_len = cache.tone_len()
    step = sym_len + cache.guard_len()
    min_tail = cache.sync_len() + cache.guard_len() + step * (PREAMBLE_LEN + LENGTH_SYM + 4)
    if sync_start < 0 or sync_start + min_tail > len(work):
        return None

    if not AIRPLAY_MODE:
        return verify_symbols(decode_symbols(work, sync_start, None))

    def _try(offset: int) -> Optional[str]:
        pos = sync_start + offset
        if pos < 0 or pos + min_tail > len(work):
            return None
        eq = _estimate_eq(work, pos)
        return verify_symbols(decode_symbols(work, pos, eq))

    hit = _try(0)
    if hit:
        return hit
    best = None
    coarse = 0
    for offset in range(-2400, 2401, 256):
        hit = _try(offset)
        if hit and (best is None or len(hit) > len(best)):
            best, coarse = hit, offset
            if len(hit) >= 19:
                return hit
    for offset in range(coarse - 300, coarse + 301, 64):
        hit = _try(offset)
        if hit and (best is None or len(hit) > len(best)):
            best = hit
            if len(hit) >= 19:
                return hit
    return best


def decode_audio(audio: np.ndarray, sample_rate: int, *, realtime: bool = False) -> list[dict]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != CFG.sample_rate:
        audio = resample(audio, int(len(audio) * CFG.sample_rate / sample_rate)).astype(np.float64)
    work = bandpass(audio, fast=True)
    syncs = find_sync_positions(work, require_room=realtime or AIRPLAY_MODE)
    candidates = syncs if (realtime or AIRPLAY_MODE) else sorted({0} | set(syncs))
    results: list[dict] = []
    seen: set[str] = set()
    for pos in candidates:
        text = decode_at(work, pos)
        if text and text not in seen:
            seen.add(text)
            results.append({"time_s": round(pos / CFG.sample_rate, 3), "payload": text})
    if AIRPLAY_MODE and results and not realtime:
        return [max(results, key=lambda r: len(r["payload"]))]
    return results


def load_audio_bytes(data: bytes, *, max_seconds: Optional[float] = None) -> tuple[np.ndarray, int, dict]:
    """Load mono float64 audio; truncate to max_seconds for cloud safety."""
    limit = max_seconds if max_seconds is not None else MAX_DECODE_SECONDS
    meta: dict = {"truncated": False, "duration_s": None, "used_seconds": None}

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            total = wf.getnframes()
            meta["duration_s"] = round(total / sr, 3) if sr else 0.0
            take = total
            if sr > 0 and limit > 0:
                take = min(total, int(limit * sr))
                meta["truncated"] = take < total
            meta["used_seconds"] = round(take / sr, 3) if sr else 0.0
            frames = wf.readframes(take)
            if sw == 1:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
                audio = (audio - 128.0) / 128.0
            elif sw == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
            elif sw == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2**31
            else:
                raise ValueError(f"unsupported sample width: {sw}")
            if nch > 1:
                audio = audio.reshape(-1, nch).mean(axis=1)
            return audio, sr, meta
    except (wave.Error, ValueError):
        seg = (AudioSegment.from_file(io.BytesIO(data))
               .set_frame_rate(CFG.sample_rate)
               .set_channels(1)
               .set_sample_width(4))
        dur_ms = len(seg)
        meta["duration_s"] = round(dur_ms / 1000.0, 3)
        if limit > 0 and dur_ms > limit * 1000:
            seg = seg[: int(limit * 1000)]
            meta["truncated"] = True
        meta["used_seconds"] = round(len(seg) / 1000.0, 3)
        audio = np.frombuffer(seg.raw_data, dtype=np.int32).astype(np.float64) / 2**31
        return audio, CFG.sample_rate, meta


def decode_bytes(data: bytes, *, realtime: bool = False, airplay: Optional[bool] = None) -> tuple[list[dict], dict]:
    prev_airplay = AIRPLAY_MODE
    if airplay is not None:
        configure_decoder(airplay=airplay)
    try:
        audio, sr, meta = load_audio_bytes(data)
        return decode_audio(audio, sr, realtime=realtime), meta
    finally:
        if airplay is not None:
            configure_decoder(airplay=prev_airplay)


class StreamSession:
    """Rolling-buffer decoder for WebSocket PCM chunks."""

    def __init__(self, *, airplay: bool = True, repeat: bool = False):
        configure_decoder(airplay=airplay)
        self.sr = CFG.sample_rate
        cache = _get_cache()
        self._scan_sec = max(REALTIME_SCAN_SEC, _room_samples() / CFG.sample_rate + 5)
        self._buf_len = int(self._scan_sec * self.sr) + cache.sync_len()
        self._min_samples = _room_samples()
        self._tile_interval = _room_samples() / CFG.sample_rate
        self._buf = np.zeros(self._buf_len, dtype=np.float64)
        self._seen: set[str] = set()
        self._last_decode_mono = 0.0
        self._cooldown = 0.0
        self._last_scan = 0.0
        self._repeat = repeat
        self._input_sr = self.sr
        self.total_samples = 0

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._seen.clear()
        self._last_decode_mono = 0.0
        self._cooldown = 0.0
        self._last_scan = 0.0
        self.total_samples = 0

    def set_input_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._input_sr = sample_rate

    @property
    def buffer_seconds(self) -> float:
        return self._scan_sec

    def push_pcm(self, chunk: np.ndarray) -> None:
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        mono = chunk.astype(np.float64, copy=False)
        if self._input_sr != self.sr:
            n_out = max(1, int(len(mono) * self.sr / self._input_sr))
            mono = resample(mono, n_out).astype(np.float64)
        n = len(mono)
        self.total_samples += n
        if n >= self._buf_len:
            self._buf[:] = mono[-self._buf_len:]
            return
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = mono

    def scan(self) -> list[dict]:
        now = time.monotonic()
        if now < self._cooldown or now - self._last_scan < REALTIME_SCAN_INTERVAL:
            return []
        self._last_scan = now
        window = self._buf.copy()
        if len(window) < self._min_samples:
            return []
        sync_score, sync_pos = max_sync_score(bandpass(window, fast=True))
        out: list[dict] = []
        for hit in decode_audio(window, self.sr, realtime=True):
            p = hit["payload"]
            if self._repeat:
                if now - self._last_decode_mono < self._tile_interval * 0.7:
                    continue
                self._last_decode_mono = now
            elif p in self._seen:
                continue
            else:
                self._seen.add(p)
            self._cooldown = now + 1.0
            out.append({
                **hit,
                "sync_score": round(sync_score, 3),
                "sync_pos_s": round(sync_pos / self.sr, 3) if sync_pos >= 0 else None,
            })
        return out

    def status(self) -> dict:
        sync_score, sync_pos = max_sync_score(bandpass(self._buf, fast=True))
        filled = min(self.total_samples, self._buf_len)
        return {
            "sync_score": round(sync_score, 3),
            "sync_threshold": SYNC_THRESH,
            "sync_pos_s": round(sync_pos / self.sr, 3) if sync_pos >= 0 else None,
            "buffer_seconds": self._scan_sec,
            "filled_seconds": round(filled / self.sr, 2),
            "sample_rate": self.sr,
            "input_sample_rate": self._input_sr,
        }
