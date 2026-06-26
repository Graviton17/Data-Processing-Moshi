"""Audio I/O and DSP helpers.

Loading tries ``sphn`` (same lib moshi-finetune uses), then ``soundfile``, then
falls back to **ffmpeg**, which decodes formats libsndfile can't (mp3/m4a/webm/
opus, or files with a misleading ``.wav`` extension). Heavy imports are local so
this module stays importable for the pure-logic helpers even without audio libs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..models import AudioBuffer


def _load_libsndfile(path: str) -> AudioBuffer:
    try:
        import sphn

        samples, sr = sphn.read(path)  # (channels, samples), float32
        return AudioBuffer(samples=np.asarray(samples, dtype=np.float32), sample_rate=sr)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True)  # (samples, channels)
        return AudioBuffer(samples=data.T.astype(np.float32), sample_rate=sr)


def _ffprobe(path: str) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def _load_ffmpeg(path: str, target_sr: int | None, mono: bool) -> AudioBuffer:
    """Decode any ffmpeg-readable file to float32 PCM. Applies mono/resample
    inline, so the returned buffer already honours ``target_sr``/``mono``."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found; cannot decode this file. Install ffmpeg "
            "or convert the file to a standard PCM WAV first."
        )
    src_sr, src_ch = _ffprobe(path)
    out_sr = target_sr or src_sr
    out_ch = 1 if mono else src_ch
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(out_ch), "-ar", str(out_sr), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype=np.float32).copy()
    if out_ch > 1:
        data = data.reshape(-1, out_ch).T  # (channels, samples)
    else:
        data = data[None, :]
    return AudioBuffer(samples=data, sample_rate=out_sr)


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> AudioBuffer:
    """Load an audio file as an :class:`AudioBuffer`, optionally resampling."""
    path = str(path)
    try:
        buf = _load_libsndfile(path)
    except Exception:
        # ffmpeg fallback already applies mono + resample.
        return _load_ffmpeg(path, target_sr, mono)

    if mono and buf.num_channels > 1:
        buf = AudioBuffer(samples=buf.samples.mean(axis=0, keepdims=True), sample_rate=buf.sample_rate)
    if target_sr is not None and buf.sample_rate != target_sr:
        buf = resample(buf, target_sr)
    return buf


def resample(buf: AudioBuffer, target_sr: int) -> AudioBuffer:
    if buf.sample_rate == target_sr:
        return buf
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(buf.samples)
    out = AF.resample(t, buf.sample_rate, target_sr)
    return AudioBuffer(samples=out.numpy().astype(np.float32), sample_rate=target_sr)


def save_wav(path: str | Path, buf: AudioBuffer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sphn

        sphn.write_wav(str(path), buf.samples, buf.sample_rate)
    except Exception:
        import soundfile as sf

        sf.write(str(path), buf.samples.T, buf.sample_rate)


# --- pure DSP helpers (no heavy deps) -------------------------------------

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x, dtype=np.float64)))


def slice_samples(buf: AudioBuffer, start: float, end: float, channel: int = 0) -> np.ndarray:
    a = max(0, int(round(start * buf.sample_rate)))
    b = min(buf.num_samples, int(round(end * buf.sample_rate)))
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return buf.channel(channel)[a:b]


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x) if x.size else x


def peak_limit(x: np.ndarray, ceiling_db: float) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > ceiling and peak > 0:
        x = x * (ceiling / peak)
    return x


def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Classic pre-emphasis filter. Provided for completeness only; it is OFF by
    default because Mimi is a learned codec trained on raw waveforms and this
    filter shifts the signal away from that distribution."""
    if x.size == 0:
        return x
    return np.append(x[0], x[1:] - coeff * x[:-1])


def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """EBU R128 loudness normalisation. Silence is gated out by the meter, so a
    mostly-silent masked channel is normalised by its speech loudness."""
    if x.size == 0:
        return x
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x.astype(np.float64))
    if not np.isfinite(loudness):
        return x
    return pyln.normalize.loudness(x, loudness, target_lufs).astype(np.float32)
