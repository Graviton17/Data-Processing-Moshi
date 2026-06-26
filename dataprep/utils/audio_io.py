"""Audio I/O and DSP helpers.

Loading uses ``sphn`` (same lib moshi-finetune uses) when available, falling
back to ``soundfile``. Heavy imports are local so this module stays importable
for the pure-logic helpers (energy / RMS) even without audio libs installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import AudioBuffer


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> AudioBuffer:
    """Load an audio file as an :class:`AudioBuffer`, optionally resampling."""
    path = str(path)
    try:
        import sphn

        samples, sr = sphn.read(path)  # (channels, samples), float32
    except Exception:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True)  # (samples, channels)
        samples = data.T.astype(np.float32)

    if mono and samples.shape[0] > 1:
        samples = samples.mean(axis=0, keepdims=True)

    buf = AudioBuffer(samples=samples.astype(np.float32), sample_rate=sr)
    if target_sr is not None and sr != target_sr:
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
