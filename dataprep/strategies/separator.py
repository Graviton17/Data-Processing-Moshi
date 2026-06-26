"""Source-separation strategy: split a waveform into named stems.

Used for localized music/noise suppression: we keep the ``vocals`` stem and use
the rest only to *measure* how much non-speech energy is present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import AudioBuffer


class SourceSeparator(ABC):
    @abstractmethod
    def separate(self, audio: AudioBuffer) -> dict[str, AudioBuffer]:
        """Return a dict of stem name -> AudioBuffer (e.g. vocals/drums/bass/other)."""


class NullSeparator(SourceSeparator):
    """No-op separator: treats the whole signal as 'vocals'. Used when music
    suppression is disabled."""

    def separate(self, audio: AudioBuffer) -> dict[str, AudioBuffer]:
        return {"vocals": audio}


class DemucsSeparator(SourceSeparator):
    """Wraps Demucs (htdemucs). Model loaded lazily and reused across files."""

    def __init__(self, model: str = "htdemucs", device: str = "cuda"):
        self.model_name = model
        self.device = device
        self._model: Any = None

    def _load(self):
        if self._model is not None:
            return self._model
        from demucs.pretrained import get_model

        model = get_model(self.model_name)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def separate(self, audio: AudioBuffer) -> dict[str, AudioBuffer]:
        import torch
        from demucs.apply import apply_model

        model = self._load()
        # Demucs expects stereo at model.samplerate.
        sr = model.samplerate
        from ..utils.audio_io import resample

        buf = resample(audio, sr) if audio.sample_rate != sr else audio
        wav = torch.from_numpy(buf.samples)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(
                model, wav[None].to(self.device), device=self.device, progress=False
            )[0]
        sources = sources * ref.std() + ref.mean()
        sources = sources.cpu().numpy().astype(np.float32)

        stems: dict[str, AudioBuffer] = {}
        for name, src in zip(model.sources, sources):
            mono = src.mean(axis=0, keepdims=True)
            stems[name] = AudioBuffer(samples=mono, sample_rate=sr)
        return stems
