"""Source-separation strategy: split a waveform into named stems.

Used for localized music/noise suppression: we keep the ``vocals`` stem and use
the rest only to *measure* how much non-speech energy is present.

NOTE: the upstream ``demucs`` package (facebookresearch/demucs) is no longer
maintained.  This file targets ``demucs-infer`` -- a maintained inference-only
fork with PyTorch 2.x support and an identical public API.  Install with:

    pip install demucs-infer>=4.1.2
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
    """Wraps Demucs (htdemucs) via the ``demucs-infer`` package.

    Model is loaded lazily and reused across files.  ``demucs-infer`` is a
    drop-in inference replacement for the unmaintained ``demucs`` package and
    exposes the same ``get_model`` / ``apply_model`` API.
    """

    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "cuda",
        chunk_seconds: float = 30.0,
    ):
        self.model_name = model
        self.device = device
        # Separate the file in time-chunks and offload each to CPU, so GPU memory
        # is bounded by one chunk instead of the whole file. A 68-min file would
        # otherwise need a single ~5.7 GB output tensor (4 stems x 2 ch x len) and
        # OOM a 16 GB T4. demucs already overlap-adds *within* each apply_model
        # call, so only the chunk boundaries lack overlap (negligible for the
        # downstream energy-ratio measurement and vocal substitution).
        self.chunk_seconds = chunk_seconds
        self._model: Any = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            # demucs-infer (maintained fork) -- preferred
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise ImportError(
                "Could not import demucs.pretrained. "
                "Install the maintained inference fork: pip install demucs-infer>=4.1.2"
            ) from exc

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
        # Global normalisation (same stats as before, computed once on CPU).
        ref = wav.mean(0)
        ref_mean = float(ref.mean())
        ref_std = float(ref.std()) + 1e-8
        wav = (wav - ref_mean) / ref_std

        total = wav.shape[1]
        chunk = max(1, int(self.chunk_seconds * sr))
        names = list(model.sources)
        # Accumulate only the mono stems we need (4 x len), not the full
        # 4 x 2 x len tensor, to keep host memory modest too.
        mono_acc = {name: np.empty(total, dtype=np.float32) for name in names}

        with torch.no_grad():
            for start in range(0, total, chunk):
                end = min(total, start + chunk)
                seg = wav[:, start:end][None].to(self.device)
                out = apply_model(model, seg, device=self.device, progress=False)[0]
                out = out * ref_std + ref_mean            # (sources, ch, n)
                out_np = out.cpu().numpy().astype(np.float32)
                for i, name in enumerate(names):
                    mono_acc[name][start:end] = out_np[i].mean(axis=0)
                del seg, out, out_np
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        return {
            name: AudioBuffer(samples=mono_acc[name][None, :], sample_rate=sr)
            for name in names
        }
