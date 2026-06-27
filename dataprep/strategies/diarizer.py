"""Diarization strategy: mono waveform -> list[Segment]."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from ..models import AudioBuffer, Segment


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, audio: AudioBuffer) -> list[Segment]: ...


class PyannoteDiarizer(Diarizer):
    """Wraps a ``pyannote.audio`` >= 4.0 speaker-diarization pipeline.

    Default model is ``pyannote/speaker-diarization-community-1``, the OSS
    model shipped with pyannote.audio 4.0.  The previous 3.1 model requires
    pyannote.audio 3.x and is incompatible with this class.

    The model is loaded lazily on first use and cached on the instance, so
    constructing the strategy is cheap and import-safe without torch installed.

    Note on output iteration
    ------------------------
    pyannote.audio 4.x changed the pipeline return type.  The result object
    exposes two iteration surfaces:
      * ``output.speaker_diarization``  -- may have overlapping turns
      * ``output.exclusive_speaker_diarization`` -- one speaker per frame
                                                    (easier STT alignment)
    We use ``speaker_diarization`` by default; pass ``exclusive=True`` for
    non-overlapping segments (simpler transcript reconciliation).
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",  # updated: 3.1 -> community-1
        min_speakers: int = 1,
        max_speakers: int = 2,
        hf_token_env: str = "HF_TOKEN",
        device: str = "cuda",
        exclusive: bool = False,  # new: use exclusive_speaker_diarization if True
    ):
        self.model = model
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token_env = hf_token_env
        self.device = device
        self.exclusive = exclusive
        self._pipeline: Any = None

    def _resolve_token(self) -> str | None:
        """Accept either an env-var *name* (preferred) or a literal token.

        A real HF token starts with ``hf_``; anything else is treated as the
        name of an environment variable to read the token from.
        """
        val = self.hf_token_env
        if not val:
            return None
        if val.startswith("hf_"):
            return val  # literal token pasted directly (not recommended)
        return os.environ.get(val)

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline

        token = self._resolve_token()
        # Export the token so downstream hub calls (model weights, embeddings)
        # pick it up automatically without needing an explicit kwarg.
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

        # pyannote.audio >= 4.0 uses `token=` (huggingface_hub >= 1.0).
        # If a stale 3.x install is present, from_pretrained() will either
        # raise TypeError (unknown kwarg) or succeed but return an incompatible
        # Annotation object that lacks .speaker_diarization -- catch both.
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token or None)
        except TypeError as exc:
            raise RuntimeError(
                "Pipeline.from_pretrained() rejected the `token` kwarg. "
                "A stale pyannote.audio 3.x is likely installed (incompatible "
                f"with huggingface_hub >= 1.0): {exc}. "
                "Upgrade and restart the kernel: "
                'pip install -U "pyannote.audio>=4.0"'
            ) from exc

        if pipeline is None:
            raise RuntimeError(
                f"Could not load pyannote pipeline {self.model!r}. "
                "Check the model name and that a valid HuggingFace token is "
                f"available (via {self.hf_token_env!r}). "
                "You must also accept the model licence at "
                "https://hf.co/pyannote/speaker-diarization-community-1"
            )

        pipeline.to(torch.device(self.device))
        self._pipeline = pipeline
        return pipeline

    def diarize(self, audio: AudioBuffer) -> list[Segment]:
        import torch

        pipeline = self._load()

        # pyannote expects shape (channels, samples); AudioBuffer already
        # guarantees 2-D (channels, samples) via __post_init__, so no unsqueeze needed.
        waveform = torch.from_numpy(audio.samples)

        output = pipeline(
            {"waveform": waveform, "sample_rate": audio.sample_rate},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # pyannote.audio 4.x output iteration API (changed from 3.x):
        #   output.speaker_diarization           -> (Segment, speaker) 2-tuples
        #   output.exclusive_speaker_diarization -> same, non-overlapping
        # Old 3.x API was: annotation.itertracks(yield_label=True) -> 3-tuples
        track_iter = (
            output.exclusive_speaker_diarization
            if self.exclusive
            else output.speaker_diarization
        )

        segments = [
            Segment(speaker=str(speaker), start=float(turn.start), end=float(turn.end))
            for turn, speaker in track_iter
        ]
        segments.sort(key=lambda s: s.start)
        return segments
