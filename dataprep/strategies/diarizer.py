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
    """Wraps ``pyannote.audio`` 3.x speaker-diarization pipeline.

    The model is loaded lazily on first use and cached on the instance, so
    constructing the strategy is cheap and import-safe without torch installed.
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        min_speakers: int = 1,
        max_speakers: int = 2,
        hf_token_env: str = "HF_TOKEN",
        device: str = "cuda",
    ):
        self.model = model
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token_env = hf_token_env
        self.device = device
        self._pipeline: Any = None

    def _resolve_token(self) -> str | None:
        """Accept either an env-var *name* (preferred) or a literal token.

        A real token looks like ``hf_...``; anything else is treated as the name
        of an environment variable to read the token from.
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
        # pyannote.audio renamed `use_auth_token` -> `token` in 3.x; fall back
        # for older installs. The legacy kwarg is passed via a dict so static
        # type checkers don't flag the (now-removed) parameter name.
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token)
        except TypeError:
            # Older pyannote.audio (<3.1) uses the legacy `use_auth_token` kwarg.
            pipeline = Pipeline.from_pretrained(self.model, **{"use_auth_token": token})
        if pipeline is None:
            raise RuntimeError(
                f"Could not load pyannote pipeline {self.model!r}. Check the model "
                "name and that a valid HuggingFace token is available "
                f"(via {self.hf_token_env!r})."
            )
        pipeline.to(torch.device(self.device))
        self._pipeline = pipeline
        return pipeline

    def diarize(self, audio: AudioBuffer) -> list[Segment]:
        import torch

        pipeline = self._load()
        waveform = torch.from_numpy(audio.samples)
        annotation = pipeline(
            {"waveform": waveform, "sample_rate": audio.sample_rate},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        segments = [
            Segment(speaker=str(label), start=float(turn.start), end=float(turn.end))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s.start)
        return segments
