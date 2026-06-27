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
        # huggingface_hub resolves auth in this order: explicit `token=` arg ->
        # HF_TOKEN env var -> cached `hf auth login`. Export the token to the env
        # too so the rest of the stack (model + embedding downloads) picks it up.
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        # pyannote.audio >= 4.0 takes the modern `token=` kwarg and forwards it to
        # huggingface_hub. Passing None is fine -- the hub then falls back to the
        # HF_TOKEN env / cached login. Older pyannote (3.x) only knows
        # `use_auth_token` and is incompatible with huggingface_hub >= 1.0; if such
        # a stale build is installed, `token=` raises TypeError -- surface a fix.
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token or None)
        except TypeError as exc:
            raise RuntimeError(
                "Pipeline.from_pretrained() rejected the `token` kwarg, which means "
                "a stale pyannote.audio 3.x is installed (incompatible with "
                f"huggingface_hub >= 1.0): {exc}. Upgrade and RESTART the kernel: "
                'pip install -U "pyannote.audio>=4.0"'
            ) from exc
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
