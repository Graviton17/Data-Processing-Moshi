"""Transcription strategy: a mono speech channel -> word-level alignments.

Output is a list of ``[text, [start, end]]`` word entries; the stage attaches
the speaker label. WhisperX transcribes at 16 kHz internally (its own resample);
we feed it the cleaned channel directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import AudioBuffer


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> list[list]:
        """Return ``[[text, [start, end]], ...]`` word entries."""


class WhisperXTranscriber(Transcriber):
    WHISPER_SR = 16000

    def __init__(
        self,
        model: str = "large-v3",
        language: str = "en",
        align: bool = True,
        device: str = "cuda",
        batch_size: int = 16,
        compute_type: str = "float16",
    ):
        self.model_name = model
        self.language = language
        self.align = align
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self._model: Any = None
        self._align_model: Any = None
        self._align_meta: Any = None

    def _load(self):
        if self._model is not None:
            return
        import whisperx

        self._model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        if self.align:
            self._align_model, self._align_meta = whisperx.load_align_model(
                language_code=self.language, device=self.device
            )

    def transcribe(self, audio: AudioBuffer) -> list[list]:
        import whisperx
        from ..utils.audio_io import resample

        self._load()
        buf = resample(audio, self.WHISPER_SR) if audio.sample_rate != self.WHISPER_SR else audio
        wav = buf.channel(0).astype(np.float32)

        result = self._model.transcribe(wav, batch_size=self.batch_size, language=self.language)
        if self.align and result.get("segments"):
            result = whisperx.align(
                result["segments"], self._align_model, self._align_meta, wav, self.device,
                return_char_alignments=False,
            )

        words: list[list] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue
                words.append([w["word"], [float(w["start"]), float(w["end"])]])
        return words
