"""Stage [5a]: target-speaker extraction -> clean per-speaker channels.

Runs between ``filter`` and ``streamize`` when ``separation.enabled``. It builds
one isolated channel per retained speaker and stashes them on
``ctx.metadata["speaker_channels"]``; StreamizeStage then stacks those into the
stereo variants instead of masking the mono mixture.

Unlike masking, this preserves *overlapping* speech with each channel still
clean -- which is what Moshi (full-duplex) needs and what makes the purity
re-diarization pass.

Two modes (config ``separation.overlap_only``):

* ``overlap_only: true`` (default, cheaper) -- mask the mono for solo speech
  (already clean there) and only run the separator on the spans where both
  hosts talk at once, splicing the extracted audio back in. Bounds GPU cost to
  the overlap (~minutes) instead of the whole file, and avoids separator
  artifacts on clean solo speech.
* ``overlap_only: false`` -- run the extractor over the whole mixture.

Enrollment voice-prints are taken from each speaker's *solo* (non-overlapping)
segments. If a speaker has too little solo speech to enroll reliably, the stage
no-ops and streamize falls back to masking.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import AudioBuffer, PipelineContext
from ..strategies.speech_separator import TargetSpeakerExtractor
from .base import Stage
from .filters import merge_intervals, overcrowded_intervals, subtract


class SeparateSpeakersStage(Stage):
    def __init__(self, extractor: TargetSpeakerExtractor, config: Config):
        super().__init__()
        self.extractor = extractor
        self.config = config

    # -- helpers ---------------------------------------------------------

    def _solo_intervals(self, ctx: PipelineContext, speaker: str) -> list[tuple[float, float]]:
        """Spans where ``speaker`` talks and no one else does."""
        spk = merge_intervals([(s.start, s.end) for s in ctx.segments if s.speaker == speaker])
        overlap = overcrowded_intervals(ctx.segments, 1)  # >1 active speaker
        out: list[tuple[float, float]] = []
        for a, b in spk:
            out.extend(subtract(a, b, overlap))
        return out

    def _enrollment(
        self, mono: np.ndarray, sr: int, intervals: list[tuple[float, float]], max_sec: float
    ) -> AudioBuffer | None:
        cfg = self.config.separation
        budget = int(max_sec * sr)
        pieces: list[np.ndarray] = []
        got = 0
        for a, b in sorted(intervals, key=lambda iv: iv[1] - iv[0], reverse=True):
            i, j = max(0, int(a * sr)), min(len(mono), int(b * sr))
            if j <= i:
                continue
            pieces.append(mono[i:j])
            got += j - i
            if got >= budget:
                break
        if got < int(cfg.enroll_min_sec * sr) or not pieces:
            return None
        return AudioBuffer(samples=np.concatenate(pieces)[None, :], sample_rate=sr)

    def _masked_channel(self, mono: np.ndarray, sr: int, speaker: str, ctx: PipelineContext) -> np.ndarray:
        ch = np.zeros_like(mono)
        n = len(mono)
        for s in ctx.segments:
            if s.speaker != speaker:
                continue
            a, b = max(0, int(s.start * sr)), min(n, int(s.end * sr))
            if b > a:
                ch[a:b] = mono[a:b]
        return ch

    # -- main ------------------------------------------------------------

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        cfg = self.config.separation
        if not cfg.enabled or ctx.audio is None or ctx.main_pair is None:
            return ctx

        sr = ctx.audio.sample_rate
        mono = ctx.audio.channel(0)
        speakers = list(ctx.main_pair)

        enrollments: dict[str, AudioBuffer] = {}
        for spk in speakers:
            enr = self._enrollment(mono, sr, self._solo_intervals(ctx, spk), cfg.enroll_max_sec)
            if enr is None:
                self.log.debug(
                    "%s: speaker %s has <%.1fs solo speech to enroll; "
                    "skipping separation (streamize will mask)",
                    ctx.stem, spk, cfg.enroll_min_sec,
                )
                return ctx
            enrollments[spk] = enr

        if cfg.overlap_only:
            channels = {spk: self._masked_channel(mono, sr, spk, ctx) for spk in speakers}
            overlap = overcrowded_intervals(ctx.segments, 1)
            sep_sec = 0.0
            for a, b in overlap:
                i, j = max(0, int(a * sr)), min(len(mono), int(b * sr))
                if j <= i:
                    continue
                seg = AudioBuffer(samples=mono[i:j][None, :], sample_rate=sr)
                extracted = self.extractor.extract(seg, enrollments)
                for spk in speakers:
                    channels[spk][i:j] = extracted[spk].channel(0)[: j - i]
                sep_sec += (j - i) / sr
            ctx.metadata["separation"] = {"mode": "overlap_only", "separated_sec": round(sep_sec, 2)}
            sep_channels = {spk: AudioBuffer(samples=ch[None, :], sample_rate=sr) for spk, ch in channels.items()}
        else:
            sep_channels = self.extractor.extract(ctx.audio, enrollments)
            ctx.metadata["separation"] = {"mode": "full", "separated_sec": round(len(mono) / sr, 2)}

        ctx.metadata["speaker_channels"] = sep_channels
        return ctx
