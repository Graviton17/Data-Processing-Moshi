"""Stage [2]: localized music/noise suppression.

Intro/outro/standalone music has no speech and is dropped later by the
segment masking, so here we only worry about music *under* speech: within each
diarized speech segment we slide a window, and where the non-vocal energy ratio
exceeds the threshold we substitute the Demucs vocals stem for that window.
Clean windows are left untouched to avoid separation artefacts.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import AudioBuffer, PipelineContext
from ..strategies import MusicDetector, SourceSeparator
from ..strategies.music_detector import EnergyRatioMusicDetector
from ..utils.audio_io import resample
from .base import Stage


class MusicSuppressStage(Stage):
    def __init__(self, separator: SourceSeparator, detector: MusicDetector, config: Config):
        super().__init__()
        self.separator = separator
        self.detector = detector
        self.config = config

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        cfg = self.config.music
        if not cfg.enabled or ctx.audio is None or not ctx.segments:
            return ctx

        stems = self.separator.separate(ctx.audio)
        vocals = stems.get("vocals")
        if vocals is None:
            return ctx

        sr = ctx.audio.sample_rate
        out = ctx.audio.channel(0).copy()
        voc = resample(vocals, sr) if vocals.sample_rate != sr else vocals
        voc_ch = voc.channel(0)

        is_music = (
            self.detector.is_music
            if isinstance(self.detector, EnergyRatioMusicDetector)
            else lambda s, a, b: self.detector.music_ratio(s, a, b) > cfg.music_ratio_threshold
        )

        replaced = 0.0
        win = cfg.analysis_window_sec
        for seg in ctx.segments:
            t = seg.start
            while t < seg.end:
                w1 = min(seg.end, t + win)
                if is_music(stems, t, w1):
                    a = int(t * sr)
                    b = min(len(out), len(voc_ch), int(w1 * sr))
                    if b > a:
                        out[a:b] = voc_ch[a:b]
                        replaced += (b - a) / sr
                t = w1

        ctx.audio = AudioBuffer(samples=out, sample_rate=sr)
        ctx.metadata["music_replaced_sec"] = round(replaced, 2)
        return ctx
