"""Stage [3]: light cleanup of the mono mixture (shared gain for both streams).

Cleaning the mono source *before* masking means the agent/user channels keep
their relative loudness. NO pre-emphasis (Mimi is a learned raw-waveform codec).
"""

from __future__ import annotations

from ..config import Config
from ..models import AudioBuffer, PipelineContext
from ..utils.audio_io import loudness_normalize, peak_limit, pre_emphasis, remove_dc
from .base import Stage


class CleanStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.audio is None:
            return ctx
        cfg = self.config.cleanup
        sr = ctx.audio.sample_rate
        x = ctx.audio.channel(0).copy()

        if cfg.remove_dc_offset:
            x = remove_dc(x)
        if cfg.pre_emphasis:  # off by default; warn if a user turns it on
            self.log.warning("pre_emphasis is enabled — not recommended for Mimi")
            x = pre_emphasis(x)
        x = loudness_normalize(x, sr, cfg.loudness_lufs)
        x = peak_limit(x, cfg.peak_limit_db)

        ctx.audio = AudioBuffer(samples=x, sample_rate=sr)
        return ctx
