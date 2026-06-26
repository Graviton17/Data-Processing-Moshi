"""Stage [1]: load mono audio @ target SR and diarize into speaker segments."""

from __future__ import annotations

from dataclasses import asdict

from ..config import Config
from ..models import PipelineContext, Segment
from ..strategies import Diarizer
from ..utils.audio_io import load_audio
from ..utils.cache import Cache
from .base import Stage


class DiarizeStage(Stage):
    def __init__(self, diarizer: Diarizer, config: Config, cache: Cache):
        super().__init__()
        self.diarizer = diarizer
        self.config = config
        self.cache = cache

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.audio = load_audio(ctx.raw_path, target_sr=self.config.sample_rate, mono=True)

        cached = self.cache.load_json(ctx.stem, "diarize")
        if cached is not None:
            ctx.segments = [Segment(**s) for s in cached]
        else:
            segments = self.diarizer.diarize(ctx.audio)
            self.cache.save_json(ctx.stem, "diarize", [asdict(s) for s in segments])
            ctx.segments = segments

        ctx.metadata["n_speakers_raw"] = len(ctx.speakers())
        if not ctx.segments:
            ctx.drop("no speech detected by diarizer")
        return ctx
