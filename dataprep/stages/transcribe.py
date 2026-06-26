"""Stage [7]: per-channel transcription + alignment.

Each per-speaker channel is transcribed once; both variants reuse the words and
only differ in the label (SPEAKER_MAIN for the agent channel, SPEAKER_OTHER for
the user channel). The interleaver in moshi-finetune ties the text stream to
SPEAKER_MAIN, so the agent channel must carry the main label.

Output alignments are ``[[text, [start, end], label], ...]`` sorted by start.
"""

from __future__ import annotations

from ..config import Config
from ..models import PipelineContext
from ..strategies import Transcriber
from ..utils.cache import Cache
from .base import Stage


class TranscribeStage(Stage):
    def __init__(self, transcriber: Transcriber, config: Config, cache: Cache):
        super().__init__()
        self.transcriber = transcriber
        self.config = config
        self.cache = cache

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        cfg = self.config.transcribe
        channels = ctx.metadata.get("speaker_channels", {})
        speakers = sorted({v.main_speaker for v in ctx.active_variants()}
                          | {v.user_speaker for v in ctx.active_variants()})

        cached = self.cache.load_json(ctx.stem, "transcribe") or {}
        per_speaker: dict[str, list] = {}
        for spk in speakers:
            if spk in cached:
                per_speaker[spk] = cached[spk]
            else:
                per_speaker[spk] = self.transcriber.transcribe(channels[spk])
        self.cache.save_json(ctx.stem, "transcribe", per_speaker)

        for v in ctx.active_variants():
            alignments = []
            for text, ts in per_speaker.get(v.main_speaker, []):
                alignments.append([text, ts, cfg.main_label])
            for text, ts in per_speaker.get(v.user_speaker, []):
                alignments.append([text, ts, cfg.user_label])
            alignments.sort(key=lambda a: a[1][0])
            v.alignments = alignments
        return ctx
