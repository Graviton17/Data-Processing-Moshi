"""Stage [6]: channel-purity validation.

Masking guarantees purity *if diarization was correct*. It cannot catch
diarization errors (a span labelled A that actually contains B). To catch those
we independently re-diarize each per-speaker channel: a clean channel should
contain a single dominant speaker. The crosstalk ratio is the fraction of the
channel's speech time attributed to any non-dominant cluster. Variants whose
channels exceed ``max_crosstalk_ratio`` are marked dropped (quarantined).
"""

from __future__ import annotations

from ..config import Config
from ..models import PipelineContext
from ..strategies import Diarizer
from .base import Stage
from .filters import talk_times


class PurityStage(Stage):
    def __init__(self, diarizer: Diarizer, config: Config):
        super().__init__()
        self.diarizer = diarizer
        self.config = config

    def _crosstalk(self, channel_buf) -> float:
        segs = self.diarizer.diarize(channel_buf)
        if not segs:
            return 0.0
        times = talk_times(segs)
        total = sum(times.values())
        if total <= 0:
            return 0.0
        dominant = max(times.values())
        return 1.0 - dominant / total

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        cfg = self.config.purity
        if not cfg.enabled:
            return ctx

        channels = ctx.metadata.get("speaker_channels", {})
        crosstalk = {spk: self._crosstalk(buf) for spk, buf in channels.items()}
        ctx.metadata["crosstalk"] = {k: round(v, 3) for k, v in crosstalk.items()}

        for v in ctx.variants:
            v.purity = {
                "main": crosstalk.get(v.main_speaker, 0.0),
                "user": crosstalk.get(v.user_speaker, 0.0),
            }
            worst = max(v.purity.values())
            if worst > cfg.max_crosstalk_ratio:
                v.dropped = True
                v.drop_reason = f"channel crosstalk {worst:.2f} > {cfg.max_crosstalk_ratio}"

        if not ctx.active_variants():
            ctx.drop("all variants quarantined by purity check")
        return ctx
