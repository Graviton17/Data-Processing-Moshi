"""Stage [5]: build the two-stream stereo from the cleaned mono mixture.

For each retained speaker we mask the mono mixture to that speaker's segments
(silence elsewhere). Moshi expects L=agent/main, R=user. With channel-swap
augmentation we emit both orderings so the model is not biased to one position.

Per-speaker masked channels are cached on the context so the transcription stage
can transcribe each one *once* and the two variants just relabel them.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import AudioBuffer, PipelineContext, StreamVariant
from .base import Stage


class StreamizeStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _build_channel(self, mono: np.ndarray, sr: int, speaker: str, segments) -> np.ndarray:
        ch = np.zeros_like(mono)
        n = len(mono)
        for s in segments:
            if s.speaker != speaker:
                continue
            a = max(0, int(s.start * sr))
            b = min(n, int(s.end * sr))
            if b > a:
                ch[a:b] = mono[a:b]
        return ch

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.audio is None or ctx.main_pair is None:
            ctx.drop("streamize: missing audio or speaker pair")
            return ctx

        sr = ctx.audio.sample_rate
        mono = ctx.audio.channel(0)
        spk_a, spk_b = ctx.main_pair

        # If SeparateSpeakersStage already produced clean per-speaker channels
        # (target-speaker extraction), reuse them -- they keep overlapped speech
        # clean, which masking can't. Otherwise fall back to masking the mono.
        sep = ctx.metadata.get("speaker_channels")
        if sep and spk_a in sep and spk_b in sep:
            channels = {spk_a: sep[spk_a].channel(0), spk_b: sep[spk_b].channel(0)}
        else:
            channels = {
                spk_a: self._build_channel(mono, sr, spk_a, ctx.segments),
                spk_b: self._build_channel(mono, sr, spk_b, ctx.segments),
            }
            ctx.metadata["speaker_channels"] = {
                spk: AudioBuffer(samples=ch, sample_rate=sr) for spk, ch in channels.items()
            }

        orderings = [(spk_a, spk_b)]
        if self.config.augment.channel_swap:
            orderings.append((spk_b, spk_a))

        for main, user in orderings:
            stereo = np.stack([channels[main], channels[user]], axis=0)
            ctx.variants.append(
                StreamVariant(
                    name=f"{ctx.stem}__{main}_agent",
                    main_speaker=main,
                    user_speaker=user,
                    stereo=AudioBuffer(samples=stereo, sample_rate=sr),
                )
            )
        return ctx
