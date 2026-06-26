"""Stage [8]: write stereo WAV + transcript JSON per variant.

Active variants go to ``out_dir`` with a sibling ``.json`` in the exact
moshi-finetune format. Variants quarantined by the purity check are written to
``out_dir/<quarantine_dir>`` (WAV only) for inspection.
"""

from __future__ import annotations

import json

from ..config import Config
from ..models import PipelineContext
from ..utils.audio_io import save_wav
from .base import Stage


class WriteOutputsStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        quarantine_dir = ctx.out_dir / self.config.purity.quarantine_dir

        for v in ctx.variants:
            if v.stereo is None:
                continue
            if v.dropped:
                wav_path = quarantine_dir / f"{v.name}.wav"
                save_wav(wav_path, v.stereo)
                v.out_wav = wav_path
                continue

            wav_path = ctx.out_dir / f"{v.name}.wav"
            save_wav(wav_path, v.stereo)
            json_path = wav_path.with_suffix(".json")
            tmp = json_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump({"alignments": v.alignments}, f, ensure_ascii=False)
            tmp.rename(json_path)
            v.out_wav = wav_path
            v.out_json = json_path

        return ctx
