"""Stage [8]: write stereo WAV + enriched transcript JSON per variant.

Every variant gets a stereo WAV (channel 0 = main/agent speaker, channel 1 =
the other speaker) and a sibling ``.json``. The JSON is **moshi-finetune
compatible** -- the interleaver reads ``data["alignments"]`` and ignores any
extra keys -- but we also store, for inspection and downstream use:

  * ``alignments``            -- moshi format: ``[[word, [start, end], label], ...]``
                                with both speakers, labelled SPEAKER_MAIN / SPEAKER_OTHER
  * ``segments``              -- the raw diarization: ``[{speaker, start, end}, ...]``
  * ``transcript_by_speaker`` -- the word stream split per speaker (user 1 vs user 2),
                                each ``[[word, [start, end]], ...]``
  * ``speakers``             -- which diarization speaker id maps to main vs user
  * ``purity``               -- per-channel crosstalk + pass/quarantine flag

This stage runs even on dropped contexts so quarantined variants (purity
``enforce=True``) still get a full inspection dump. Quarantined variants are
written under ``out_dir/<quarantine_dir>`` so the manifest (which scans only the
top level of out_dir) excludes them from the training set.
"""

from __future__ import annotations

import json

from ..config import Config
from ..models import PipelineContext
from ..utils.audio_io import save_wav
from .base import Stage


class WriteOutputsStage(Stage):
    # Write whatever variants exist even if the file was dropped, so quarantined
    # variants are still persisted for inspection.
    runs_on_dropped = True

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        tcfg = self.config.transcribe
        main_label, user_label = tcfg.main_label, tcfg.user_label
        quarantine_dir = ctx.out_dir / self.config.purity.quarantine_dir

        segments = [
            {"speaker": s.speaker, "start": s.start, "end": s.end} for s in ctx.segments
        ]

        for v in ctx.variants:
            if v.stereo is None:
                continue

            target_dir = quarantine_dir if v.dropped else ctx.out_dir
            wav_path = target_dir / f"{v.name}.wav"
            save_wav(wav_path, v.stereo)
            v.out_wav = wav_path

            payload = {
                # --- consumed by moshi-finetune's interleaver ---
                "alignments": v.alignments,
                # --- inspection / downstream extras (ignored by the interleaver) ---
                "segments": segments,
                "transcript_by_speaker": {
                    main_label: [[t, ts] for t, ts, lbl in v.alignments if lbl == main_label],
                    user_label: [[t, ts] for t, ts, lbl in v.alignments if lbl == user_label],
                },
                "speakers": {"main": v.main_speaker, "user": v.user_speaker},
                "purity": {
                    "main_crosstalk": v.purity.get("main"),
                    "user_crosstalk": v.purity.get("user"),
                    "pass": v.purity_pass,
                    "quarantined": v.dropped,
                },
            }

            json_path = wav_path.with_suffix(".json")
            tmp = json_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.rename(json_path)
            v.out_json = json_path

        return ctx
