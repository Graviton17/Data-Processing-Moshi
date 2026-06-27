"""Factory: build a fully-wired :class:`Pipeline` from a :class:`Config`.

This is the single place that knows which concrete strategy implements each
interface, so swapping (e.g.) the diarizer is a one-line change here.
"""

from __future__ import annotations

from .config import Config
from .pipeline import Pipeline
from .stages import (
    CleanStage,
    DiarizeStage,
    FilterStage,
    MusicSuppressStage,
    PurityStage,
    StreamizeStage,
    TranscribeStage,
    WriteOutputsStage,
)
from .strategies import (
    DemucsSeparator,
    EnergyRatioMusicDetector,
    NullSeparator,
    PyannoteDiarizer,
    WhisperXTranscriber,
)
from .utils.cache import Cache


def build_pipeline(config: Config) -> Pipeline:
    device = config.device
    cache = Cache(config.cache_dir)

    diarizer = PyannoteDiarizer(
        model=config.diarization.model,
        min_speakers=config.diarization.min_speakers,
        max_speakers=config.diarization.max_speakers,
        hf_token_env=config.diarization.hf_token_env,
        device=device,
        exclusive=config.diarization.exclusive,  # new: forward exclusive flag
    )
    separator = (
        DemucsSeparator(model=config.music.demucs_model, device=device)
        if config.music.enabled
        else NullSeparator()
    )
    detector = EnergyRatioMusicDetector(threshold=config.music.music_ratio_threshold)
    transcriber = WhisperXTranscriber(
        model=config.transcribe.model,
        language=config.transcribe.language,
        align=config.transcribe.align,
        device=device,
        batch_size=config.transcribe.batch_size,
        compute_type=config.transcribe.compute_type,
    )

    stages = [
        DiarizeStage(diarizer, config, cache),
        MusicSuppressStage(separator, detector, config),
        CleanStage(config),
        FilterStage(config),
        StreamizeStage(config),
        PurityStage(diarizer, config),
        TranscribeStage(transcriber, config, cache),
        WriteOutputsStage(config),
    ]
    return Pipeline(stages)
