"""Typed configuration loaded from YAML.

Uses nested dataclasses so the rest of the code gets attribute access and
IDE completion instead of dictionary spelunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DiarizationConfig:
    # Updated default: community-1 is the OSS model for pyannote.audio >= 4.0.
    # speaker-diarization-3.1 requires pyannote.audio 3.x and will NOT work
    # with the current library version.
    model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: int = 1
    max_speakers: int = 2
    hf_token_env: str = "HF_TOKEN"
    # exclusive=True uses output.exclusive_speaker_diarization (non-overlapping),
    # which simplifies STT reconciliation. False (default) uses speaker_diarization.
    exclusive: bool = False


@dataclass
class MusicConfig:
    enabled: bool = True
    music_ratio_threshold: float = 0.5
    # demucs-infer exposes the same model names as the original demucs package.
    demucs_model: str = "htdemucs"
    analysis_window_sec: float = 1.0
    # Separation is done in time-chunks offloaded to CPU to bound GPU memory on
    # long files. Lower this if you still OOM on a small GPU; raise for fewer
    # chunk-boundary seams on a large GPU.
    separation_chunk_sec: float = 30.0


@dataclass
class FiltersConfig:
    max_speakers_per_chunk: int = 2
    max_turn_seconds: float = 60.0
    turn_merge_gap_sec: float = 0.5
    min_variant_speech_sec: float = 2.0


@dataclass
class CleanupConfig:
    loudness_lufs: float = -23.0
    peak_limit_db: float = -1.0
    pre_emphasis: bool = False  # kept OFF: Mimi is a learned codec on raw audio
    remove_dc_offset: bool = True


@dataclass
class AugmentConfig:
    channel_swap: bool = True


@dataclass
class PurityConfig:
    enabled: bool = True
    max_crosstalk_ratio: float = 0.15
    quarantine_dir: str = "quarantine"
    # enforce=True (default): variants over the crosstalk threshold are dropped
    # and routed to quarantine_dir. enforce=False: keep every variant in the main
    # dataset and only record the crosstalk/pass result as metadata in the output
    # JSON (nothing is dropped for purity reasons).
    enforce: bool = True


@dataclass
class TranscribeConfig:
    model: str = "large-v3"
    language: str = "en"
    align: bool = True
    main_label: str = "SPEAKER_MAIN"
    user_label: str = "SPEAKER_OTHER"
    batch_size: int = 16
    compute_type: str = "float16"


@dataclass
class Config:
    sample_rate: int = 24000
    device: str = "cuda"
    cache_dir: str = ".cache"
    out_dir: str = "dataset"
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    purity: PurityConfig = field(default_factory=PurityConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        nested = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in nested:
                raise ValueError(f"Unknown config key: {key!r}")
            field_type = cls.__dataclass_fields__[key].default_factory  # type: ignore[attr-defined]
            if isinstance(value, dict) and field_type is not None and field_type not in (dict,):
                kwargs[key] = field_type(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
