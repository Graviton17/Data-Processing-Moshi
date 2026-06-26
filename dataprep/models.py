"""Domain models passed between pipeline stages (Data Transfer Objects).

These are deliberately framework-agnostic: no torch / whisperx types leak in
here, so the models import cleanly and can be unit-tested without the heavy ML
dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AudioBuffer:
    """A waveform plus its sample rate.

    ``samples`` is always 2-D with shape ``(channels, num_samples)`` so that
    mono and stereo are handled uniformly.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[None, :]
        if self.samples.ndim != 2:
            raise ValueError(
                f"AudioBuffer expects (channels, samples), got shape {self.samples.shape}"
            )

    @property
    def num_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    def channel(self, idx: int) -> np.ndarray:
        return self.samples[idx]


@dataclass
class Segment:
    """A diarized span of speech attributed to a single speaker."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end


@dataclass
class Word:
    """A transcribed word with timing, attributed to a stream label."""

    text: str
    start: float
    end: float
    speaker: str


@dataclass
class StreamVariant:
    """One channel-assignment of a conversation (Moshi expects stereo: L=agent).

    The channel-swap augmentation produces two variants per input file: one with
    speaker A as the agent and one with speaker B as the agent.
    """

    name: str
    main_speaker: str          # -> left channel (channel 0), labelled SPEAKER_MAIN
    user_speaker: str          # -> right channel (channel 1)
    stereo: AudioBuffer | None = None
    alignments: list[list[Any]] = field(default_factory=list)  # interleaver format
    out_wav: Path | None = None
    out_json: Path | None = None
    dropped: bool = False
    drop_reason: str | None = None
    purity: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable state threaded through every stage for a single input file."""

    raw_path: Path
    sample_rate: int
    cache_dir: Path
    out_dir: Path

    audio: AudioBuffer | None = None
    segments: list[Segment] = field(default_factory=list)
    excluded: list[tuple[float, float]] = field(default_factory=list)
    main_pair: tuple[str, str] | None = None
    variants: list[StreamVariant] = field(default_factory=list)

    dropped: bool = False
    drop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.raw_path.stem

    def drop(self, reason: str) -> None:
        self.dropped = True
        self.drop_reason = reason

    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})

    def active_variants(self) -> list[StreamVariant]:
        return [v for v in self.variants if not v.dropped]
