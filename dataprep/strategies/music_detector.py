"""Music-detection strategy: per-window decision whether to substitute the
separated vocals stem for the original signal.

The default uses the energy ratio of the non-vocal stems to the vocals stem,
computed within speech windows only (intro/outro music with no speech is
already removed by diarization/VAD).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import AudioBuffer
from ..utils.audio_io import energy


class MusicDetector(ABC):
    @abstractmethod
    def music_ratio(self, stems: dict[str, AudioBuffer], start: float, end: float) -> float:
        """Ratio of non-vocal energy to vocal energy within [start, end] seconds."""


class EnergyRatioMusicDetector(MusicDetector):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _window(buf: AudioBuffer, start: float, end: float) -> np.ndarray:
        a = max(0, int(start * buf.sample_rate))
        b = min(buf.num_samples, int(end * buf.sample_rate))
        return buf.channel(0)[a:b] if b > a else np.zeros(0, dtype=np.float32)

    def music_ratio(self, stems: dict[str, AudioBuffer], start: float, end: float) -> float:
        vocals = stems.get("vocals")
        if vocals is None:
            return 0.0
        voc_e = energy(self._window(vocals, start, end)) + 1e-9
        other_e = 0.0
        for name, buf in stems.items():
            if name == "vocals":
                continue
            other_e += energy(self._window(buf, start, end))
        return other_e / voc_e

    def is_music(self, stems: dict[str, AudioBuffer], start: float, end: float) -> bool:
        return self.music_ratio(stems, start, end) > self.threshold
