"""Swappable algorithm implementations (Strategy pattern).

Each interface is an ABC; concrete classes wrap a specific library. Stages
depend on the ABC, not the concrete class, so a different diarizer / separator /
ASR can be dropped in without touching stage code.
"""

from .diarizer import Diarizer, PyannoteDiarizer
from .separator import SourceSeparator, DemucsSeparator, NullSeparator
from .music_detector import MusicDetector, EnergyRatioMusicDetector
from .transcriber import Transcriber, WhisperXTranscriber

__all__ = [
    "Diarizer",
    "PyannoteDiarizer",
    "SourceSeparator",
    "DemucsSeparator",
    "NullSeparator",
    "MusicDetector",
    "EnergyRatioMusicDetector",
    "Transcriber",
    "WhisperXTranscriber",
]
