"""Tiny on-disk cache so the slow stages (Demucs/diarization/Whisper) don't
re-run on every invocation.

Each cache entry is keyed by ``(stem, stage_name)`` and stored as JSON. Stages
decide what to persist; numpy arrays should be saved separately as .npy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, stem: str, stage: str, suffix: str = ".json") -> Path:
        d = self.root / stem
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stage}{suffix}"

    def has(self, stem: str, stage: str, suffix: str = ".json") -> bool:
        return self._path(stem, stage, suffix).exists()

    def load_json(self, stem: str, stage: str) -> Any | None:
        p = self._path(stem, stage)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def save_json(self, stem: str, stage: str, data: Any) -> None:
        p = self._path(stem, stage)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.rename(p)

    def path_for(self, stem: str, stage: str, suffix: str) -> Path:
        return self._path(stem, stage, suffix)
