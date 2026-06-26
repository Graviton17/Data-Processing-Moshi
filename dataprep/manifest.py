"""Build the moshi-finetune ``.jsonl`` manifest from produced outputs.

Each line: ``{"path": "<relative wav>", "duration": <seconds>}``. Only WAVs that
have a sibling ``.json`` transcript are included (quarantined variants are
skipped).
"""

from __future__ import annotations

import json
from pathlib import Path


def build_manifest(out_dir: str | Path, manifest_name: str = "dataset.jsonl") -> Path:
    out_dir = Path(out_dir)
    wavs = sorted(p for p in out_dir.glob("*.wav") if p.with_suffix(".json").exists())

    try:
        import sphn

        durations = sphn.durations([str(p) for p in wavs])
    except Exception:
        import soundfile as sf

        durations = []
        for p in wavs:
            info = sf.info(str(p))
            durations.append(info.frames / info.samplerate)

    manifest_path = out_dir / manifest_name
    with open(manifest_path, "w") as f:
        for wav, dur in zip(wavs, durations):
            if dur is None:
                continue
            rel = wav.relative_to(out_dir)
            f.write(json.dumps({"path": str(rel), "duration": float(dur)}) + "\n")
    return manifest_path
