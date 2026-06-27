"""Build the moshi-finetune ``.jsonl`` manifest from produced outputs.

Each line: ``{"path": "<relative wav>", "duration": <seconds>}``. Only WAVs that
have a sibling ``.json`` transcript are included (quarantined variants are
skipped).

sphn API note
-------------
We pin sphn==0.1.12 (sphn 0.2.0 was never published). Duration is read
per-file via ``sphn.read()`` (returns ``(ndarray, sample_rate)``), which works
on every published sphn release, so no batch helper is needed. The soundfile
fallback path is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path


def _duration_sphn(path: str) -> float:
    """Read duration via sphn (per-file read; works on all published releases)."""
    import sphn
    samples, sr = sphn.read(path)  # returns (ndarray, int)
    # samples shape is (channels, num_samples) or (num_samples,)
    import numpy as np
    arr = np.asarray(samples)
    n = arr.shape[-1]
    return n / sr


def build_manifest(out_dir: str | Path, manifest_name: str = "dataset.jsonl") -> Path:
    out_dir = Path(out_dir)
    wavs = sorted(p for p in out_dir.glob("*.wav") if p.with_suffix(".json").exists())

    manifest_path = out_dir / manifest_name
    if not wavs:
        manifest_path.write_text("")
        return manifest_path

    with open(manifest_path, "w") as f:
        for wav in wavs:
            try:
                try:
                    dur = _duration_sphn(str(wav))
                except Exception:
                    # soundfile fallback (always available as it's in requirements)
                    import soundfile as sf
                    info = sf.info(str(wav))
                    dur = info.frames / info.samplerate

                rel = wav.relative_to(out_dir)
                f.write(json.dumps({"path": str(rel), "duration": float(dur)}) + "\n")
            except Exception:
                # Skip corrupt/unreadable files rather than aborting the manifest.
                continue

    return manifest_path
