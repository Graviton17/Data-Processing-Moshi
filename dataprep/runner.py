"""Drive the pipeline over a directory of raw audio files."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .factory import build_pipeline
from .manifest import build_manifest
from .models import PipelineContext
from .pipeline import Pipeline
from .utils.logging import get_logger

log = get_logger("runner")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def iter_audio_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def process_file(pipeline: Pipeline, config: Config, path: Path) -> PipelineContext:
    ctx = PipelineContext(
        raw_path=path,
        sample_rate=config.sample_rate,
        cache_dir=Path(config.cache_dir),
        out_dir=Path(config.out_dir),
    )
    return pipeline.run(ctx)


def process_dir(config: Config, raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(config)
    log.info("Pipeline: %s", pipeline)

    files = iter_audio_files(raw_dir)
    log.info("Found %d audio files under %s", len(files), raw_dir)

    report = {"files": [], "n_kept": 0, "n_dropped": 0, "n_variants": 0, "n_quarantined": 0}
    for i, path in enumerate(files, 1):
        try:
            ctx = process_file(pipeline, config, path)
        except Exception as err:  # keep going on per-file failures
            log.exception("Failed on %s: %s", path, err)
            report["files"].append({"path": str(path), "error": repr(err)})
            continue

        kept = ctx.active_variants()
        quarantined = [v for v in ctx.variants if v.dropped]
        report["n_kept"] += int(not ctx.dropped)
        report["n_dropped"] += int(ctx.dropped)
        report["n_variants"] += len(kept)
        report["n_quarantined"] += len(quarantined)
        report["files"].append(
            {
                "path": str(path),
                "dropped": ctx.dropped,
                "drop_reason": ctx.drop_reason,
                "variants": [v.name for v in kept],
                "quarantined": [v.name for v in quarantined],
                "metadata": {k: ctx.metadata.get(k) for k in
                             ("n_speakers_raw", "music_replaced_sec", "excluded_sec", "crosstalk")},
            }
        )
        if i % 25 == 0:
            log.info("Processed %d/%d files", i, len(files))

    manifest_path = build_manifest(out_dir)
    report["manifest"] = str(manifest_path)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        "Done. kept=%d dropped=%d variants=%d quarantined=%d -> %s",
        report["n_kept"], report["n_dropped"], report["n_variants"],
        report["n_quarantined"], manifest_path,
    )
    return report
