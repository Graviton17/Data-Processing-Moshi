"""Drive the pipeline over a directory of raw audio files.

Multi-GPU is handled by **data parallelism**: the file list is sharded across the
available GPUs and one worker process is pinned to each device (via
``CUDA_VISIBLE_DEVICES``). On Kaggle's 2x T4 this gives ~2x throughput with no
model changes. Outputs are keyed by file stem so shards never collide, and the
manifest is built once after all workers finish.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config
from .factory import build_pipeline
from .manifest import build_manifest
from .models import PipelineContext
from .pipeline import Pipeline
from .utils.logging import get_logger, init_logging

log = get_logger("runner")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def iter_audio_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def detect_num_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _empty_report() -> dict:
    return {"files": [], "n_kept": 0, "n_dropped": 0, "n_variants": 0, "n_quarantined": 0}


def _merge_reports(reports: list[dict]) -> dict:
    merged = _empty_report()
    for r in reports:
        merged["files"].extend(r["files"])
        for k in ("n_kept", "n_dropped", "n_variants", "n_quarantined"):
            merged[k] += r[k]
    return merged


def process_file(pipeline: Pipeline, config: Config, path: Path) -> PipelineContext:
    ctx = PipelineContext(
        raw_path=path,
        sample_rate=config.sample_rate,
        cache_dir=Path(config.cache_dir),
        out_dir=Path(config.out_dir),
    )
    return pipeline.run(ctx)


def _process_files(
    pipeline: Pipeline, config: Config, files: list[Path], tag: str = ""
) -> dict:
    """Run the pipeline over a list of files and return a partial report."""
    report = _empty_report()
    total = len(files)
    for i, path in enumerate(files, 1):
        try:
            ctx = process_file(pipeline, config, path)
        except Exception as err:  # keep going on per-file failures
            log.exception("[%s] failed on %s: %s", tag, path, err)
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
                "metadata": {
                    k: ctx.metadata.get(k)
                    for k in ("n_speakers_raw", "music_replaced_sec", "excluded_sec", "crosstalk")
                },
            }
        )
        if i % 25 == 0:
            log.info("[%s] processed %d/%d files", tag, i, total)
    return report


def _run_shard(gpu_id: int, config: Config, files: list[Path], queue) -> None:
    """Worker entrypoint: pin to one GPU and process this shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    init_logging()
    config.device = "cuda"  # the single device now visible to this process
    pipeline = build_pipeline(config)
    log.info("GPU %d: %d files | %s", gpu_id, len(files), pipeline)
    report = _process_files(pipeline, config, files, tag=f"gpu{gpu_id}")
    queue.put(report)


def _process_multigpu(config: Config, files: list[Path], num_gpus: int) -> dict:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    shards = [files[i::num_gpus] for i in range(num_gpus)]

    procs = []
    for gpu_id, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(target=_run_shard, args=(gpu_id, config, shard, queue))
        p.start()
        procs.append(p)

    reports = [queue.get() for _ in procs]  # drain before join to avoid deadlock
    for p in procs:
        p.join()
    return _merge_reports(reports)


def process_dir(config: Config, raw_dir: str | Path, num_gpus: int | None = None) -> dict:
    """Process every audio file under ``raw_dir``.

    ``num_gpus``: None -> auto-detect and use all GPUs; 1 -> single process;
    N -> use N GPUs.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_audio_files(raw_dir)
    log.info("Found %d audio files under %s", len(files), raw_dir)

    if num_gpus is None:
        num_gpus = detect_num_gpus()
    use_gpus = num_gpus if num_gpus and num_gpus > 1 else 1

    # Parallelism is data-parallel *by file*: a shard per GPU. With fewer files
    # than GPUs (e.g. a single long file) the extra GPUs sit idle -- to use them,
    # supply more files (or pre-split the long file). actual_gpus reflects the
    # path actually taken, not the number of GPUs detected.
    if use_gpus > 1 and len(files) > 1:
        actual_gpus = min(use_gpus, len(files))
        log.info("Running data-parallel across %d GPUs", actual_gpus)
        report = _process_multigpu(config, files, use_gpus)
    else:
        actual_gpus = 1
        pipeline = build_pipeline(config)
        if use_gpus > 1 and len(files) <= 1:
            log.info(
                "Running single-process: %d file(s) but %d GPUs detected -- "
                "file-sharded parallelism needs >=2 files to use multiple GPUs | %s",
                len(files), use_gpus, pipeline,
            )
        else:
            log.info("Running single-process | %s", pipeline)
        report = _process_files(pipeline, config, files, tag="single")

    manifest_path = build_manifest(out_dir)
    report["manifest"] = str(manifest_path)
    report["num_gpus_used"] = actual_gpus
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        "Done. kept=%d dropped=%d variants=%d quarantined=%d (gpus=%d) -> %s",
        report["n_kept"], report["n_dropped"], report["n_variants"],
        report["n_quarantined"], actual_gpus, manifest_path,
    )
    return report
