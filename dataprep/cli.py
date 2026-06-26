"""Command-line entrypoint.

    python -m dataprep.cli --raw path/to/raw_audio --config config.yaml

Or build just the manifest from an already-processed output dir:

    python -m dataprep.cli --manifest-only --out dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .manifest import build_manifest
from .runner import process_dir
from .utils.logging import init_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Moshi two-stream dataset builder")
    parser.add_argument("--raw", type=Path, help="Directory of raw mono audio files")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--out", type=Path, default=None, help="Override output directory")
    parser.add_argument("--cache", type=Path, default=None, help="Override cache directory")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPUs to use. Default: auto-detect and use all "
                             "(data-parallel sharding). Set 1 to force single process.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only (re)build the manifest from --out")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    init_logging(args.verbose)

    config = Config.from_yaml(args.config) if args.config else Config()
    if args.out:
        config.out_dir = str(args.out)
    if args.cache:
        config.cache_dir = str(args.cache)
    if args.device:
        config.device = args.device

    if args.manifest_only:
        path = build_manifest(config.out_dir)
        print(f"Wrote manifest: {path}")
        return

    if not args.raw:
        parser.error("--raw is required unless --manifest-only is set")
    process_dir(config, args.raw, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
