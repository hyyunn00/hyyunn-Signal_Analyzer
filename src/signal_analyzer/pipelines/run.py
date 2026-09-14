"""Top-level pipeline dispatcher + CLI: picks ``run_with_registration`` or
``run_without_registration`` based on the loaded config's
``brain.needs_registration``, so callers (and the CLI) don't need to know
which brain type they're looking at ahead of time.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ..common.config import RunConfig, load_config
from .run_with_registration import run_with_registration
from .run_without_registration import run_without_registration


def run_pipeline(
    config: RunConfig,
    output_dir: Optional[str | Path] = None,
    roi_acronyms: Optional[list[str]] = None,
    functional_system: Optional[dict] = None,
) -> dict:
    """Run the appropriate pipeline for ``config.brain.needs_registration``.

    ``roi_acronyms``/``functional_system`` are ignored (with a warning) for
    brain type B, since neither applies without an ``annotation.tif``.
    """
    if config.brain.needs_registration:
        return run_with_registration(config, output_dir, roi_acronyms, functional_system)

    if roi_acronyms or functional_system:
        logging.getLogger(__name__).warning(
            "brain '%s' has needs_registration=False (no annotation.tif) -- "
            "ignoring roi_acronyms/functional_system, which require registration.",
            config.brain.id,
        )
    return run_without_registration(config, output_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Signal_Analyzer pipeline for one brain, per its config file.",
    )
    parser.add_argument("config", type=Path, help="Path to the per-brain/run YAML config.")
    parser.add_argument("--defaults", type=Path, default=None, help="Optional lab-wide defaults YAML to merge under the config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config.paths.output_dir.")
    parser.add_argument("--roi", action="append", default=None, help="Allen CCF acronym to extract via roi_extract (repeatable). Registered brains only.")
    parser.add_argument(
        "--functional-system", type=Path, default=None,
        help="Path to a single functional-system entry as YAML/JSON (keywords/acronyms/exact_acronym_match/cell_threshold). Registered brains only.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")

    config = load_config(args.config, defaults_path=args.defaults)

    functional_system = None
    if args.functional_system:
        import yaml
        with open(args.functional_system, "r", encoding="utf-8") as f:
            functional_system = yaml.safe_load(f)

    results = run_pipeline(config, output_dir=args.output_dir, roi_acronyms=args.roi, functional_system=functional_system)

    print(json.dumps(
        {name: {k: str(v) for k, v in stage.items()} for name, stage in results.items()},
        indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
